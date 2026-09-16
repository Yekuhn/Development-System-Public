"""
engineering_orchestrator_agent.py

Purpose
-------
A minimal, scalable engineering orchestrator for Ascendant Path that enables
easy team expansion from 1 → N engineers via a pull-based work queue.

Key design choices
------------------
- Engineering Lead produces a role-agnostic task graph: work_items[] (DAG).
- Engineers are identical "workers" that CLAIM tasks from a shared queue.
- Orchestrator enforces:
    - dependency gating (DAG)
    - claim locks (no duplicate work)
    - optional file/module locks (reduce parallel collisions)
    - claim TTL (auto-recover from crashed workers)
- No need for "nt" parameter or a static team_info doc.

This file does NOT require different engineer prompts/files per headcount.
You can copy-paste the same engineer agent code and just run more workers.
"""

from __future__ import annotations

# -------------------------------------------------------------------
# Agent metadata (for registry/logging; NOT an LLM prompt)
# -------------------------------------------------------------------

ORCH_NAME = "Engineering Orchestrator"
ORCH_VERSION = "0.1.0"

ORCH_ROLE_DESCRIPTION = """\
You are a deterministic orchestration component that coordinates engineering
execution at runtime.

You do NOT generate code and you do NOT 'think' like an LLM agent.
Your job is to provide safe parallelism and correct sequencing by:
- ingesting the Engineering Lead plan (work_items[])
- allowing engineers to CLAIM tasks (pull-based)
- enforcing dependency gating (DAG) and claim/file locks
- recording results and updating task status

This design enables easy scaling from 1 to N engineers by simply running more
identical workers, without changing prompts or feeding team-size parameters.
"""

SUPPORTED_COMMANDS = [
    "ingest_plan",
    "claim_next",
    "submit_result",
    "get_state",
    "get_results",
    "list_blocked",
    "unblock_task",
    "release_task",
    "recover_orphaned_claims",
    "reset",
]
# A lightweight command contract for callers (Coordinator/router).
COMMAND_CONTRACT = {
    "ingest_plan": {"eng_lead_output": "dict"},
    "claim_next": {"engineer_id": "str", "engineer_capabilities": "list[str] (optional)"},
    "submit_result": {
        "result": {
            "task_id": "str",
            "engineer_id": "str",
            "changes": "any",
            "verification_run": "list[str]",
            "notes": "str (optional)",
            "handoff_interfaces": "list[str] (optional)",
        }
    },
}

import json
import os
import time
import threading
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple, Callable, Iterable


# -----------------------------
# Data contracts (work + result)
# -----------------------------

@dataclass(frozen=True)
class WorkItem:
    task_id: str
    summary: str
    capabilities_required: List[str]
    dependencies: List[str]
    scope_in: str
    scope_out: str
    interfaces: List[str]
    acceptance_criteria: List[str]
    verification: List[str]
    files_expected: List[str]
    risk_notes: str


@dataclass
class WorkItemRuntime:
    """Runtime state for a WorkItem."""
    status: str = "todo"  # todo | claimed | done | blocked
    claimed_by: Optional[str] = None
    claimed_at: Optional[float] = None
    claim_expires_at: Optional[float] = None
    notes: str = ""


@dataclass(frozen=True)
class WorkResult:
    task_id: str
    engineer_id: str
    changes: Any  # patch/diff/file bundle/etc. Keep flexible.
    verification_run: List[str]  # what was run + results (as lines).
    notes: str = ""
    handoff_interfaces: List[str] = field(default_factory=list)


# -----------------------------
# Task queue / lock manager
# -----------------------------

class InMemoryTaskQueue:
    """
    In-memory orchestration state.

    For production/multi-process, swap the state backend to Redis/Postgres.
    The interface here is intentionally simple to keep that migration easy.
    """

    def __init__(
        self,
        *,
        claim_ttl_seconds: int = 30 * 60,
        enable_file_locks: bool = True,
        state_path: Optional[str] = None,
    ) -> None:
        self.claim_ttl_seconds = int(claim_ttl_seconds)
        self.enable_file_locks = bool(enable_file_locks)
        self.state_path = state_path

        self._mu = threading.Lock()

        self._work_items: Dict[str, WorkItem] = {}
        self._runtime: Dict[str, WorkItemRuntime] = {}

        # Locks prevent parallel edits in the same area. They are advisory but effective.
        # lock_key -> task_id that holds it
        self._locks: Dict[str, str] = {}

        # Optional: store submitted results (last write wins per task_id)
        self._results: Dict[str, Dict[str, Any]] = {}

        if self.state_path:
            self._load_state_best_effort()

    # ------------------
    # Public API
    # ------------------

    def ingest_eng_lead_output(self, eng_lead_payload: Dict[str, Any], *, preserve_existing_state: bool = False) -> None:
        """
        Accepts Engineering Lead output (the JSON from eng_lead_agent_core.py),
        specifically the 'work_items' array.

        By default this resets the queue to match the provided plan. When
        preserve_existing_state=True and the persisted queue already matches
        the same task ids, runtime/results/locks are retained for resume.
        """
        work_items = eng_lead_payload.get("work_items", None)
        if not isinstance(work_items, list):
            raise ValueError("eng_lead_payload must contain work_items: []")

        parsed: Dict[str, WorkItem] = {}
        for wi in work_items:
            parsed_item = self._parse_work_item(wi)
            if parsed_item.task_id in parsed:
                raise ValueError(f"Duplicate task_id in work_items: {parsed_item.task_id}")
            parsed[parsed_item.task_id] = parsed_item

        with self._mu:
            existing_ids = set(self._work_items.keys())
            incoming_ids = set(parsed.keys())
            if preserve_existing_state and existing_ids == incoming_ids:
                self._work_items = parsed
                for tid in parsed.keys():
                    self._runtime.setdefault(tid, WorkItemRuntime(status="todo"))
                # Remove stale locks/results only if they reference impossible task ids.
                self._locks = {lk: holder for lk, holder in self._locks.items() if holder in incoming_ids}
                self._results = {tid: res for tid, res in self._results.items() if tid in incoming_ids}
            else:
                self._work_items = parsed
                self._runtime = {tid: WorkItemRuntime(status="todo") for tid in parsed.keys()}
                self._locks = {}
                self._results = {}
            self._reconcile_blocked_dependencies_locked()
            self._persist_state_locked()

    def claim_next(
        self,
        *,
        engineer_id: str,
        engineer_capabilities: Optional[Iterable[str]] = None,
        allow_partial_capability_match: bool = True,
    ) -> Optional[WorkItem]:
        """
        Pull-based claim: return a WorkItem the engineer should execute next.

        Capability routing:
          - If engineer_capabilities is None/empty: treat as generalist.
          - If provided: prefer tasks whose capabilities match.
          - If allow_partial_capability_match is True: use overlap scoring.
            Otherwise require all task capabilities to be present.

        Returns None when no claimable task exists.
        """
        caps = set([c.strip() for c in (engineer_capabilities or []) if str(c).strip()])
        now = time.time()

        with self._mu:
            self._expire_stale_claims_locked(now)

            candidates: List[Tuple[int, WorkItem]] = []
            for task_id, item in self._work_items.items():
                rt = self._runtime.get(task_id)
                if rt is None:
                    continue
                if rt.status != "todo":
                    continue
                if not self._deps_satisfied_locked(item.dependencies):
                    continue
                if self.enable_file_locks and not self._locks_available_locked(item, now):
                    continue

                score = self._capability_score(item, caps, allow_partial_capability_match)
                if score < 0:
                    continue
                candidates.append((score, item))

            if not candidates:
                return None

            # Highest score first; stable tie-break by task_id for determinism.
            candidates.sort(key=lambda x: (-x[0], x[1].task_id))
            chosen = candidates[0][1]

            self._claim_locked(chosen.task_id, engineer_id, now)
            self._persist_state_locked()
            return chosen

    def submit_result(
        self,
        *,
        result: WorkResult,
        mark_done: bool = True,
        mark_blocked: bool = False,
        blocked_reason: Optional[str] = None,
    ) -> None:
        """
        Submit an engineer's result. This updates queue state and releases locks.

        - mark_done: set task to 'done' if True (default)
        - mark_blocked: set task to 'blocked' if True (overrides mark_done)
        """
        if mark_done and mark_blocked:
            raise ValueError("mark_done and mark_blocked cannot both be True.")

        now = time.time()

        with self._mu:
            if result.task_id not in self._work_items:
                raise ValueError(f"Unknown task_id: {result.task_id}")

            rt = self._runtime[result.task_id]
            if rt.status not in ("claimed", "todo"):
                # Allow late results; keep strict by default.
                raise ValueError(f"Task {result.task_id} is not claimable/submittable; status={rt.status}")

            # Save result
            self._results[result.task_id] = {
                "task_id": result.task_id,
                "engineer_id": result.engineer_id,
                "changes": result.changes,
                "verification_run": list(result.verification_run),
                "notes": result.notes,
                "handoff_interfaces": list(result.handoff_interfaces),
                "submitted_at": now,
            }

            # Release locks held by this task (if any)
            self._release_locks_locked(result.task_id)

            # Update status
            if mark_blocked:
                rt.status = "blocked"
                rt.notes = blocked_reason or "blocked"
            elif mark_done:
                rt.status = "done"
                rt.notes = "done"
            else:
                # keep in todo if not done; useful if you want iterative submissions
                rt.status = "todo"
                rt.notes = "returned_to_queue"

            rt.claimed_by = None
            rt.claimed_at = None
            rt.claim_expires_at = None

            # Recompute any tasks that became blocked due to missing deps
            self._reconcile_blocked_dependencies_locked()
            self._persist_state_locked()

    def get_state(self) -> Dict[str, Any]:
        with self._mu:
            return {
                "summary": self._summary_locked(),
                "work_items": {tid: asdict(wi) for tid, wi in self._work_items.items()},
                "runtime": {tid: asdict(rt) for tid, rt in self._runtime.items()},
                "locks": dict(self._locks),
                "results": dict(self._results),
            }


    def list_blocked(self) -> List[Dict[str, Any]]:
        """Return blocked tasks with runtime notes."""
        with self._mu:
            out: List[Dict[str, Any]] = []
            for tid, rt in self._runtime.items():
                if rt.status == "blocked":
                    out.append({"task_id": tid, "notes": rt.notes})
            return out

    def unblock_task(self, *, task_id: str, note: str = "resources_attached") -> bool:
        """
        Move a blocked task back to todo.

        Defensive behavior: if a previous worker crashed after creating a human or
        resource request, the task can be left in `claimed` even though the run is
        waiting for an external unblock. In that case, release the orphaned claim
        and return the task to todo as well. Done tasks are never changed.
        """
        with self._mu:
            rt = self._runtime.get(task_id)
            if rt is None:
                return False
            if rt.status == "done":
                return False
            if rt.status not in {"blocked", "claimed", "todo"}:
                return False
            self._release_locks_locked(task_id)
            rt.status = "todo"
            rt.notes = str(note or "unblocked")
            rt.claimed_by = None
            rt.claimed_at = None
            rt.claim_expires_at = None
            self._persist_state_locked()
            return True

    def release_task(self, *, task_id: str, note: str = "manual_release", only_if_claimed_by: Optional[str] = None) -> bool:
        """Release one claimed task back to todo without marking it done/blocked."""
        with self._mu:
            rt = self._runtime.get(task_id)
            if rt is None or rt.status != "claimed":
                return False
            if only_if_claimed_by and str(rt.claimed_by or "") != str(only_if_claimed_by):
                return False
            self._release_locks_locked(task_id)
            rt.status = "todo"
            rt.notes = str(note or "released")
            rt.claimed_by = None
            rt.claimed_at = None
            rt.claim_expires_at = None
            self._persist_state_locked()
            return True

    def recover_orphaned_claims(self, *, note: str = "orphaned_claim_recovered") -> List[Dict[str, Any]]:
        """Release all claimed tasks after worker threads have exited.

        Call this only from the supervisor after all engineer threads have joined.
        At that point any remaining `claimed` task has no live worker and would
        otherwise deadlock the queue until the TTL expires.
        """
        recovered: List[Dict[str, Any]] = []
        with self._mu:
            for task_id, rt in self._runtime.items():
                if rt.status != "claimed":
                    continue
                recovered.append({
                    "task_id": task_id,
                    "previous_claimed_by": rt.claimed_by,
                    "claimed_at": rt.claimed_at,
                    "claim_expires_at": rt.claim_expires_at,
                })
                self._release_locks_locked(task_id)
                rt.status = "todo"
                rt.notes = str(note or "orphaned_claim_recovered")
                rt.claimed_by = None
                rt.claimed_at = None
                rt.claim_expires_at = None
            if recovered:
                self._persist_state_locked()
        return recovered

    def get_results(self) -> Dict[str, Dict[str, Any]]:
        with self._mu:
            return dict(self._results)

    def reset(self) -> None:
        with self._mu:
            self._work_items = {}
            self._runtime = {}
            self._locks = {}
            self._results = {}
            self._persist_state_locked()

    # ------------------
    # Internal helpers
    # ------------------

    def _parse_work_item(self, wi: Dict[str, Any]) -> WorkItem:
        required = [
            "task_id",
            "summary",
            "capabilities_required",
            "dependencies",
            "scope_in",
            "scope_out",
            "interfaces",
            "acceptance_criteria",
            "verification",
            "files_expected",
            "risk_notes",
        ]
        for k in required:
            if k not in wi:
                raise ValueError(f"work_item missing required field: {k}")
        return WorkItem(
            task_id=str(wi["task_id"]),
            summary=str(wi["summary"]),
            capabilities_required=[str(x) for x in (wi.get("capabilities_required") or [])],
            dependencies=[str(x) for x in (wi.get("dependencies") or [])],
            scope_in=str(wi["scope_in"]),
            scope_out=str(wi["scope_out"]),
            interfaces=[str(x) for x in (wi.get("interfaces") or [])],
            acceptance_criteria=[str(x) for x in (wi.get("acceptance_criteria") or [])],
            verification=[str(x) for x in (wi.get("verification") or [])],
            files_expected=[str(x) for x in (wi.get("files_expected") or [])],
            risk_notes=str(wi["risk_notes"]),
        )

    def _deps_satisfied_locked(self, deps: List[str]) -> bool:
        for d in deps:
            rt = self._runtime.get(d)
            if rt is None or rt.status != "done":
                return False
        return True

    def _capability_score(
        self,
        item: WorkItem,
        engineer_caps: set,
        allow_partial: bool,
    ) -> int:
        """
        Returns:
          - -1 if not eligible
          - otherwise >=0 score (higher is better)
        """
        req = set([c.strip() for c in item.capabilities_required if c.strip()])
        if not req:
            # No capability requirement → any engineer can do it
            return 0

        if not engineer_caps:
            # Treat as generalist; eligible but lower score than specialists.
            return 1

        if "*" in engineer_caps or "generalist" in engineer_caps:
            return 2

        overlap = len(req.intersection(engineer_caps))
        if overlap == 0:
            return -1

        if allow_partial:
            return overlap
        else:
            return overlap if overlap == len(req) else -1

    def _locks_available_locked(self, item: WorkItem, now: float) -> bool:
        lock_keys = self._lock_keys_for_item(item)
        for lk in lock_keys:
            holder = self._locks.get(lk)
            if holder is not None:
                return False
        return True

    def _lock_keys_for_item(self, item: WorkItem) -> List[str]:
        """
        Simple heuristic:
          - lock each expected file path
          - also lock top-level directory to reduce parallel edits in same module

        You can tune this. The goal is "safe parallelism", not perfect locking.
        """
        keys: List[str] = []
        for p in item.files_expected:
            p = (p or "").strip()
            if not p:
                continue
            keys.append(f"file:{p}")

            # module lock
            parts = p.split("/")
            if len(parts) >= 2:
                keys.append(f"module:{parts[0]}/{parts[1]}")
            elif len(parts) == 1:
                keys.append(f"module:{parts[0]}")
        # stable unique
        out = sorted(set(keys))
        return out

    def _claim_locked(self, task_id: str, engineer_id: str, now: float) -> None:
        item = self._work_items[task_id]
        rt = self._runtime[task_id]
        rt.status = "claimed"
        rt.claimed_by = engineer_id
        rt.claimed_at = now
        rt.claim_expires_at = now + self.claim_ttl_seconds
        rt.notes = "claimed"

        if self.enable_file_locks:
            for lk in self._lock_keys_for_item(item):
                self._locks[lk] = task_id

    def _release_locks_locked(self, task_id: str) -> None:
        to_del = [lk for lk, holder in self._locks.items() if holder == task_id]
        for lk in to_del:
            del self._locks[lk]

    def _expire_stale_claims_locked(self, now: float) -> None:
        """
        If a worker dies mid-task, we allow the task to be reclaimed after TTL.
        """
        for task_id, rt in self._runtime.items():
            if rt.status != "claimed":
                continue
            exp = rt.claim_expires_at
            if exp is not None and now >= exp:
                # release locks + return to todo
                self._release_locks_locked(task_id)
                rt.status = "todo"
                rt.notes = "claim_expired"
                rt.claimed_by = None
                rt.claimed_at = None
                rt.claim_expires_at = None

    def _reconcile_blocked_dependencies_locked(self) -> None:
        """
        Optional safety: if a task depends on a missing task_id, block it explicitly.
        """
        known = set(self._work_items.keys())
        for task_id, item in self._work_items.items():
            missing = [d for d in item.dependencies if d not in known]
            if missing:
                rt = self._runtime[task_id]
                rt.status = "blocked"
                rt.notes = f"missing_dependencies: {missing}"

    def _summary_locked(self) -> Dict[str, Any]:
        counts: Dict[str, int] = {"todo": 0, "claimed": 0, "done": 0, "blocked": 0}
        for rt in self._runtime.values():
            counts[rt.status] = counts.get(rt.status, 0) + 1
        return {
            "counts": counts,
            "total": len(self._runtime),
        }

    # ------------------
    # Persistence (best-effort)
    # ------------------

    def _persist_state_locked(self) -> None:
        if not self.state_path:
            return
        try:
            payload = {
                "work_items": {tid: asdict(wi) for tid, wi in self._work_items.items()},
                "runtime": {tid: asdict(rt) for tid, rt in self._runtime.items()},
                "locks": dict(self._locks),
                "results": dict(self._results),
                "meta": {
                    "claim_ttl_seconds": self.claim_ttl_seconds,
                    "enable_file_locks": self.enable_file_locks,
                    "saved_at": time.time(),
                },
            }
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception:
            # best-effort: do not fail orchestration on persistence
            pass

    def _load_state_best_effort(self) -> None:
        if not self.state_path or not os.path.exists(self.state_path):
            return
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            work_items_raw = payload.get("work_items", {})
            runtime_raw = payload.get("runtime", {})
            locks_raw = payload.get("locks", {})
            results_raw = payload.get("results", {})

            # Work items
            items: Dict[str, WorkItem] = {}
            for tid, wi in work_items_raw.items():
                items[tid] = WorkItem(**wi)

            # Runtime
            runtime: Dict[str, WorkItemRuntime] = {}
            for tid, rt in runtime_raw.items():
                runtime[tid] = WorkItemRuntime(**rt)

            self._work_items = items
            self._runtime = runtime
            self._locks = {str(k): str(v) for k, v in (locks_raw or {}).items()}
            self._results = dict(results_raw or {})
        except Exception:
            # Ignore; start fresh
            self._work_items = {}
            self._runtime = {}
            self._locks = {}
            self._results = {}


# -----------------------------------------
# Orchestrator "agent" wrapper (optional)
# -----------------------------------------

class EngineeringOrchestratorAgent:
    """
    A thin wrapper so you can treat the orchestrator like an "agent" in your system.

    It is NOT an LLM agent. It is a deterministic coordinator for engineering tasks.
    """

    name = "Engineering Orchestrator"

    def __init__(
        self,
        *,
        claim_ttl_seconds: int = 30 * 60,
        enable_file_locks: bool = True,
        state_path: Optional[str] = None,
    ) -> None:
        self.queue = InMemoryTaskQueue(
            claim_ttl_seconds=claim_ttl_seconds,
            enable_file_locks=enable_file_locks,
            state_path=state_path,
        )

    def run(self, *, agent_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Commands:
          - {"command": "ingest_plan", "eng_lead_output": {...}}
          - {"command": "claim_next", "engineer_id": "...", "engineer_capabilities": [...]}
          - {"command": "submit_result", "result": {...}, "mark_done": true/false, "mark_blocked": true/false, "blocked_reason": "..."}
          - {"command": "get_state"}
          - {"command": "get_results"}
          - {"command": "reset"}

        Returns a JSON-serializable dict.
        """
        cmd = str(agent_input.get("command", "")).strip()

        if cmd == "ingest_plan":
            payload = agent_input.get("eng_lead_output", None)
            if not isinstance(payload, dict):
                raise ValueError("ingest_plan requires eng_lead_output: dict")
            self.queue.ingest_eng_lead_output(
                payload,
                preserve_existing_state=bool(agent_input.get("preserve_existing_state", False)),
            )
            return {"ok": True, "summary": self.queue.get_state().get("summary")}

        if cmd == "claim_next":
            engineer_id = str(agent_input.get("engineer_id", "")).strip()
            if not engineer_id:
                raise ValueError("claim_next requires engineer_id")
            caps = agent_input.get("engineer_capabilities", None)
            if caps is not None and not isinstance(caps, list):
                raise ValueError("engineer_capabilities must be a list or omitted")
            item = self.queue.claim_next(
                engineer_id=engineer_id,
                engineer_capabilities=caps,
                allow_partial_capability_match=bool(agent_input.get("allow_partial_capability_match", True)),
            )
            return {"claimed": item is not None, "work_item": asdict(item) if item else None}

        if cmd == "submit_result":
            raw = agent_input.get("result", None)
            if not isinstance(raw, dict):
                raise ValueError("submit_result requires result: dict")

            # Minimal validation; keep flexible. Older/abnormal blocked-submission
            # records may omit optional payload fields such as `changes`. Treat those
            # fields as empty instead of crashing the worker and leaving the task
            # permanently claimed.
            for k in ["task_id", "engineer_id"]:
                if k not in raw:
                    raise ValueError(f"result missing required field: {k}")

            result = WorkResult(
                task_id=str(raw["task_id"]),
                engineer_id=str(raw["engineer_id"]),
                changes=raw.get("changes"),
                verification_run=[str(x) for x in (raw.get("verification_run") or [])],
                notes=str(raw.get("notes", "")),
                handoff_interfaces=[str(x) for x in (raw.get("handoff_interfaces") or [])],
            )

            mark_blocked = bool(agent_input.get("mark_blocked", False))
            # Backward-compatible safety: old blocked-submission records sometimes
            # omitted mark_done=False. Do not treat a blocked result as both done
            # and blocked; that crash leaves the task permanently claimed.
            mark_done = bool(agent_input.get("mark_done", False if mark_blocked else True))
            self.queue.submit_result(
                result=result,
                mark_done=mark_done,
                mark_blocked=mark_blocked,
                blocked_reason=agent_input.get("blocked_reason"),
            )
            return {"ok": True, "summary": self.queue.get_state().get("summary")}

        if cmd == "get_state":
            return self.queue.get_state()

        if cmd == "get_results":
            return {"results": self.queue.get_results()}

        if cmd == "list_blocked":
            return {"blocked": self.queue.list_blocked()}

        if cmd == "unblock_task":
            task_id = str(agent_input.get("task_id", "")).strip()
            if not task_id:
                raise ValueError("unblock_task requires task_id")
            note = str(agent_input.get("note", "resources_attached"))
            ok = self.queue.unblock_task(task_id=task_id, note=note)
            return {"ok": True, "unblocked": bool(ok), "task_id": task_id}

        if cmd == "release_task":
            task_id = str(agent_input.get("task_id", "")).strip()
            if not task_id:
                raise ValueError("release_task requires task_id")
            note = str(agent_input.get("note", "manual_release"))
            only_if_claimed_by = agent_input.get("only_if_claimed_by")
            ok = self.queue.release_task(
                task_id=task_id,
                note=note,
                only_if_claimed_by=str(only_if_claimed_by) if only_if_claimed_by else None,
            )
            return {"ok": True, "released": bool(ok), "task_id": task_id}

        if cmd == "recover_orphaned_claims":
            note = str(agent_input.get("note", "orphaned_claim_recovered"))
            recovered = self.queue.recover_orphaned_claims(note=note)
            return {"ok": True, "recovered_count": len(recovered), "recovered": recovered, "summary": self.queue.get_state().get("summary")}

        if cmd == "reset":
            self.queue.reset()
            return {"ok": True}

        raise ValueError(f"Unknown command: {cmd}")


# -------------------------
# Example usage (manual)
# -------------------------
if __name__ == "__main__":
    # Example local smoke test (doesn't call OpenAI).
    orch = EngineeringOrchestratorAgent(state_path=None)

    # Minimal fake eng lead output
    plan = {
        "work_items": [
            {
                "task_id": "T1",
                "summary": "Create DB schema",
                "capabilities_required": ["postgres"],
                "dependencies": [],
                "scope_in": "Define tables + migrations",
                "scope_out": "No API endpoints",
                "interfaces": ["db/schema.sql"],
                "acceptance_criteria": ["Migration runs cleanly"],
                "verification": ["run migrations"],
                "files_expected": ["db/migrations/001_init.sql"],
                "risk_notes": "Low",
            },
            {
                "task_id": "T2",
                "summary": "Create API endpoint",
                "capabilities_required": ["fastapi"],
                "dependencies": ["T1"],
                "scope_in": "Implement /health",
                "scope_out": "No auth",
                "interfaces": ["GET /health"],
                "acceptance_criteria": ["200 OK returns JSON"],
                "verification": ["pytest -k health"],
                "files_expected": ["api/routes/health.py"],
                "risk_notes": "Low",
            },
        ]
    }

    orch.run(agent_input={"command": "ingest_plan", "eng_lead_output": plan})

    # Engineer A claims first task
    wi = orch.run(agent_input={"command": "claim_next", "engineer_id": "eng-A", "engineer_capabilities": ["postgres"]})
    print("Claim:", wi)

    # Submit result for T1
    orch.run(
        agent_input={
            "command": "submit_result",
            "result": {
                "task_id": "T1",
                "engineer_id": "eng-A",
                "changes": {"patch": "..."},  # placeholder
                "verification_run": ["run migrations: PASS"],
            },
        }
    )

    # Engineer B claims next task (T2 now unblocked)
    wi2 = orch.run(agent_input={"command": "claim_next", "engineer_id": "eng-B", "engineer_capabilities": ["fastapi"]})
    print("Claim:", wi2)

    print(json.dumps(orch.run(agent_input={"command": "get_state"}), indent=2))
