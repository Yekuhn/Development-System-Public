"""
run_state.py

Lightweight RunState for the Ascendant Path / VeRealm agent model.

RunState is the workflow bookmark and pointer layer. It does not replace:
- trace.jsonl, which records history;
- the engineering orchestrator, which owns task queue state;
- resource_eval.py, which owns resource request/resolution state;
- artifact files, which remain the source of actual outputs.

Default path:
    outputs/<run_id>/run_state.json
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


NODE_INTAKE = "intake"
NODE_PM = "pm"
NODE_UX = "ux"
NODE_ENG_LEAD = "eng_lead"
NODE_ORCHESTRATOR_INGEST = "orchestrator_ingest"
NODE_ENGINEERING = "engineering"
NODE_EXECUTOR = "project_executor"
NODE_RUNBOOK = "runbook"
NODE_COORDINATOR_FINAL_HANDOFF = "coordinator_final_handoff"
NODE_DONE = "done"

NODE_ORDER = [
    NODE_INTAKE,
    NODE_PM,
    NODE_UX,
    NODE_ENG_LEAD,
    NODE_ORCHESTRATOR_INGEST,
    NODE_ENGINEERING,
    NODE_EXECUTOR,
    NODE_RUNBOOK,
    NODE_COORDINATOR_FINAL_HANDOFF,
    NODE_DONE,
]

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_BLOCKED = "blocked"
STATUS_SKIPPED = "skipped"
STATUS_FAILED = "failed"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    try:
        json.dumps(value)
        return value
    except Exception:
        return repr(value)


class RunState:
    """Persistent run_state.json manager.

    This object is intentionally small. It stores current workflow position and
    pointers to detailed records, not full artifacts/logs.
    """

    def __init__(self, run_dir: Path, *, run_id: str):
        self.run_dir = Path(run_dir)
        self.run_id = str(run_id)
        self.path = self.run_dir / "run_state.json"
        self._lock = threading.Lock()
        self.state: Dict[str, Any] = self._load_or_default()

    def _default_state(self) -> Dict[str, Any]:
        return {
            "schema_version": "run_state",
            "run_id": self.run_id,
            "created_at_utc": utc_now_iso(),
            "updated_at_utc": utc_now_iso(),
            "current_node": None,
            "resume_from": None,
            "node_status": {node: STATUS_PENDING for node in NODE_ORDER},
            "artifact_paths": {},
            "pointers": {},
            "task_attempts": {},
            "active_task_id": None,
            "active_engineer_id": None,
            "blocked": False,
            "block_type": None,
            "block_reason": None,
            "block_payload": None,
            "last_event": None,
            "last_decision": None,
            "done": False,
        }

    def _load_or_default(self) -> Dict[str, Any]:
        if self.path.exists():
            try:
                obj = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(obj, dict):
                    # Merge defaults so older state files remain usable.
                    base = self._default_state()
                    base.update(obj)
                    base.setdefault("node_status", {})
                    for node in NODE_ORDER:
                        base["node_status"].setdefault(node, STATUS_PENDING)
                    base.setdefault("artifact_paths", {})
                    base.setdefault("pointers", {})
                    base.setdefault("task_attempts", {})
                    return base
            except Exception:
                pass
        return self._default_state()

    def save(self) -> Path:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.state["updated_at_utc"] = utc_now_iso()
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.path)
        return self.path

    def initialize(self, *, run_mode: str = "development", pointers: Optional[Dict[str, Any]] = None) -> None:
        with self._lock:
            self.state["run_id"] = self.run_id
            self.state["run_mode"] = run_mode
            if pointers:
                self.state.setdefault("pointers", {}).update(_json_safe(pointers))
            self.save()

    def set_pointer(self, key: str, value: Any) -> None:
        with self._lock:
            self.state.setdefault("pointers", {})[str(key)] = _json_safe(value)
            self.save()

    def set_artifact(self, name: str, path: Any) -> None:
        with self._lock:
            self.state.setdefault("artifact_paths", {})[str(name)] = _json_safe(path)
            self.save()

    def mark_node_started(self, node: str, *, event: Optional[str] = None, payload: Optional[Dict[str, Any]] = None) -> None:
        with self._lock:
            self.state["current_node"] = node
            self.state["resume_from"] = node
            self.state.setdefault("node_status", {})[node] = STATUS_RUNNING
            self.state["blocked"] = False
            self.state["block_type"] = None
            self.state["block_reason"] = None
            self.state["block_payload"] = None
            self.state["last_event"] = event or f"{node}.started"
            if payload:
                self.state["last_payload"] = _json_safe(payload)
            self.save()

    def mark_node_completed(self, node: str, *, event: Optional[str] = None, artifact_name: Optional[str] = None, artifact_path: Any = None, payload: Optional[Dict[str, Any]] = None) -> None:
        with self._lock:
            self.state.setdefault("node_status", {})[node] = STATUS_COMPLETED
            self.state["current_node"] = node
            self.state["resume_from"] = self._next_node_after(node)
            self.state["last_event"] = event or f"{node}.completed"
            self.state["blocked"] = False
            self.state["block_type"] = None
            self.state["block_reason"] = None
            self.state["block_payload"] = None
            if artifact_name and artifact_path is not None:
                self.state.setdefault("artifact_paths", {})[artifact_name] = _json_safe(artifact_path)
            if payload:
                self.state["last_payload"] = _json_safe(payload)
            self.save()

    def mark_node_skipped(self, node: str, *, reason: str = "cached_artifact_present") -> None:
        with self._lock:
            # A skipped node is operationally completed for resume purposes.
            self.state.setdefault("node_status", {})[node] = STATUS_COMPLETED
            self.state["current_node"] = node
            self.state["resume_from"] = self._next_node_after(node)
            self.state["last_event"] = f"{node}.skipped"
            self.state["last_payload"] = {"reason": reason}
            self.state["blocked"] = False
            self.state["block_type"] = None
            self.state["block_reason"] = None
            self.state["block_payload"] = None
            self.save()

    def mark_blocked(self, *, node: Optional[str] = None, block_type: str, reason: str, payload: Optional[Dict[str, Any]] = None) -> None:
        with self._lock:
            if node:
                self.state["current_node"] = node
                self.state["resume_from"] = node
                self.state.setdefault("node_status", {})[node] = STATUS_BLOCKED
            self.state["blocked"] = True
            self.state["block_type"] = block_type
            self.state["block_reason"] = reason
            self.state["block_payload"] = _json_safe(payload or {})
            self.state["last_event"] = "run.blocked"
            self.save()

    def clear_block(self, *, event: str = "run.unblocked") -> None:
        with self._lock:
            current = self.state.get("current_node")
            node_status = self.state.setdefault("node_status", {})
            if current and node_status.get(current) == STATUS_BLOCKED:
                # Clearing a block means the current node is eligible to continue/resume.
                # Keeping node_status as "blocked" created stale UI states where the
                # global block flag was false but the dashboard still looked blocked.
                node_status[current] = STATUS_RUNNING
            self.state["blocked"] = False
            self.state["block_type"] = None
            self.state["block_reason"] = None
            self.state["block_payload"] = None
            self.state["last_event"] = event
            self.save()

    def mark_task_attempt(self, *, task_id: str, engineer_id: Optional[str] = None, attempt: int) -> None:
        with self._lock:
            key = str(task_id)
            self.state.setdefault("task_attempts", {})[key] = int(attempt)
            self.state["active_task_id"] = key
            self.state["active_engineer_id"] = engineer_id
            self.state["current_node"] = NODE_ENGINEERING
            self.state["resume_from"] = NODE_ENGINEERING
            self.state.setdefault("node_status", {})[NODE_ENGINEERING] = STATUS_RUNNING
            self.state["last_event"] = "engineering.task_attempt"
            self.save()

    def set_last_decision(self, decision: Dict[str, Any]) -> None:
        with self._lock:
            self.state["last_decision"] = _json_safe(decision)
            self.save()

    def mark_done(self) -> None:
        with self._lock:
            self.state["done"] = True
            self.state["current_node"] = NODE_DONE
            self.state["resume_from"] = None
            self.state.setdefault("node_status", {})[NODE_DONE] = STATUS_COMPLETED
            self.state["blocked"] = False
            self.state["last_event"] = "run.done"
            self.save()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return deepcopy_json(self.state)

    def _next_node_after(self, node: str) -> Optional[str]:
        try:
            idx = NODE_ORDER.index(node)
        except ValueError:
            return None
        if idx + 1 < len(NODE_ORDER):
            return NODE_ORDER[idx + 1]
        return None


def deepcopy_json(obj: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(_json_safe(obj), ensure_ascii=False))
