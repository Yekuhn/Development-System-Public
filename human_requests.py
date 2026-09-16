"""
human_requests.py

Filesystem-backed queue for general human decisions that are not asset/resource
uploads. This is intentionally small and mirrors ResourceEval enough for the
workflow runner to block, resume, and pass user directives back into tasks.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from task_id_utils import canonical_task_id as _shared_canonical_task_id


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default
    return default


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _extract_task_ids_from_context(obj: Any) -> List[str]:
    from task_id_utils import extract_task_ids_from_context
    return extract_task_ids_from_context(obj)


def _canonical_task_id(fallback: str, req: Dict[str, Any]) -> str:
    return _shared_canonical_task_id(fallback, req)




def _looks_like_hard_stop(text: str) -> bool:
    msg = str(text or "").strip().lower()
    return msg.startswith("block whole") or msg.startswith("stop whole") or "hard stop" in msg or "halt work" in msg or "stop work" in msg


def _infer_resolution_overrides(decision: str, user_message: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Infer structured workflow flags from a resolved human answer.

    The runner consumes top-level override keys. Keeping this inference here means
    UI, API, and future command-line resolutions behave the same way.
    """
    out: Dict[str, Any] = dict(overrides or {})
    decision_text = str(decision or "").strip().lower()
    msg = str(user_message or "").strip().lower()
    if decision_text in {"accept_limitation", "defer", "route_around", "mark_unavailable"}:
        out.update({
            "route_around_blocked_item": True,
            "route_around_nonessential_blockers": True,
            "defer_to_v2_if_needed": True,
        })
    if any(term in msg for term in (
        "route around", "blocked requested item", "blocked the requested item",
        "do not ask", "don't ask", "defer", "v2", "mock", "mocks", "fixture",
        "fixtures", "placeholder", "fallback", "accept code-only", "accept code only",
        "code-only", "code only", "continue based on code review",
    )):
        out.update({
            "route_around_blocked_item": True,
            "route_around_nonessential_blockers": True,
            "defer_to_v2_if_needed": True,
        })
    if any(term in msg for term in (
        "runtime", "live backend", "browser", "screenshot", "screenshots", "clipboard",
        "curl", "testclient", "playwright", "manual evidence", "runtime evidence",
        "verification evidence", "success banner", "spinner proof", "logs panel",
    )) and any(term in msg for term in (
        "do not ask", "don't ask", "defer", "route around", "unavailable", "not required",
        "accept code", "use mock", "use fixture", "v2",
    )):
        out.update({
            "defer_runtime_verification": True,
            "accept_code_only_review": True,
            "qa_waiver": True,
        })
    if any(term in msg for term in ("unavailable", "not available", "cannot provide", "can't provide")):
        out.update({"unavailable": True, "use_placeholder": True, "generate_fixtures": True})
    if any(term in msg for term in ("+3", "three additional", "3 additional", "grant 3", "grant +3")):
        out["extra_attempts_granted"] = max(int(out.get("extra_attempts_granted") or 0), 3)
    elif any(term in msg for term in ("+2", "two additional", "2 additional", "grant 2", "grant +2")):
        out["extra_attempts_granted"] = max(int(out.get("extra_attempts_granted") or 0), 2)
    elif any(term in msg for term in ("+1", "one additional", "1 additional", "grant 1", "grant +1")):
        out["extra_attempts_granted"] = max(int(out.get("extra_attempts_granted") or 0), 1)
    if _looks_like_hard_stop(msg) or decision_text in {"block", "block_task", "halt", "stop"}:
        out["user_blocked_task"] = True
    elif out.get("route_around_blocked_item"):
        out["user_blocked_task"] = False
    return out

class HumanRequestDesk:
    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)
        self.pending_path = self.base_dir / "pending_human_requests.json"
        self.resolved_log = self.base_dir / "resolved_human_requests.jsonl"
        self.requests_log = self.base_dir / "human_requests.jsonl"
        self.task_directives_dir = self.base_dir / "human_task_directives"
        self._mu = threading.Lock()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.task_directives_dir.mkdir(parents=True, exist_ok=True)
        if not self.pending_path.exists():
            _write_json(self.pending_path, [])

    def submit(
        self,
        *,
        stage: str,
        agent: str,
        reason: str,
        question: str,
        task_id: Optional[str] = None,
        options: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        req_id = str(uuid.uuid4())
        req = {
            "request_id": req_id,
            "created_at_utc": _utc_now_iso(),
            "stage": str(stage or "workflow"),
            "agent": str(agent or "Team Lead"),
            "task_id": str(task_id) if task_id else None,
            "reason": str(reason or "human_input_required"),
            "question": str(question or "Human input required."),
            "options": list(options or []),
            "context": context or {},
            "status": "pending",
            "resolved_at_utc": None,
        }
        with self._mu:
            pending = _read_json(self.pending_path, [])
            if not isinstance(pending, list):
                pending = []
            fingerprint = self._fingerprint(req)
            for existing in pending:
                if not isinstance(existing, dict):
                    continue
                if self._fingerprint(existing) == fingerprint:
                    return str(existing.get("request_id") or req_id)
            pending.append(req)
            _write_json(self.pending_path, pending)
            _append_jsonl(self.requests_log, {"event": "submitted", **req})
        return req_id


    @staticmethod
    def _fingerprint(req: Dict[str, Any]) -> str:
        try:
            data = {
                "stage": req.get("stage"),
                "agent": req.get("agent"),
                "task_id": req.get("task_id"),
                "reason": req.get("reason"),
                "question": req.get("question"),
                "options": req.get("options"),
                "context": req.get("context"),
            }
            return json.dumps(data, ensure_ascii=False, sort_keys=True, default=str)
        except Exception:
            return f"{req.get('stage')}|{req.get('agent')}|{req.get('task_id')}|{req.get('reason')}|{req.get('question')}"

    def list_pending(self) -> List[Dict[str, Any]]:
        with self._mu:
            pending = _read_json(self.pending_path, [])
            return pending if isinstance(pending, list) else []

    def get_request(self, request_id: str) -> Optional[Dict[str, Any]]:
        for req in self.list_pending():
            if str(req.get("request_id")) == str(request_id):
                return req
        return None

    def resolve(self, *, request_id: str, decision: str, user_message: str = "", overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        overrides = _infer_resolution_overrides(str(decision or "continue"), str(user_message or ""), overrides)
        with self._mu:
            pending = _read_json(self.pending_path, [])
            if not isinstance(pending, list):
                pending = []
            keep: List[Dict[str, Any]] = []
            target: Optional[Dict[str, Any]] = None
            for req in pending:
                if str(req.get("request_id")) == str(request_id):
                    target = req
                else:
                    keep.append(req)
            if target is None:
                return {"ok": True, "already": True}
            target["status"] = "resolved"
            target["resolved_at_utc"] = _utc_now_iso()
            record = {
                "event": "resolved",
                "request": target,
                "decision": str(decision or "continue"),
                "user_message": str(user_message or ""),
                "overrides": overrides,
            }
            _write_json(self.pending_path, keep)
            _append_jsonl(self.resolved_log, record)
            task_id = target.get("task_id")
            if task_id:
                # Use structured request context as the source of truth for the
                # directive target. This prevents a stale UI/run_state request from
                # writing a T4 answer into human_task_directives/T3.json.
                canonical_task_id = _canonical_task_id(str(task_id), target)
                task_id = canonical_task_id or str(task_id)
                # Merge with any existing directive instead of overwriting it.
                # A previous max-attempt approval, reassign instruction, or Team
                # Lead note may already be stored for the same task. Direct
                # HumanRequestDesk resolution should preserve that context just
                # like the UI/monitor merge path does.
                directive_path = self.task_directives_dir / f"{task_id}.json"
                existing: Dict[str, Any] = {}
                try:
                    loaded = _read_json(directive_path, {})
                    if isinstance(loaded, dict):
                        existing = loaded
                except Exception:
                    existing = {}
                old_overrides = existing.get("overrides") if isinstance(existing.get("overrides"), dict) else {}
                old_context = existing.get("request_context") if isinstance(existing.get("request_context"), dict) else {}
                new_context = target.get("context") if isinstance(target.get("context"), dict) else {}
                directive = dict(existing)
                directive.update({
                    "request_id": request_id,
                    "decision": record["decision"],
                    "user_message": record["user_message"],
                    "overrides": {**old_overrides, **overrides},
                    "source": "human_requests",
                    # Preserve the original request metadata/context so the Team
                    # Lead can turn a user decision into a precise, single-use
                    # workflow directive instead of a vague persistent note.
                    "request_stage": target.get("stage") or directive.get("request_stage"),
                    "request_agent": target.get("agent") or directive.get("request_agent"),
                    "request_reason": target.get("reason") or directive.get("request_reason"),
                    "request_options": target.get("options") or directive.get("request_options", []),
                    "request_context": {**old_context, **new_context},
                    "resolved_at_utc": target.get("resolved_at_utc"),
                })
                _write_json(directive_path, directive)
            return {"ok": True, "resolved": True, "request": target}

    def wait_until_resolved(self, request_id: str, poll_seconds: float = 1.0) -> None:
        while True:
            if self.get_request(request_id) is None:
                return
            time.sleep(max(0.25, float(poll_seconds)))

    def read_resolution(self, request_id: str) -> Optional[Dict[str, Any]]:
        if not self.resolved_log.exists():
            return None
        try:
            lines = self.resolved_log.read_text(encoding="utf-8").splitlines()
        except Exception:
            return None
        for line in reversed(lines):
            try:
                obj = json.loads(line)
            except Exception:
                continue
            req = obj.get("request") if isinstance(obj, dict) else None
            if isinstance(req, dict) and str(req.get("request_id")) == str(request_id):
                return obj
        return None
