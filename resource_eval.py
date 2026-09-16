"""resource_eval.py

Generic resource-request module for any agentic workflow.

- Any stage/agent can submit a Resource Request (files, docs, keys, etc.).
- Requests are persisted under outputs/<run_id>/resources/.
- A single persistent UI ("Resources Desk") can satisfy all requests without opening new tabs.
- Users may also drop files into resources/inbox/ and click "Scan Inbox" in the UI.

This is NOT an LLM agent.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from task_id_utils import canonical_task_id as _shared_canonical_task_id


def _utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_json(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default
    return default


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _as_bool(value: Any, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"false", "0", "no", "n", "optional", "not required"}:
        return False
    if text in {"true", "1", "yes", "y", "required"}:
        return True
    return default


def _extract_task_ids_from_context(obj: Any) -> List[str]:
    from task_id_utils import extract_task_ids_from_context
    return extract_task_ids_from_context(obj)


def _canonical_task_id(fallback: str, req: Dict[str, Any]) -> str:
    return _shared_canonical_task_id(fallback, req)




def _looks_like_hard_stop(text: str) -> bool:
    msg = str(text or "").strip().lower()
    return msg.startswith("block whole") or msg.startswith("stop whole") or "hard stop" in msg or "halt work" in msg or "stop work" in msg


def _infer_resolution_overrides(decision: str, user_message: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Infer structured fallback/defer flags from resource decisions."""
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
    if _looks_like_hard_stop(msg) or decision_text in {"block", "block_task", "halt", "stop"}:
        out["user_blocked_task"] = True
    elif out.get("route_around_blocked_item"):
        out["user_blocked_task"] = False
    return out

class ResourceEval:
    """Filesystem-backed resource request manager."""

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)
        self.inbox_dir = self.base_dir / "inbox"
        self.pending_path = self.base_dir / "pending_requests.json"
        self.resolved_log = self.base_dir / "resolved_requests.jsonl"
        self.manifest_path = self.base_dir / "asset_manifest.json"
        self.requests_log = self.base_dir / "requests.jsonl"
        self.task_directives_dir = self.base_dir / "task_directives"
        self._mu = threading.Lock()

        self.inbox_dir.mkdir(parents=True, exist_ok=True)
        self.task_directives_dir.mkdir(parents=True, exist_ok=True)
        if not self.pending_path.exists():
            _write_json(self.pending_path, [])
        if not self.manifest_path.exists():
            _write_json(self.manifest_path, {"version": 1, "assets": []})

    def inbox_path(self) -> str:
        return str(self.inbox_dir)

    def submit(
        self,
        *,
        agent: str,
        stage: str,
        items: List[Dict[str, Any]],
        task_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        user_facing: Optional[Dict[str, Any]] = None,
    ) -> str:
        req_id = str(uuid.uuid4())
        req = {
            "request_id": req_id,
            "created_at_utc": _utc_now_iso(),
            "agent": str(agent or stage or "agent"),
            "stage": str(stage or "stage"),
            "task_id": str(task_id) if task_id else None,
            "items": self._normalize_items(items),
            "context": context or {},
            "user_facing": user_facing or {},
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
                "items": req.get("items"),
                "context": req.get("context"),
                "user_facing": req.get("user_facing"),
            }
            return json.dumps(data, ensure_ascii=False, sort_keys=True, default=str)
        except Exception:
            return f"{req.get('stage')}|{req.get('agent')}|{req.get('task_id')}|{req.get('items')}"

    def list_pending(self) -> List[Dict[str, Any]]:
        with self._mu:
            pending = _read_json(self.pending_path, [])
            return pending if isinstance(pending, list) else []

    def get_request(self, request_id: str) -> Optional[Dict[str, Any]]:
        for r in self.list_pending():
            if str(r.get("request_id")) == str(request_id):
                return r
        return None

    def resolve(
        self,
        *,
        request_id: str,
        provided_assets: Optional[List[Dict[str, Any]]] = None,
        note: str = "",
        decision: str = "provided",
        user_message: str = "",
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        provided_assets = provided_assets or []
        overrides = _infer_resolution_overrides(str(decision or "provided"), str(user_message or ""), overrides)

        with self._mu:
            pending = _read_json(self.pending_path, [])
            if not isinstance(pending, list):
                pending = []

            keep: List[Dict[str, Any]] = []
            target: Optional[Dict[str, Any]] = None
            for r in pending:
                if str(r.get("request_id")) == str(request_id):
                    target = r
                else:
                    keep.append(r)

            if target is None:
                return {"ok": True, "already": True}

            target["status"] = "resolved"
            target["resolved_at_utc"] = _utc_now_iso()

            manifest = _read_json(self.manifest_path, {"version": 1, "assets": []})
            if not isinstance(manifest, dict):
                manifest = {"version": 1, "assets": []}
            assets = manifest.get("assets")
            if not isinstance(assets, list):
                assets = []

            for a in provided_assets:
                if not isinstance(a, dict):
                    continue
                a2 = dict(a)
                a2.setdefault("request_id", request_id)
                a2.setdefault("agent", target.get("agent"))
                a2.setdefault("stage", target.get("stage"))
                a2.setdefault("task_id", target.get("task_id"))
                assets.append(a2)

            manifest["assets"] = assets
            _write_json(self.manifest_path, manifest)

            record = {"event": "resolved", "request": target, "note": str(note or ""), "decision": str(decision or "provided"), "user_message": str(user_message or ""), "overrides": overrides, "provided_assets": provided_assets}
            _write_json(self.pending_path, keep)
            _append_jsonl(self.resolved_log, record)

            # If this resource decision belongs to an engineering task, write the
            # directive immediately. This avoids a race where the UI clears
            # run_state before the background resource monitor has copied the
            # decision into task_directives/.
            task_id = target.get("task_id")
            if task_id:
                # Use structured request context as the source of truth. The UI can
                # surface stale run_state prompts, so a resource fallback for T3 must
                # not be written into task_directives/T4.json or vice versa.
                canonical_task_id = _canonical_task_id(str(task_id), target)
                task_id = canonical_task_id or str(task_id)
                directive_path = self.task_directives_dir / f"{task_id}.json"
                existing: Dict[str, Any] = {}
                try:
                    loaded = _read_json(directive_path, {})
                    if isinstance(loaded, dict):
                        existing = loaded
                except Exception:
                    existing = {}

                directive = dict(existing)
                directive.update({
                    "request_id": request_id,
                    "decision": record["decision"],
                    "user_message": record["user_message"],
                    "overrides": {**(existing.get("overrides") if isinstance(existing.get("overrides"), dict) else {}), **overrides},
                    "source": "resource_eval",
                    # Preserve request metadata immediately. The background
                    # monitor also writes/merges this context, but if the UI or
                    # a CLI resolver satisfies a request while the backend is not
                    # running, the next Engineer still needs to know which agent,
                    # stage, task, and item names the resource decision answered.
                    "request_stage": target.get("stage"),
                    "request_agent": target.get("agent"),
                    "request_items": target.get("items") if isinstance(target.get("items"), list) else directive.get("request_items", []),
                    "request_context": target.get("context") if isinstance(target.get("context"), dict) else directive.get("request_context", {}),
                    "request_user_facing": target.get("user_facing") if isinstance(target.get("user_facing"), dict) else directive.get("request_user_facing", {}),
                    "resolved_at_utc": target.get("resolved_at_utc"),
                })
                if provided_assets:
                    directive["provided_assets"] = provided_assets
                else:
                    directive.setdefault("provided_assets", [])
                _write_json(directive_path, directive)

            return {"ok": True, "resolved": True, "request": target}

    def wait_until_resolved(self, request_id: str, poll_seconds: float = 1.0) -> None:
        while True:
            if self.get_request(request_id) is None:
                return
            time.sleep(max(0.25, float(poll_seconds)))

    def read_manifest(self) -> Dict[str, Any]:
        with self._mu:
            m = _read_json(self.manifest_path, {"version": 1, "assets": []})
            return m if isinstance(m, dict) else {"version": 1, "assets": []}

    def scan_inbox_for_request(self, request: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
        items = request.get("items") or []
        if not isinstance(items, list):
            items = []

        wanted: List[Tuple[str, bool]] = []
        required = set()
        for it in items:
            if not isinstance(it, dict):
                continue
            nm = str(it.get("name") or "").strip()
            if not nm:
                continue
            req = _as_bool(it.get("required"), True)
            wanted.append((nm, req))
            if req:
                required.add(nm.lower())

        files = [p for p in self.inbox_dir.glob("*") if p.is_file()]
        provided: List[Dict[str, Any]] = []
        matched_required = set()

        for f in files:
            fn = f.name.lower()
            for nm, req in wanted:
                key = nm.lower()
                if key and key in fn:
                    provided.append({"name": nm, "type": "file", "path": f"inbox/{f.name}", "original_filename": f.name})
                    if req:
                        matched_required.add(key)
                    break

        missing = sorted([x for x in required if x not in matched_required])
        return provided, missing

    
    def read_resolution(self, request_id: str) -> Optional[Dict[str, Any]]:
        """
        Return the most recent resolved record for a given request_id, if any.
        Scans resolved_requests.jsonl from bottom.
        """
        path = Path(self.resolved_log)
        if not path.exists():
            return None
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except Exception:
            return None
        for ln in reversed(lines):
            try:
                obj = json.loads(ln)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            req = obj.get("request")
            if isinstance(req, dict) and str(req.get("request_id")) == str(request_id):
                return obj
        return None
    def _normalize_items(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not isinstance(items, list):
            return out
        for it in items:
            if isinstance(it, str):
                s = it.strip()
                if not s:
                    continue
                out.append({"name": s, "kind": "resource", "required": True, "preferred_formats": [], "notes": ""})
                continue
            if not isinstance(it, dict):
                continue
            nm = str(it.get("name") or it.get("id") or "").strip()
            if not nm:
                continue
            formats = it.get("preferred_formats") or it.get("formats") or []
            if isinstance(formats, str):
                formats = [formats]
            elif not isinstance(formats, list):
                formats = [formats]
            out.append({
                "name": nm,
                "kind": str(it.get("kind") or "resource"),
                "required": _as_bool(it.get("required"), True),
                "preferred_formats": [str(x) for x in formats if str(x).strip()],
                "notes": str(it.get("notes") or ""),
            })
        return out
