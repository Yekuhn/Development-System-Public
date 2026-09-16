"""Utilities for task-id extraction from nested workflow/request payloads.

These helpers keep HumanRequestDesk, ResourceEval, and operation.py aligned so
resolved user/resource decisions cannot be written to the wrong task directive.
"""

from __future__ import annotations

from typing import Any, Dict, List


def _clean_task_id(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"none", "null"}:
        return ""
    return text


def _iter_dicts(obj: Any) -> List[Dict[str, Any]]:
    """Return nested dictionaries in deterministic root-to-leaf order."""
    out: List[Dict[str, Any]] = []
    seen: set[int] = set()

    def walk(x: Any) -> None:
        if not isinstance(x, dict):
            return
        ident = id(x)
        if ident in seen:
            return
        seen.add(ident)
        out.append(x)
        for key in ("request_context", "context", "block_payload", "workflow_decision", "context_pack", "work_item"):
            val = x.get(key)
            if isinstance(val, dict):
                walk(val)
        # Preserve coverage for uncommon nested dicts without exploding into
        # lists of issue details or code outputs.
        for key, val in x.items():
            if key in {"request_context", "context", "block_payload", "workflow_decision", "context_pack", "work_item"}:
                continue
            if isinstance(val, dict) and any(k in val for k in ("task_id", "target_task_id", "affected_task_id", "work_item", "workflow_decision")):
                walk(val)

    walk(obj)
    return out


def extract_task_ids_from_context(obj: Any) -> List[str]:
    """Extract unique task ids from nested workflow/request payloads."""
    found: List[str] = []

    def add(value: Any) -> None:
        text = _clean_task_id(value)
        if text and text not in found:
            found.append(text)

    for d in _iter_dicts(obj):
        add(d.get("task_id"))
        add(d.get("target_task_id"))
        add(d.get("affected_task_id"))
        wi = d.get("work_item")
        if isinstance(wi, dict):
            add(wi.get("task_id"))
        wd = d.get("workflow_decision")
        if isinstance(wd, dict):
            add(wd.get("affected_task_id"))
            cp = wd.get("context_pack")
            if isinstance(cp, dict):
                add(cp.get("task_id"))
                wi2 = cp.get("work_item")
                if isinstance(wi2, dict):
                    add(wi2.get("task_id"))
    return found


def canonical_task_id(fallback: str, obj: Any) -> str:
    """Choose the safest task id for writing a directive.

    Priority favors the actual workflow/work_item context over a possibly stale
    UI-selected or run_state top-level task_id. This directly prevents T3/T4
    directive cross-contamination during resumed runs.
    """
    fallback_clean = _clean_task_id(fallback)
    dicts = _iter_dicts(obj)

    # Highest confidence: explicit workflow decision affected task, deepest last.
    for d in reversed(dicts):
        wd = d.get("workflow_decision")
        if isinstance(wd, dict):
            tid = _clean_task_id(wd.get("affected_task_id"))
            if tid:
                return tid

    # Next: the concrete work item being discussed, deepest last.
    for d in reversed(dicts):
        wi = d.get("work_item")
        if isinstance(wi, dict):
            tid = _clean_task_id(wi.get("task_id"))
            if tid:
                return tid

    # Then direct affected/target task fields in nested payloads.
    for d in reversed(dicts):
        for key in ("affected_task_id", "target_task_id"):
            tid = _clean_task_id(d.get(key))
            if tid:
                return tid

    ids = extract_task_ids_from_context(obj)
    return ids[-1] if ids else fallback_clean
