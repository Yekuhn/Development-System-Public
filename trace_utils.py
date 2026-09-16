"""
trace_utils.py

Lightweight trace observability for the Ascendant Path / VeRealm agent model.

Trace is intentionally not a UI. It is a structured JSONL record of internal
workflow events so failed or expensive runs can be debugged later.

Default path is set by operation.py:
    ASCENDANT_TRACE_PATH=outputs/<run_id>/trace.jsonl

If ASCENDANT_TRACE_PATH is not set, trace_event() is a no-op.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_json_value(value: Any) -> Any:
    """Best-effort JSON-safe conversion without dumping giant raw prompts/files."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _safe_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe_json_value(v) for v in value]
    try:
        json.dumps(value)
        return value
    except Exception:
        return repr(value)


def trace_event(event: str, **payload: Any) -> None:
    """Append a structured trace event if tracing is enabled."""
    trace_path = os.getenv("ASCENDANT_TRACE_PATH", "").strip()
    if not trace_path:
        return

    record: Dict[str, Any] = {
        "ts_utc": _utc_now_iso(),
        "run_id": os.getenv("ASCENDANT_RUN_ID", ""),
        "event": event,
    }
    record.update({k: _safe_json_value(v) for k, v in payload.items()})

    try:
        path = Path(trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        # Trace should never break the workflow.
        return
