# record.py
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:  # append-only
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


@dataclass
class Recorder:
    run_id: str
    log_path: Path
    out_dir: Path

    @classmethod
    def new(cls, *, logs_dir: str = "logs", outputs_dir: str = "outputs") -> "Recorder":
        run_id = str(uuid.uuid4())
        log_path = Path(logs_dir) / f"run_{run_id}.jsonl"
        out_dir = Path(outputs_dir) / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        return cls(run_id=run_id, log_path=log_path, out_dir=out_dir)

    def log(self, event: str, **payload: Any) -> None:
        append_jsonl(
            self.log_path,
            {
                "ts_utc": now_iso_utc(),
                "run_id": self.run_id,
                "event": event,
                **payload,
            },
        )

    def save_json(self, name: str, obj: Dict[str, Any]) -> Path:
        path = self.out_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)  # <-- added: auto-create subfolders
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        self.log("file_saved", file=str(path))
        return path
