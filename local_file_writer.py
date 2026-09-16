"""
local_file_writer.py

Deterministic local file writer for EngineerAgent code_output.

Purpose
-------
The Engineer Agent may return structured code output. This module is the only
component that turns that code output into actual files in the local workflow
workspace.

Design rules
------------
- AI generates code; Python writes files.
- Never write outside the configured workspace.
- Never allow absolute paths or path traversal.
- Do not run shell commands.
- Return a structured write report for QA and audit.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Sequence

from trace_utils import trace_event


ALLOWED_WRITE_MODES = {"create", "overwrite", "create_or_overwrite"}


def _trace_write_report(report: Dict[str, Any]) -> None:
    trace_event(
        "local_file_write_result",
        task_id=report.get("task_id"),
        engineer_id=report.get("engineer_id"),
        workspace_dir=report.get("workspace_dir"),
        should_write_to_file=report.get("should_write_to_file"),
        status=report.get("status"),
        files_written_count=len(report.get("files_written", [])),
        directories_created_count=len(report.get("directories_created", [])),
        files_skipped_count=len(report.get("files_skipped", [])),
        errors_count=len(report.get("errors", [])),
        files_written=[x.get("path") for x in report.get("files_written", []) if isinstance(x, dict)],
        directories_created=[x.get("path") for x in report.get("directories_created", []) if isinstance(x, dict)],
        files_skipped=[{"path": x.get("path"), "reason": x.get("reason")} for x in report.get("files_skipped", []) if isinstance(x, dict)],
    )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _safe_rel_path(raw_path: str) -> Tuple[Optional[Path], Optional[str]]:
    """Validate and normalize an agent-provided relative file path."""
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None, "empty_path"

    raw_path = raw_path.strip().replace("\\", "/")
    p = Path(raw_path)

    if p.is_absolute():
        return None, "absolute_paths_are_not_allowed"

    parts = p.parts
    if any(part in {"", ".", ".."} for part in parts):
        return None, "path_traversal_or_empty_component_not_allowed"

    # Block common sensitive/runtime paths. This is intentionally conservative.
    blocked_prefixes = {
        ".git",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        "node_modules",
        "outputs",
        "logs",
    }
    if parts and parts[0] in blocked_prefixes:
        return None, f"blocked_top_level_path:{parts[0]}"

    return p, None



def _normalize_allowed_paths(allowed_paths: Optional[Sequence[str]]) -> List[str]:
    """Normalize an optional task file-scope allowlist.

    Preserve a trailing slash for directory scopes. Without this,
    `sudoku-solver/frontend/` becomes `sudoku-solver/frontend` and files such
    as `sudoku-solver/frontend/.gitkeep` are incorrectly rejected.
    """
    out: List[str] = []
    if not allowed_paths:
        return out
    for raw in allowed_paths:
        raw_s = str(raw or "").strip().replace("\\", "/")
        is_dir_scope = raw_s.endswith("/")
        rel_path, err = _safe_rel_path(raw_s.rstrip("/"))
        if err or rel_path is None:
            continue
        normalized = str(rel_path).replace("\\", "/")
        if is_dir_scope and not normalized.endswith("/"):
            normalized += "/"
        out.append(normalized)
    return sorted(set(out))


def _allowed_directory_scopes(allowed: Sequence[str]) -> List[str]:
    dirs: List[str] = []
    for raw in allowed:
        item = str(raw or "").strip().replace("\\", "/")
        if item.endswith("/"):
            dirs.append(item.rstrip("/"))
    return sorted(set(dirs))


def _path_allowed_by_scope(rel_path: Path, allowed: Sequence[str]) -> bool:
    """Return True if rel_path is inside the Engineering Lead files_expected scope."""
    if not allowed:
        return False
    candidate = str(rel_path).replace("\\", "/")
    for raw in allowed:
        item = str(raw or "").strip().replace("\\", "/")
        if not item:
            continue
        if item.endswith("/"):
            if candidate.startswith(item):
                return True
        elif candidate == item:
            return True
    return False

def _resolve_inside_workspace(workspace_dir: Path, rel_path: Path) -> Tuple[Optional[Path], Optional[str]]:
    workspace_resolved = workspace_dir.resolve()
    target = (workspace_dir / rel_path).resolve()
    try:
        target.relative_to(workspace_resolved)
    except ValueError:
        return None, "resolved_path_escapes_workspace"
    return target, None


def write_code_output(
    *,
    code_output: Dict[str, Any],
    workspace_dir: str | Path,
    task_id: str = "",
    engineer_id: str = "",
    allowed_paths: Optional[Sequence[str]] = None,
    enforce_allowed_paths: bool = False,
) -> Dict[str, Any]:
    """
    Write EngineerAgent code_output files into workspace_dir.

    Expected code_output shape:
        {
          "should_write_to_file": true,
          "files": [
            {"path": "src/main.py", "content": "...", "write_mode": "create_or_overwrite"}
          ],
          "notes": "..."
        }
    """
    workspace = Path(workspace_dir)
    workspace.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "task_id": task_id or "",
        "engineer_id": engineer_id or "",
        "workspace_dir": str(workspace),
        "created_at_utc": _utc_now_iso(),
        "should_write_to_file": bool(code_output.get("should_write_to_file")) if isinstance(code_output, dict) else False,
        "status": "skipped",
        "files_written": [],
        "directories_created": [],
        "placeholder_files_created": [],
        "files_skipped": [],
        "errors": [],
        "notes": "",
        "scope": {
            "enforced": bool(enforce_allowed_paths),
            "allowed_paths": _normalize_allowed_paths(allowed_paths),
            "violations": [],
        },
    }

    def _attach_verification_evidence() -> None:
        report["verification_evidence"] = {
            "directories_created": [x.get("path") for x in report.get("directories_created", []) if isinstance(x, dict)],
            "placeholder_files_created": [x.get("path") for x in report.get("placeholder_files_created", []) if isinstance(x, dict)],
            "files_written": [x.get("path") for x in report.get("files_written", []) if isinstance(x, dict)],
            "scope_enforced": bool(report.get("scope", {}).get("enforced")),
            "scope_violations": report.get("scope", {}).get("violations", []),
        }

    if not isinstance(code_output, dict):
        report["status"] = "error"
        report["errors"].append({"error": "code_output_not_dict"})
        _attach_verification_evidence()
        _trace_write_report(report)
        return report

    report["notes"] = str(code_output.get("notes") or "")

    def _materialize_allowed_dirs() -> None:
        if not enforce_allowed_paths:
            return
        if os.getenv("LOCAL_FILE_WRITER_MATERIALIZE_DIRS", "1").strip().lower() in {"0", "false", "no", "off"}:
            return
        allowed = report.get("scope", {}).get("allowed_paths", [])
        for dir_s in _allowed_directory_scopes(allowed):
            rel_dir = Path(dir_s)
            target_dir, err = _resolve_inside_workspace(workspace, rel_dir)
            if err or target_dir is None:
                report["files_skipped"].append({"path": dir_s + "/", "reason": err or "invalid_directory_scope"})
                continue
            try:
                existed_before = target_dir.exists()
                target_dir.mkdir(parents=True, exist_ok=True)
                report["directories_created"].append({
                    "path": dir_s + "/",
                    "absolute_path": str(target_dir),
                    "existed_before": existed_before,
                })
                if os.getenv("LOCAL_FILE_WRITER_GITKEEP_EMPTY_DIRS", "1").strip().lower() not in {"0", "false", "no", "off"}:
                    gitkeep_rel = Path(dir_s) / ".gitkeep"
                    if _path_allowed_by_scope(gitkeep_rel, allowed):
                        gitkeep_path = target_dir / ".gitkeep"
                        g_existed = gitkeep_path.exists()
                        if not g_existed:
                            gitkeep_path.write_text("", encoding="utf-8")
                        report["placeholder_files_created"].append({
                            "path": str(gitkeep_rel).replace("\\", "/"),
                            "absolute_path": str(gitkeep_path),
                            "existed_before": g_existed,
                            "new_sha256": _sha256_text(""),
                            "new_size_bytes": 0,
                        })
            except Exception as exc:
                report["files_skipped"].append({"path": dir_s + "/", "reason": f"directory_materialize_error:{exc}"})

    if report["should_write_to_file"]:
        _materialize_allowed_dirs()

    if not report["should_write_to_file"]:
        report["status"] = "skipped"
        report["notes"] = report["notes"] or "Engineer output did not request local file writing."
        _attach_verification_evidence()
        _trace_write_report(report)
        return report

    files = code_output.get("files")
    if not isinstance(files, list) or not files:
        if report.get("directories_created") or report.get("placeholder_files_created"):
            report["status"] = "written"
            report["notes"] = report["notes"] or "Materialized allowed directory scopes/placeholders; no explicit file list was provided."
        else:
            report["status"] = "error"
            report["errors"].append({"error": "should_write_to_file_true_but_no_files"})
        _attach_verification_evidence()
        _trace_write_report(report)
        return report

    for idx, item in enumerate(files):
        if not isinstance(item, dict):
            report["files_skipped"].append({"index": idx, "reason": "file_item_not_dict"})
            continue

        raw_path = item.get("path")
        content = item.get("content")
        write_mode = str(item.get("write_mode") or "create_or_overwrite").strip()

        if write_mode not in ALLOWED_WRITE_MODES:
            report["files_skipped"].append({"index": idx, "path": raw_path, "reason": f"invalid_write_mode:{write_mode}"})
            continue

        if not isinstance(content, str):
            report["files_skipped"].append({"index": idx, "path": raw_path, "reason": "content_not_string"})
            continue

        rel_path, path_error = _safe_rel_path(str(raw_path or ""))
        if path_error or rel_path is None:
            reason = path_error or "invalid_path"
            report["files_skipped"].append({"index": idx, "path": raw_path, "reason": reason})
            report["errors"].append({"error": reason, "path": raw_path})
            continue

        allowed = report.get("scope", {}).get("allowed_paths", [])
        if enforce_allowed_paths and not _path_allowed_by_scope(rel_path, allowed):
            violation = {"index": idx, "path": str(rel_path), "reason": "path_outside_task_scope"}
            report["files_skipped"].append(violation)
            report["scope"]["violations"].append(violation)
            continue

        target, resolve_error = _resolve_inside_workspace(workspace, rel_path)
        if resolve_error or target is None:
            report["files_skipped"].append({"index": idx, "path": raw_path, "reason": resolve_error or "invalid_resolved_path"})
            continue

        existed_before = target.exists()
        if write_mode == "create" and existed_before:
            report["files_skipped"].append({"index": idx, "path": str(rel_path), "reason": "file_exists_create_mode"})
            continue
        if write_mode == "overwrite" and not existed_before:
            report["files_skipped"].append({"index": idx, "path": str(rel_path), "reason": "file_missing_overwrite_mode"})
            continue

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            previous_sha256 = None
            previous_size_bytes = None
            if existed_before:
                try:
                    old_text = target.read_text(encoding="utf-8")
                    previous_sha256 = _sha256_text(old_text)
                    previous_size_bytes = len(old_text.encode("utf-8"))
                except Exception:
                    previous_sha256 = "unreadable_previous_file"
                    previous_size_bytes = None

            target.write_text(content, encoding="utf-8")

            report["files_written"].append({
                "path": str(rel_path),
                "absolute_path": str(target),
                "write_mode": write_mode,
                "existed_before": existed_before,
                "previous_sha256": previous_sha256,
                "previous_size_bytes": previous_size_bytes,
                "new_sha256": _sha256_text(content),
                "new_size_bytes": len(content.encode("utf-8")),
            })
        except Exception as exc:
            report["files_skipped"].append({"index": idx, "path": str(rel_path), "reason": f"write_error:{exc}"})

    if report.get("scope", {}).get("violations"):
        report["status"] = "scope_error"
        report["errors"].append({"error": "path_outside_task_scope", "count": len(report["scope"]["violations"])})
    elif report["errors"]:
        report["status"] = "error"
    elif (report["files_written"] or report["directories_created"] or report["placeholder_files_created"]) and not report["files_skipped"]:
        report["status"] = "written"
    elif (report["files_written"] or report["directories_created"] or report["placeholder_files_created"]) and report["files_skipped"]:
        report["status"] = "partial"
    elif report["files_skipped"]:
        report["status"] = "error"
        if not report["errors"]:
            report["errors"].append({"error": "all_files_skipped", "count": len(report["files_skipped"])})
    else:
        report["status"] = "skipped"

    # Deterministic verification evidence that QA can use without demanding
    # shell-command output from the Engineer model.
    _attach_verification_evidence()

    _trace_write_report(report)
    return report


def write_code_output_report(report: Dict[str, Any], report_path: str | Path) -> Path:
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
