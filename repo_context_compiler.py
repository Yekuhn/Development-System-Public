"""
repo_context_compiler.py

Deterministic workspace-context compiler for Engineer tasks.

Before an Engineer is asked to write or overwrite files, this module reads the
current workspace state relevant to that WorkItem and dependency results. This
prevents sequential full-file overwrites from accidentally deleting earlier work.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


_BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".webp", ".pdf", ".zip", ".gz", ".tar",
    ".woff", ".woff2", ".ttf", ".otf", ".pyc", ".class", ".dll", ".so", ".dylib",
}
_BLOCKED_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "env", "outputs", "logs"}


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _safe_rel_path(raw_path: str) -> Tuple[Optional[Path], Optional[str]]:
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None, "empty_path"
    raw = raw_path.strip().replace("\\", "/")
    if any(ch in raw for ch in ["*", "?", "["]):
        return None, "glob_patterns_not_supported"
    p = Path(raw)
    if p.is_absolute():
        return None, "absolute_paths_are_not_allowed"
    if any(part in {"", ".", ".."} for part in p.parts):
        return None, "path_traversal_or_empty_component_not_allowed"
    if p.parts and p.parts[0] in _BLOCKED_DIRS:
        return None, f"blocked_top_level_path:{p.parts[0]}"
    return p, None


def _is_probably_binary(path: Path) -> bool:
    return path.suffix.lower() in _BINARY_EXTENSIONS


def _read_text_limited(path: Path, max_chars: int) -> Dict[str, Any]:
    try:
        if _is_probably_binary(path):
            return {"path": str(path), "included": False, "reason": "binary_or_asset_file", "size_bytes": path.stat().st_size}
        text = path.read_text(encoding="utf-8")
        truncated = len(text) > max_chars
        content = text[:max_chars]
        return {
            "path": str(path),
            "included": True,
            "truncated": truncated,
            "size_bytes": len(text.encode("utf-8")),
            "sha256": _sha256_text(text),
            "content": content,
        }
    except UnicodeDecodeError:
        return {"path": str(path), "included": False, "reason": "not_utf8_text", "size_bytes": path.stat().st_size if path.exists() else None}
    except Exception as exc:
        return {"path": str(path), "included": False, "reason": f"read_error:{exc}"}


def _iter_workspace_files(workspace: Path, limit: int = 250) -> List[str]:
    if not workspace.exists():
        return []
    out: List[str] = []
    for p in sorted(workspace.rglob("*")):
        try:
            rel = p.relative_to(workspace)
        except ValueError:
            continue
        if any(part in _BLOCKED_DIRS for part in rel.parts):
            continue
        if p.is_file():
            out.append(str(rel).replace("\\", "/"))
        if len(out) >= limit:
            out.append(f"... truncated after {limit} files ...")
            break
    return out


def _resolve_inside_workspace(workspace: Path, rel: Path) -> Optional[Path]:
    root = workspace.resolve()
    target = (workspace / rel).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        return None
    return target


def compile_task_context(
    *,
    work_item: Dict[str, Any],
    workspace_dir: str | Path,
    orchestrator_results: Optional[Dict[str, Any]] = None,
    max_files: int = 20,
    max_chars_per_file: int = 12000,
    max_total_chars: int = 60000,
) -> Dict[str, Any]:
    """Return a compact, JSON-safe snapshot relevant to the current WorkItem."""
    workspace = Path(workspace_dir)
    task_id = str(work_item.get("task_id", "")) if isinstance(work_item, dict) else ""
    expected = work_item.get("files_expected", []) if isinstance(work_item, dict) else []
    expected = expected if isinstance(expected, list) else []

    snapshots: List[Dict[str, Any]] = []
    path_errors: List[Dict[str, Any]] = []
    total_chars = 0

    for raw in expected:
        if len(snapshots) >= max_files or total_chars >= max_total_chars:
            break
        rel, err = _safe_rel_path(str(raw or ""))
        if err or rel is None:
            path_errors.append({"path": raw, "reason": err or "invalid_path"})
            continue
        target = _resolve_inside_workspace(workspace, rel)
        if target is None:
            path_errors.append({"path": raw, "reason": "resolved_path_escapes_workspace"})
            continue
        if str(raw).strip().replace("\\", "/").endswith("/") or target.is_dir():
            if not target.exists():
                snapshots.append({"path": str(rel).replace("\\", "/"), "exists": False, "scope_type": "directory"})
                continue
            for child in sorted(target.rglob("*")):
                if len(snapshots) >= max_files or total_chars >= max_total_chars:
                    break
                try:
                    rel_child = child.relative_to(workspace)
                except ValueError:
                    continue
                if any(part in _BLOCKED_DIRS for part in rel_child.parts):
                    continue
                if not child.is_file():
                    continue
                item = _read_text_limited(child, max_chars_per_file)
                item["path"] = str(rel_child).replace("\\", "/")
                item["exists"] = True
                item["scope_type"] = "directory_child"
                if item.get("included"):
                    total_chars += len(str(item.get("content", "")))
                snapshots.append(item)
        else:
            if not target.exists():
                snapshots.append({"path": str(rel).replace("\\", "/"), "exists": False, "scope_type": "file"})
                continue
            item = _read_text_limited(target, max_chars_per_file)
            item["path"] = str(rel).replace("\\", "/")
            item["exists"] = True
            item["scope_type"] = "file"
            if item.get("included"):
                total_chars += len(str(item.get("content", "")))
            snapshots.append(item)

    dep_ids = [str(x) for x in work_item.get("dependencies", []) if str(x).strip()] if isinstance(work_item, dict) else []
    dependency_results: Dict[str, Any] = {}
    if isinstance(orchestrator_results, dict):
        raw_results = orchestrator_results.get("results") if isinstance(orchestrator_results.get("results"), dict) else orchestrator_results
        if isinstance(raw_results, dict):
            for dep_id in dep_ids:
                if dep_id in raw_results:
                    res = raw_results[dep_id]
                    dependency_results[dep_id] = {
                        "engineer_id": res.get("engineer_id") if isinstance(res, dict) else None,
                        "verification_run": res.get("verification_run") if isinstance(res, dict) else None,
                        "handoff_interfaces": res.get("handoff_interfaces") if isinstance(res, dict) else None,
                        "notes": res.get("notes") if isinstance(res, dict) else None,
                    }

    existing_expected_files = [s.get("path") for s in snapshots if s.get("exists") and s.get("scope_type") == "file"]
    return {
        "schema_version": "repo_context_compiler.v1",
        "task_id": task_id,
        "workspace_dir": str(workspace),
        "workspace_exists": workspace.exists(),
        "workspace_file_tree_sample": _iter_workspace_files(workspace),
        "expected_file_snapshots": snapshots,
        "path_errors": path_errors,
        "dependency_results": dependency_results,
        "overwrite_policy": {
            "full_file_overwrite_requires_reading_current_file": True,
            "existing_expected_files": existing_expected_files,
            "instruction": "If you overwrite an existing file, preserve compatible existing behavior unless the WorkItem or QA feedback explicitly requires changing it.",
        },
        "limits": {
            "max_files": max_files,
            "max_chars_per_file": max_chars_per_file,
            "max_total_chars": max_total_chars,
            "actual_total_chars": total_chars,
        },
    }
