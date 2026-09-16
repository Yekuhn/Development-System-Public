"""
task_graph_validator.py

Deterministic Engineering Lead plan validator and repair helper for the Ascendant Path workflow.

This module checks task graph structure before orchestrator ingest and repairs
common LLM formatting/ordering defects so the UI does not die on recoverable
plan errors.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

_BLOCKED_TOP_LEVEL = {".git", ".venv", "venv", "env", "__pycache__", "node_modules", "outputs", "logs"}
_KNOWN_EXTENSIONLESS_FILES = {"Dockerfile", "Makefile", "Procfile", "LICENSE", "NOTICE", "README", ".gitignore"}
_REQUIRED_WORK_ITEM_FIELDS = [
    "task_id", "summary", "capabilities_required", "dependencies", "scope_in", "scope_out",
    "interfaces", "acceptance_criteria", "verification", "files_expected", "risk_notes",
]
_CODE_HINTS = {
    "implement", "create", "build", "write", "add", "update", "modify", "refactor", "fix",
    "frontend", "backend", "api", "component", "test", "script", "database", "migration",
}

@dataclass
class ValidationIssue:
    severity: str
    code: str
    message: str
    task_id: Optional[str] = None
    field: Optional[str] = None
    details: Dict[str, Any] = dc_field(default_factory=dict)
    def as_dict(self) -> Dict[str, Any]:
        return {"severity": self.severity, "code": self.code, "message": self.message, "task_id": self.task_id, "field": self.field, "details": self.details}

def _strip_path_annotation(raw: str) -> str:
    s = str(raw or "").strip().strip('"').strip("'").replace("\\", "/")
    s = re.sub(r"\s*\([^/()]*\)\s*$", "", s).strip()
    s = re.sub(r"\s+[—–-]\s+(a11y|ocr|summary|tests?|tweaks?|notes?).*$", "", s, flags=re.I).strip()
    return s

def _glob_to_directory(raw: str) -> str:
    s = str(raw or "").replace("\\", "/").strip()
    if not any(ch in s for ch in ["*", "?", "["]):
        return s
    idxs = [i for i in [s.find("*"), s.find("?"), s.find("[")] if i >= 0]
    cut = min(idxs) if idxs else len(s)
    prefix = s[:cut]
    if "/" in prefix:
        return prefix.rsplit("/", 1)[0] + "/"
    return ""

def normalize_files_expected_path(raw_path: str, *, allow_repair: bool = False) -> Tuple[Optional[str], Optional[str]]:
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None, "empty_path"
    cleaned = _strip_path_annotation(raw_path)
    if any(ch in cleaned for ch in ["*", "?", "["]):
        if not allow_repair:
            return None, "glob_patterns_not_supported"
        cleaned = _glob_to_directory(cleaned)
    if not cleaned:
        return None, "empty_path_after_cleanup"
    if "(" in cleaned or ")" in cleaned:
        return None, "path_contains_annotation_text"
    p = Path(cleaned)
    if p.is_absolute():
        return None, "absolute_paths_are_not_allowed"
    parts = p.parts
    if any(part in {"", ".", ".."} for part in parts):
        return None, "path_traversal_or_empty_component_not_allowed"
    if parts and parts[0] in _BLOCKED_TOP_LEVEL:
        return None, f"blocked_top_level_path:{parts[0]}"
    normalized = str(p).replace("\\", "/")
    if cleaned.endswith("/") and not normalized.endswith("/"):
        normalized += "/"
    return normalized, None

def _safe_rel_path(raw_path: str) -> Tuple[Optional[str], Optional[str]]:
    return normalize_files_expected_path(raw_path, allow_repair=False)

def _scope_overlaps(a: str, b: str) -> bool:
    a, b = a.strip().replace("\\", "/"), b.strip().replace("\\", "/")
    return bool(a and b and (a == b or (a.endswith("/") and b.startswith(a)) or (b.endswith("/") and a.startswith(b))))

def _looks_like_code_task(wi: Dict[str, Any]) -> bool:
    text = " ".join(str(wi.get(k, "")) for k in ["summary", "scope_in", "scope_out", "risk_notes"]).lower()
    text += " " + " ".join(str(x).lower() for x in wi.get("verification", []) if isinstance(x, str))
    text += " " + " ".join(str(x).lower() for x in wi.get("capabilities_required", []) if isinstance(x, str))
    return any(h in text for h in _CODE_HINTS)

def _has_dependency_path(src: str, dst: str, deps: Dict[str, List[str]]) -> bool:
    stack, seen = list(deps.get(src, [])), set()
    while stack:
        cur = stack.pop()
        if cur == dst:
            return True
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(deps.get(cur, []))
    return False

def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen, out = set(), []
    for x in items:
        s = str(x).strip()
        if s and s not in seen:
            seen.add(s); out.append(s)
    return out

def validate_task_graph(eng_lead_output: Dict[str, Any]) -> Dict[str, Any]:
    issues: List[ValidationIssue] = []
    if not isinstance(eng_lead_output, dict):
        return {"ok": False, "error_count": 1, "warning_count": 0, "issues": [ValidationIssue("error", "eng_lead_output_not_dict", "Engineering Lead output must be a dict.").as_dict()], "normalized_file_scopes": {}, "summary": {"task_count": 0}}
    raw_items = eng_lead_output.get("work_items")
    if not isinstance(raw_items, list) or not raw_items:
        return {"ok": False, "error_count": 1, "warning_count": 0, "issues": [ValidationIssue("error", "missing_work_items", "eng_lead_output.work_items must be a non-empty list.", field="work_items").as_dict()], "normalized_file_scopes": {}, "summary": {"task_count": 0}}
    task_by_id: Dict[str, Dict[str, Any]] = {}
    deps: Dict[str, List[str]] = {}
    normalized_scopes: Dict[str, List[str]] = {}
    for idx, item in enumerate(raw_items):
        task_label = f"index_{idx}"
        if not isinstance(item, dict):
            issues.append(ValidationIssue("error", "work_item_not_dict", "Each work_item must be a dict.", task_id=task_label)); continue
        task_id = str(item.get("task_id", "")).strip() or task_label
        if task_id == task_label:
            issues.append(ValidationIssue("error", "empty_task_id", "work_item.task_id cannot be empty.", task_id=task_label, field="task_id"))
        if task_id in task_by_id:
            issues.append(ValidationIssue("error", "duplicate_task_id", f"Duplicate task_id: {task_id}", task_id=task_id, field="task_id"))
        task_by_id[task_id] = item
        for field_name in _REQUIRED_WORK_ITEM_FIELDS:
            if field_name not in item:
                issues.append(ValidationIssue("error", "missing_required_field", f"work_item missing required field: {field_name}", task_id=task_id, field=field_name))
        for list_field in ["capabilities_required", "dependencies", "interfaces", "acceptance_criteria", "verification", "files_expected"]:
            if list_field in item and not isinstance(item.get(list_field), list):
                issues.append(ValidationIssue("error", "field_must_be_list", f"{list_field} must be a list.", task_id=task_id, field=list_field))
        dep_list = [str(x).strip() for x in item.get("dependencies", []) if str(x).strip()] if isinstance(item.get("dependencies"), list) else []
        deps[task_id] = _dedupe_preserve_order(dep_list)
        files_expected = item.get("files_expected") if isinstance(item.get("files_expected"), list) else []
        if not files_expected and _looks_like_code_task(item):
            issues.append(ValidationIssue("warning", "code_task_has_empty_files_expected", "Code-like task has empty files_expected; local file writer may reject writes for this task.", task_id=task_id, field="files_expected"))
        norm_paths = []
        for raw in files_expected:
            norm, err = _safe_rel_path(str(raw or ""))
            if err or norm is None:
                issues.append(ValidationIssue("error", "invalid_files_expected_path", f"Invalid files_expected path: {raw!r} ({err})", task_id=task_id, field="files_expected", details={"path": raw, "reason": err})); continue
            name = Path(norm.rstrip("/")).name
            if "." not in name and not norm.endswith("/") and name not in _KNOWN_EXTENSIONLESS_FILES:
                issues.append(ValidationIssue("warning", "ambiguous_directory_scope", f"files_expected path {norm!r} looks like a directory but does not end with '/'.", task_id=task_id, field="files_expected", details={"path": norm}))
            norm_paths.append(norm)
        normalized_scopes[task_id] = sorted(set(norm_paths))
        verification = item.get("verification") if isinstance(item.get("verification"), list) else []
        if not verification:
            issues.append(ValidationIssue("warning", "missing_verification", "Task has no verification steps.", task_id=task_id, field="verification"))
    known = set(task_by_id.keys())
    for tid, dep_list in deps.items():
        for dep in dep_list:
            if dep not in known:
                issues.append(ValidationIssue("error", "missing_dependency", f"Task depends on unknown task_id: {dep}", task_id=tid, field="dependencies", details={"dependency": dep}))
            if dep == tid:
                issues.append(ValidationIssue("error", "self_dependency", "Task cannot depend on itself.", task_id=tid, field="dependencies"))
    visiting, visited, path = set(), set(), []
    def dfs(tid: str) -> None:
        if tid in visited: return
        if tid in visiting:
            try: cycle = path[path.index(tid):] + [tid]
            except ValueError: cycle = path + [tid]
            issues.append(ValidationIssue("error", "dependency_cycle", "Dependency cycle detected.", task_id=tid, field="dependencies", details={"cycle": cycle})); return
        visiting.add(tid); path.append(tid)
        for dep in deps.get(tid, []):
            if dep in known: dfs(dep)
        path.pop(); visiting.remove(tid); visited.add(tid)
    for tid in list(known): dfs(tid)
    tids = sorted(normalized_scopes.keys())
    for i, a in enumerate(tids):
        for b in tids[i + 1:]:
            if _has_dependency_path(a, b, deps) or _has_dependency_path(b, a, deps): continue
            for pa in normalized_scopes.get(a, []):
                for pb in normalized_scopes.get(b, []):
                    if not _scope_overlaps(pa, pb): continue
                    severity = "error" if pa == pb and not pa.endswith("/") else "warning"
                    code = "same_file_owned_by_parallel_tasks" if severity == "error" else "overlapping_file_scope_without_dependency"
                    issues.append(ValidationIssue(severity, code, f"Tasks {a} and {b} overlap on file scope without a dependency edge.", task_id=a, field="files_expected", details={"other_task_id": b, "path_a": pa, "path_b": pb}))
    issue_dicts = [issue.as_dict() for issue in issues]
    error_count = sum(1 for issue in issue_dicts if issue.get("severity") == "error")
    warning_count = sum(1 for issue in issue_dicts if issue.get("severity") == "warning")
    return {"ok": error_count == 0, "error_count": error_count, "warning_count": warning_count, "issues": issue_dicts, "normalized_file_scopes": normalized_scopes, "summary": {"task_count": len(task_by_id), "tasks_with_files_expected": sum(1 for paths in normalized_scopes.values() if paths), "tasks_with_dependencies": sum(1 for d in deps.values() if d)}}

def _ensure_work_item_shape(wi: Dict[str, Any], idx: int) -> None:
    wi.setdefault("task_id", f"T{idx+1:02d}"); wi.setdefault("summary", f"Engineering task {idx+1}"); wi.setdefault("capabilities_required", ["generalist"]); wi.setdefault("dependencies", []); wi.setdefault("scope_in", ""); wi.setdefault("scope_out", ""); wi.setdefault("interfaces", []); wi.setdefault("acceptance_criteria", []); wi.setdefault("verification", []); wi.setdefault("files_expected", []); wi.setdefault("risk_notes", "")
    for key in ["capabilities_required", "dependencies", "interfaces", "acceptance_criteria", "verification", "files_expected"]:
        if not isinstance(wi.get(key), list): wi[key] = [str(wi.get(key))] if wi.get(key) is not None else []

def _break_cycles(items: List[Dict[str, Any]]) -> List[str]:
    changes = []
    for _ in range(10):
        report = validate_task_graph({"work_items": items})
        cycles = [i for i in report.get("issues", []) if i.get("code") == "dependency_cycle"]
        if not cycles: break
        by_id = {str(w.get("task_id")): w for w in items if isinstance(w, dict)}
        changed = False
        for issue in cycles:
            cycle = (issue.get("details") or {}).get("cycle") or []
            if len(cycle) < 2: continue
            owner, dep = str(cycle[-2]), str(cycle[-1])
            wi = by_id.get(owner)
            if wi and isinstance(wi.get("dependencies"), list) and dep in wi["dependencies"]:
                wi["dependencies"] = [x for x in wi["dependencies"] if x != dep]
                changes.append(f"Removed cycle dependency {owner}->{dep}."); changed = True; break
        if not changed: break
    return changes

def repair_task_graph(eng_lead_output: Dict[str, Any], *, max_passes: int = 5) -> Dict[str, Any]:
    repaired = copy.deepcopy(eng_lead_output) if isinstance(eng_lead_output, dict) else {}
    changes: List[str] = []
    items = repaired.get("work_items")
    if not isinstance(items, list):
        return {"ok": False, "repaired": False, "changes": changes, "eng_lead_output": repaired, "validation_report": validate_task_graph(repaired)}
    seen_ids, id_map = set(), {}
    for idx, wi in enumerate(items):
        if not isinstance(wi, dict): items[idx] = {"task_id": f"T{idx+1:02d}", "summary": str(wi)}; wi = items[idx]
        old_id = str(wi.get("task_id", "")).strip() or f"T{idx+1:02d}"; new_id = old_id
        if new_id in seen_ids: new_id = f"{old_id}_{idx+1}"; changes.append(f"Renamed duplicate task_id {old_id} -> {new_id}.")
        wi["task_id"] = new_id; seen_ids.add(new_id); id_map[old_id] = new_id; _ensure_work_item_shape(wi, idx)
    known = {str(w.get("task_id")) for w in items if isinstance(w, dict)}
    for wi in items:
        if not isinstance(wi, dict): continue
        tid = str(wi.get("task_id")); deps = []
        for d in wi.get("dependencies", []):
            ds = id_map.get(str(d).strip(), str(d).strip())
            if not ds or ds == tid or ds not in known:
                if ds: changes.append(f"Removed invalid dependency {tid}->{ds}.")
                continue
            deps.append(ds)
        wi["dependencies"] = _dedupe_preserve_order(deps)
        clean_paths = []
        for raw in wi.get("files_expected", []):
            norm, err = normalize_files_expected_path(str(raw or ""), allow_repair=True)
            if norm:
                clean_paths.append(norm)
                if str(raw).strip() != norm: changes.append(f"Cleaned files_expected for {tid}: {raw!r} -> {norm!r}.")
            else:
                changes.append(f"Dropped invalid files_expected for {tid}: {raw!r} ({err}).")
        wi["files_expected"] = _dedupe_preserve_order(clean_paths)
    for _ in range(max(1, int(max_passes))):
        report = validate_task_graph(repaired)
        errors = [i for i in report.get("issues", []) if i.get("severity") == "error"]
        if not errors: break
        by_id = {str(w.get("task_id")): w for w in items if isinstance(w, dict)}
        order = {str(w.get("task_id")): i for i, w in enumerate(items) if isinstance(w, dict)}
        changed = False
        for issue in errors:
            code, tid, details = issue.get("code"), str(issue.get("task_id") or ""), issue.get("details") or {}
            if code == "same_file_owned_by_parallel_tasks":
                other = str(details.get("other_task_id") or "")
                if tid in by_id and other in by_id:
                    later, earlier = (other, tid) if order.get(tid, 0) <= order.get(other, 0) else (tid, other)
                    deps = by_id[later].setdefault("dependencies", [])
                    if earlier not in deps: deps.append(earlier); changes.append(f"Added dependency {later}->{earlier} to serialize shared file ownership."); changed = True
            elif code == "invalid_files_expected_path":
                raw = details.get("path"); wi = by_id.get(tid)
                if wi and isinstance(wi.get("files_expected"), list): wi["files_expected"] = [p for p in wi["files_expected"] if p != raw]; changes.append(f"Dropped still-invalid path for {tid}: {raw!r}."); changed = True
            elif code in {"missing_dependency", "self_dependency"}:
                wi = by_id.get(tid); bad = str(details.get("dependency") or tid)
                if wi and isinstance(wi.get("dependencies"), list): wi["dependencies"] = [d for d in wi["dependencies"] if str(d) != bad and str(d) != tid]; changes.append(f"Removed invalid dependency from {tid}: {bad}."); changed = True
        cyc = _break_cycles(items); changes.extend(cyc); changed = changed or bool(cyc)
        if validate_task_graph(repaired).get("ok") or not changed: break
    final_report = validate_task_graph(repaired)
    return {"ok": bool(final_report.get("ok")), "repaired": bool(changes), "changes": changes, "eng_lead_output": repaired, "validation_report": final_report}

def validate_task_graph_file(path: str | Path) -> Dict[str, Any]:
    return validate_task_graph(json.loads(Path(path).read_text(encoding="utf-8")))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate or repair an Engineering Lead work_items task graph.")
    parser.add_argument("path"); parser.add_argument("--repair", action="store_true")
    args = parser.parse_args(); obj = json.loads(Path(args.path).read_text(encoding="utf-8"))
    result = repair_task_graph(obj) if args.repair else validate_task_graph(obj)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result.get("ok") else 1)
