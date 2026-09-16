# operation.py
"""
Ascendant Path agentic software workflow runner.

Workflow:
  1) Team Lead intake (supervised UI) -> produces FINAL initial_input -> write initial_input.json
  2) PM -> UX -> Eng Lead -> Eng Orchestrator
  3) N Engineer workers + QA loop -> submit results back to orchestrator
  4) Workflow Coordinator summarizes & produces final deliverables plan

Important design:
- supervise_ui is GENERIC. It only supervises an agent.
- Operation.py decides what to do after approval.

This file assumes these local modules exist:
  supervise_ui.py, record.py, team_lead_agent.py, pm_agent.py, ux_agent.py,
  eng_lead_agent.py, eng_orchestrator_agent.py, eng_agent.py, qa_agent.py, coordinator_agent.py
"""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List, Iterable

from record import Recorder
from supervise_ui import supervise_ui
from resource_eval import ResourceEval

from team_lead_agent import TeamLeadAgent
from pm_agent import PMAgent
from ux_agent import UXDesignerAgent
from eng_lead_agent import EngineeringLeadAgent
from eng_orchestrator_agent import EngineeringOrchestratorAgent
from eng_agent import EngineerAgent
from qa_agent import QAAgent
from coordinator_agent import CoordinatorAgent
from local_file_writer import write_code_output
from trace_utils import trace_event
from preflight import run_preflight
from task_graph_validator import validate_task_graph, repair_task_graph
from repo_context_compiler import compile_task_context
from project_executor import run_project_executor
from human_requests import HumanRequestDesk
from run_state import (
    RunState,
    NODE_INTAKE,
    NODE_PM,
    NODE_UX,
    NODE_ENG_LEAD,
    NODE_ORCHESTRATOR_INGEST,
    NODE_ENGINEERING,
    NODE_EXECUTOR,
    NODE_RUNBOOK,
    NODE_COORDINATOR_FINAL_HANDOFF,
)


# -------------------------
# Utilities
# -------------------------

def _ensure_openai_key() -> None:
    """Load OPENAI_API_KEY from env or ./openai_api_key file if available.

    Provider routing may use Ollama/browser-only stages in some run modes, so this
    function no longer fails immediately. OpenAI provider calls will still fail
    clearly if a selected stage requires OpenAI and no key is available.
    """
    if os.getenv("OPENAI_API_KEY", "").strip():
        return
    p = Path("openai_api_key")
    if p.exists():
        os.environ["OPENAI_API_KEY"] = p.read_text(encoding="utf-8").strip()


def _read_json_best_effort(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def _write_pretty_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _is_mvp_local_simple_app(initial_input: Dict[str, Any]) -> bool:
    try:
        text = json.dumps(initial_input, ensure_ascii=False).lower()
    except Exception:
        text = str(initial_input).lower()
    markers = (
        "mvp_local_simple_web_app",
        "simple local mvp",
        "simple local web app",
        "local-only mvp",
        "local only mvp",
        "not a giant industry-level product",
        "not an enterprise-grade production system",
    )
    return any(m in text for m in markers)


def _mvp_bloat_terms() -> List[str]:
    return [
        "ocr", "tesseract", "image upload", "screenshot", "ci", "github actions",
        "docker", "compose", "playwright", "e2e", "end-to-end", "sse", "streaming",
        "benchmark", "kaggle", "runbook", "architecture doc", "performance doc",
        "observability", "metrics", "tracing", "deployment", "production", "container",
    ]


def _text_mentions_active_bloat(value: Any) -> bool:
    """Return True only when V2/enterprise terms appear as active V1 work.

    MVP plans should be allowed to mention OCR/Docker/CI/etc. in scope_out,
    non_goals, or deferral notes. The previous gate scanned the whole JSON blob,
    so a good task saying "scope_out: Docker" was treated as overbuilt and
    could be replaced unnecessarily.
    """
    try:
        text = json.dumps(value, ensure_ascii=False).lower() if not isinstance(value, str) else value.lower()
    except Exception:
        text = str(value).lower()
    if not any(term in text for term in _mvp_bloat_terms()):
        return False
    negators = (
        "scope_out", "scope out", "out of scope", "non_goal", "non-goal",
        "not required", "not include", "do not include", "do not create",
        "defer", "deferred", "v2", "future version", "later", "unless explicitly requested",
    )
    return not any(n in text for n in negators)


def _work_item_mentions_bloat(item: Dict[str, Any]) -> bool:
    if not isinstance(item, dict):
        return False
    # Only active-delivery fields should trigger the hard gate. Negative fields
    # such as scope_out/risk_notes are allowed to name deferred V2 features.
    active_fields = (
        "summary", "scope_in", "interfaces", "acceptance_criteria",
        "verification", "files_expected", "capabilities_required",
    )
    for key in active_fields:
        if key in item and _text_mentions_active_bloat(item.get(key)):
            return True
    return False


def _is_sudoku_project(initial_input: Dict[str, Any]) -> bool:
    try:
        text = json.dumps(initial_input, ensure_ascii=False).lower()
    except Exception:
        text = str(initial_input).lower()
    return "sudoku" in text


def _canonical_local_mvp_plan(initial_input: Dict[str, Any]) -> Dict[str, Any]:
    """Return a small generic MVP plan, with a Sudoku-specific plan only for Sudoku briefs."""
    if not _is_sudoku_project(initial_input):
        try:
            brief = str(initial_input.get("brief") or "local web app").strip()
        except Exception:
            brief = "local web app"
        if len(brief) > 240:
            brief = brief[:237].rstrip() + "..."
        return {
            "review_feedback": "Engineering plan was normalized to a small local MVP to match the user-approved scope and avoid enterprise bloat.",
            "architecture_plan": "Local-only Python FastAPI backend plus React + TypeScript + Vite frontend. Keep V1 to the smallest working app described in the brief; no production deployment or enterprise infrastructure.",
            "execution_plan": "Build the backend/API first, then the frontend/API integration, then one local README with run commands and smoke checks.",
            "work_items": [
                {
                    "task_id": "ENG-01",
                    "summary": "Implement the minimal backend/API for the local MVP described by the product brief.",
                    "capabilities_required": ["python", "fastapi", "backend"],
                    "dependencies": [],
                    "scope_in": f"Minimal backend/API behavior needed for: {brief}",
                    "scope_out": "OCR, CI, Docker, Playwright, SSE, benchmarks, auth, persistence, production deployment, and broad documentation.",
                    "interfaces": ["GET /health returns ok", "Minimal API endpoints needed by the frontend"],
                    "acceptance_criteria": ["Backend runs locally", "Core V1 action returns a clear success or validation error", "No nonessential external services are required"],
                    "verification": ["python -m compileall backend", "manual local smoke test for /health and the core API"],
                    "files_expected": ["backend/"],
                    "risk_notes": "Keep backend deterministic and minimal; defer expensive integrations to V2."
                },
                {
                    "task_id": "ENG-02",
                    "summary": "Implement the minimal React frontend and connect it to the backend API.",
                    "capabilities_required": ["react", "typescript", "frontend"],
                    "dependencies": ["ENG-01"],
                    "scope_in": f"Simple local UI for the V1 flow described by: {brief}",
                    "scope_out": "Advanced animations, production design system, Playwright/e2e tests, and deployment polish.",
                    "interfaces": ["Calls the backend health/core endpoints", "Displays success/error states clearly"],
                    "acceptance_criteria": ["User can complete the core V1 flow locally", "Frontend displays backend results or clear errors", "UI is usable on localhost without authentication"],
                    "verification": ["npm install", "npm run build"],
                    "files_expected": ["frontend/"],
                    "risk_notes": "Prioritize a working local flow over styling complexity."
                },
                {
                    "task_id": "ENG-03",
                    "summary": "Add minimal local run instructions and smoke-check notes.",
                    "capabilities_required": ["documentation", "developer-experience"],
                    "dependencies": ["ENG-01", "ENG-02"],
                    "scope_in": "One README with local setup, backend/frontend commands, and one simple smoke-check scenario.",
                    "scope_out": "Architecture docs, QA docs, performance docs, runbooks, CI matrices, and deployment docs.",
                    "interfaces": ["README documents local backend and frontend ports"],
                    "acceptance_criteria": ["README lets a local user start backend and frontend", "README contains one smoke-check path"],
                    "verification": ["Review README commands for consistency with generated files"],
                    "files_expected": ["README.md"],
                    "risk_notes": "Do not expand documentation beyond local run instructions."
                }
            ],
            "risks_and_mitigations": "Risk: scope creep reintroduces enterprise features; mitigation: defer OCR/CI/Docker/Playwright/SSE/benchmarks/docs/deployment to V2 unless explicitly requested.",
            "open_questions": "None for V1; use standard local ports unless occupied.",
            "definition_of_done": "Local backend and frontend run, the core V1 flow works, success/error states display, and README gives local commands."
        }

    return {
        "review_feedback": "Engineering plan was normalized to a small local MVP to match the user-approved scope and avoid enterprise bloat.",
        "architecture_plan": "Local-only FastAPI backend exposes validation and solve endpoints using a Norvig-style Python solver; React + TypeScript + Vite frontend provides manual 9x9 entry, solve action, solution display, and a concise solving log.",
        "execution_plan": "Build backend solver/API first, then frontend UI/API integration, then one local README with run commands and smoke checks.",
        "work_items": [
            {
                "task_id": "ENG-01",
                "summary": "Implement Python Sudoku validation, Norvig-style solver, and FastAPI endpoints for local MVP.",
                "capabilities_required": ["python", "fastapi", "algorithm"],
                "dependencies": [],
                "scope_in": "Backend package, puzzle validation, solve logic, /health, /api/validate, and /api/solve endpoints.",
                "scope_out": "OCR, Tesseract, CI, Docker, SSE, benchmarks, persistence, auth, production deployment.",
                "interfaces": ["POST /api/validate accepts 81-char/string or 9x9 grid and returns validity/errors", "POST /api/solve returns solved grid and concise log", "GET /health returns ok"],
                "acceptance_criteria": ["Valid puzzles solve correctly", "Invalid format and contradiction cases return clear errors", "Backend can run locally with uvicorn"],
                "verification": ["python -m compileall backend", "manual curl or local smoke test for /health and /api/solve"],
                "files_expected": ["backend/"],
                "risk_notes": "Keep solver deterministic and simple; no image/OCR dependencies in V1."
            },
            {
                "task_id": "ENG-02",
                "summary": "Implement React manual Sudoku entry UI and connect it to the FastAPI backend.",
                "capabilities_required": ["react", "typescript", "frontend"],
                "dependencies": ["ENG-01"],
                "scope_in": "Vite React app, 9x9 input grid, validation feedback, solve button, solution display, and concise solving log panel.",
                "scope_out": "Animations, candidate visualization, OCR upload, Playwright/e2e tests, production design system.",
                "interfaces": ["Calls /api/validate and /api/solve", "Displays backend errors and solved grid"],
                "acceptance_criteria": ["User can enter a puzzle manually", "Solve button displays solution or clear error", "UI is usable on localhost without authentication"],
                "verification": ["npm install", "npm run build"],
                "files_expected": ["frontend/"],
                "risk_notes": "Prioritize a working local flow over styling complexity."
            },
            {
                "task_id": "ENG-03",
                "summary": "Add minimal local run instructions and smoke-check notes.",
                "capabilities_required": ["documentation", "developer-experience"],
                "dependencies": ["ENG-01", "ENG-02"],
                "scope_in": "One README with backend/frontend setup, run commands, and a sample puzzle.",
                "scope_out": "Architecture docs, QA docs, performance docs, runbooks, CI matrices, deployment docs.",
                "interfaces": ["README documents local backend and frontend ports"],
                "acceptance_criteria": ["README lets a local user start backend and frontend", "README includes one sample puzzle and expected behavior"],
                "verification": ["Review README commands for consistency with generated files"],
                "files_expected": ["README.md"],
                "risk_notes": "Do not expand documentation beyond local run instructions."
            }
        ],
        "risks_and_mitigations": "Risk: old scope creep reintroduces enterprise features; mitigation: defer OCR/CI/Docker/Playwright/SSE/benchmarks/docs to V2.",
        "open_questions": "None for V1; use standard local ports unless occupied.",
        "definition_of_done": "Local backend and frontend run, user enters a Sudoku puzzle, validation/solve works, solution/log appear, and README gives local commands."
    }


def _enforce_mvp_engineering_plan(initial_input: Dict[str, Any], eng_lead_out: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Hard gate to keep simple MVP runs from becoming enterprise task graphs."""
    report: Dict[str, Any] = {"applied": False, "reason": "not_mvp_or_no_bloat", "changes": []}
    if not _is_mvp_local_simple_app(initial_input) or not isinstance(eng_lead_out, dict):
        return eng_lead_out, report

    items = eng_lead_out.get("work_items") if isinstance(eng_lead_out.get("work_items"), list) else []
    bloat_items = [str(x.get("task_id") or i) for i, x in enumerate(items) if isinstance(x, dict) and _work_item_mentions_bloat(x)]
    too_many_tasks = len(items) > 6
    # Look for active bloat only in fields that define what V1 will build;
    # allow scope_out/non_goal text to mention deferred V2 features.
    active_plan_fields = [eng_lead_out.get(k) for k in ("architecture_plan", "execution_plan", "definition_of_done")]
    bloat_present = any(_text_mentions_active_bloat(v) for v in active_plan_fields) or bool(bloat_items)
    if not too_many_tasks and not bloat_items and not bloat_present:
        return eng_lead_out, report

    pruned: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if _work_item_mentions_bloat(item):
            continue
        pruned.append(item)
        if len(pruned) >= 6:
            break

    # If the remaining plan is too thin or still risky, replace it with a known-safe V1 plan.
    if len(pruned) < 2:
        new_plan = _canonical_local_mvp_plan(initial_input)
        report.update({"applied": True, "reason": "replaced_with_canonical_mvp_plan", "changes": ["Replaced overbuilt Engineering Lead plan with canonical 3-task local MVP plan."], "removed_or_deferred_items": bloat_items, "original_task_count": len(items), "new_task_count": len(new_plan["work_items"])})
        return new_plan, report

    new_out = dict(eng_lead_out)
    new_out["work_items"] = pruned
    new_out["review_feedback"] = str(new_out.get("review_feedback") or "") + "\nMVP hard gate applied: deferred enterprise/V2 items."
    new_out["architecture_plan"] = "Local-only MVP architecture. " + str(new_out.get("architecture_plan") or "")
    new_out["execution_plan"] = "Keep V1 to at most 6 local MVP tasks; defer OCR/CI/Docker/Playwright/SSE/benchmarks/docs/deployment to V2. " + str(new_out.get("execution_plan") or "")
    new_out["risks_and_mitigations"] = str(new_out.get("risks_and_mitigations") or "") + "\nScope creep risk mitigated by MVP hard gate."
    new_out["open_questions"] = str(new_out.get("open_questions") or "None for V1.")
    
    if _is_sudoku_project(initial_input):
        done_note = "V1 done means local manual-input Sudoku solve flow works; V2 features are not required."
    else:
        done_note = "V1 done means the core local MVP flow works end-to-end; V2/enterprise features are not required."
    new_out["definition_of_done"] = str(new_out.get("definition_of_done") or "") + "\n" + done_note
    report.update({"applied": True, "reason": "pruned_overbuilt_mvp_plan", "changes": ["Removed work_items mentioning deferred/V2 enterprise features.", "Capped work_items at 6."], "removed_or_deferred_items": bloat_items, "original_task_count": len(items), "new_task_count": len(pruned)})
    return new_out, report


def _read_team_lead_directives(
    resource_dir: str,
    *,
    target_agents: Optional[List[str]] = None,
    task_id: Optional[str] = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """Read Team Lead-routed user directives for downstream agents.

    Agent Inspector messages are intentionally not delivered directly to PM/UX/
    Engineering/QA. supervise_ui records them as Team Lead directives in the
    resources directory. This helper lets each stage receive only relevant
    directives through its normal agent_input packet, preserving the centralized
    Team Lead routing model while ensuring targeted guidance is actually used.
    """
    try:
        base = Path(resource_dir)
        wanted = {str(x) for x in (target_agents or []) if str(x)}
        out: List[Dict[str, Any]] = []

        # Prefer JSONL event history because it preserves multiple directives.
        jsonl = base / "team_lead_directives.jsonl"
        if jsonl.exists():
            for line in jsonl.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                target = str(obj.get("target_agent") or "")
                obj_task = obj.get("target_task_id")
                if wanted and target not in wanted and target != "Team Lead / Intake":
                    continue
                if task_id and obj_task not in {task_id, None, ""}:
                    continue
                out.append(obj)

        # Fallback to per-directive files if the JSONL log is absent/incomplete.
        d = base / "team_lead_directives"
        if d.exists():
            seen = {str(x.get("created_at_utc")) + "|" + str(x.get("target_agent")) + "|" + str(x.get("user_message")) for x in out}
            for fp in sorted(d.glob("*.json"), key=lambda q: q.stat().st_mtime):
                try:
                    obj = json.loads(fp.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                target = str(obj.get("target_agent") or "")
                obj_task = obj.get("target_task_id")
                if wanted and target not in wanted and target != "Team Lead / Intake":
                    continue
                if task_id and obj_task not in {task_id, None, ""}:
                    continue
                key = str(obj.get("created_at_utc")) + "|" + target + "|" + str(obj.get("user_message"))
                if key not in seen:
                    out.append(obj)
                    seen.add(key)

        out.sort(key=lambda x: str(x.get("created_at_utc") or ""))
        return out[-max(1, int(limit)):]
    except Exception:
        return []



# -------------------------
# Pause / Resume (quota, transient errors)
# -------------------------

_PAUSE_EVENT = threading.Event()

def _looks_like_quota_or_rate_limit_error(err: Exception) -> bool:
    name = err.__class__.__name__.lower()
    msg = str(err).lower()
    # openai-python commonly raises RateLimitError / APIStatusError
    if "ratelimit" in name or "rate_limit" in msg or "rate limit" in msg:
        return True
    if "insufficient_quota" in msg or ("insufficient" in msg and "quota" in msg):
        return True
    if "quota" in msg and ("exceeded" in msg or "insufficient" in msg):
        return True
    return False


def _pause_run(rec: Recorder, *, reason: str, err: Exception) -> None:
    """Mark the run as paused and wait for RESUME.flag."""
    try:
        _PAUSE_EVENT.set()
        paused_path = rec.out_dir / "PAUSED.json"
        _write_pretty_json(
            paused_path,
            {
                "ts_utc": _utc_now_iso(),
                "reason": reason,
                "error_type": err.__class__.__name__,
                "error": str(err),
            },
        )
        rec.log("run_paused", ts=_utc_now_iso(), reason=reason, error_type=err.__class__.__name__, error=str(err))
    except Exception:
        pass


def _wait_for_resume(rec: Recorder, *, poll_seconds: float = 2.0) -> None:
    """Block until RESUME.flag appears (or PAUSED.json disappears)."""
    resume_flag = rec.out_dir / "RESUME.flag"
    paused_path = rec.out_dir / "PAUSED.json"

    while True:
        if not paused_path.exists():
            break
        if resume_flag.exists():
            try:
                resume_flag.unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                try:
                    resume_flag.unlink()
                except Exception:
                    pass
            try:
                paused_path.unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                try:
                    paused_path.unlink()
                except Exception:
                    pass
            break
        time.sleep(max(0.5, float(poll_seconds)))

    _PAUSE_EVENT.clear()
    try:
        rec.log("run_resumed", ts=_utc_now_iso())
    except Exception:
        pass


def _maybe_wait_if_paused(rec: Recorder) -> None:
    """If the run is already paused (e.g., previous crash), wait for resume."""
    if (rec.out_dir / "PAUSED.json").exists():
        _PAUSE_EVENT.set()
        _wait_for_resume(rec)


# -------------------------
# Resume discovery (auto)
# -------------------------

def _find_latest_incomplete_run(*, outputs_dir: str = "outputs") -> Optional[Path]:
    base = Path(outputs_dir)
    if not base.exists():
        return None
    runs = [p for p in base.iterdir() if p.is_dir()]
    candidates = []
    for d in runs:
        if (d / "DONE.flag").exists():
            continue
        if (d / "initial_input.json").exists() or (d / "PAUSED.json").exists() or (d / "run_state.json").exists():
            try:
                mtime = max([d.stat().st_mtime] + [p.stat().st_mtime for p in d.rglob("*") if p.is_file()])
            except Exception:
                mtime = d.stat().st_mtime
            candidates.append((mtime, d))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _make_recorder_for_run(run_dir: Path, *, logs_dir: str = "logs") -> Recorder:
    run_id = run_dir.name
    log_path = Path(logs_dir) / f"run_{run_id}.jsonl"
    run_dir.mkdir(parents=True, exist_ok=True)
    return Recorder(run_id=run_id, log_path=log_path, out_dir=run_dir)
def _utc_now_iso() -> str:
    # simple ISO-ish string without tz dependency
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _as_str(x: Any) -> str:
    return "" if x is None else str(x)


def _as_bool(value: Any, default: bool = True) -> bool:
    """Coerce LLM/UI truthy strings without making 'false' truthy."""
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


def _normalize_resource_request_item(item: Any) -> Optional[Dict[str, Any]]:
    """Normalize agent resource request items for stable UI rendering.

    Older code defaulted every string request to SVG/PNG, which made internal
    engineering requests look like image uploads. Keep the item generic unless
    the agent explicitly supplied formats.
    """
    if isinstance(item, str):
        name = item.strip()
        if not name:
            return None
        return {"name": name, "kind": "resource", "required": True, "preferred_formats": [], "notes": ""}
    if not isinstance(item, dict):
        return None
    name = str(item.get("name") or item.get("id") or "").strip()
    if not name:
        return None
    formats = item.get("preferred_formats") or item.get("formats") or []
    if not isinstance(formats, list):
        formats = [formats]
    return {
        "name": name,
        "kind": str(item.get("kind") or "resource"),
        "required": _as_bool(item.get("required"), True),
        "preferred_formats": [str(x) for x in formats if str(x).strip()],
        "notes": str(item.get("notes") or ""),
    }

def _extract_resource_requests(out: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generic extraction of resource requests across agents.

    Supported keys:
      - out["resource_requests"] : list[dict|str]
      - out["assets_needed"] : list[str]
      - out["engineer_handoff"]["assets_needed"] : list[str]
    """
    reqs: List[Dict[str, Any]] = []
    if not isinstance(out, dict):
        return reqs

    rr = out.get("resource_requests")
    if isinstance(rr, list):
        for x in rr:
            norm = _normalize_resource_request_item(x)
            if norm:
                reqs.append(norm)

    an = out.get("assets_needed")
    if isinstance(an, list):
        for x in an:
            norm = _normalize_resource_request_item(x)
            if norm:
                reqs.append(norm)

    eh = out.get("engineer_handoff")
    if isinstance(eh, dict):
        an2 = eh.get("assets_needed")
        if isinstance(an2, list):
            for x in an2:
                norm = _normalize_resource_request_item(x)
                if norm:
                    reqs.append(norm)

    # De-dupe by name
    seen = set()
    out2: List[Dict[str, Any]] = []
    for r in reqs:
        if not isinstance(r, dict):
            continue
        nm = str(r.get("name") or "").strip()
        if not nm:
            continue
        k = nm.lower()
        if k in seen:
            continue
        seen.add(k)
        out2.append(r)
    return out2


def _strip_resource_request_fields(out: Dict[str, Any]) -> Dict[str, Any]:
    """Remove request-only fields after Team Lead resolves them as internal/nonblocking.

    Some agents keep repeating a generated/internal resource request even after
    receiving resource_decision. The workflow should not surface those stale
    requests to the user or leak them downstream as if still pending. Preserve
    the substantive artifact fields and record that Team Lead auto-resolved the
    request.
    """
    if not isinstance(out, dict):
        return out
    cleaned = dict(out)
    cleaned.pop("resource_requests", None)
    cleaned.pop("assets_needed", None)
    eh = cleaned.get("engineer_handoff")
    if isinstance(eh, dict):
        eh2 = dict(eh)
        eh2.pop("assets_needed", None)
        cleaned["engineer_handoff"] = eh2
    cleaned["team_lead_resource_auto_resolution"] = {
        "status": "resolved_internal_nonblocking",
        "note": "Internal/generated resource requests were auto-resolved by Team Lead and stripped after retry attempts."
    }
    return cleaned


def _filter_resource_requests(reqs: List[Dict[str, Any]], resource_decision: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter out requests the user has denied/redirected to avoid loops."""
    if not reqs or not isinstance(reqs, list):
        return []
    reqs = [r for r in reqs if isinstance(r, dict) and not _resource_request_is_nonblocking(r)]
    if not reqs:
        return []
    if not resource_decision or not isinstance(resource_decision, dict):
        return reqs
    overrides = resource_decision.get("overrides") if isinstance(resource_decision.get("overrides"), dict) else {}
    deny = overrides.get("do_not_request_again") or []
    if deny is True:
        return []
    if isinstance(deny, str):
        deny = [deny]
    if not isinstance(deny, list):
        deny = []
    deny_set = {str(x).strip().lower() for x in deny if str(x).strip()}
    if not deny_set:
        return reqs
    out = []
    for r in reqs:
        if not isinstance(r, dict):
            continue
        nm = str(r.get("name") or "").strip()
        if nm and nm.lower() in deny_set:
            continue
        out.append(r)
    return out


def _resource_request_is_nonblocking(req: Dict[str, Any]) -> bool:
    """Return True for resources that should not stop the pipeline.

    Human resource blocking should be reserved for genuinely unavailable external
    facts/assets/secrets. Generated fixtures, README examples, QA screenshots,
    benchmark examples, optional design polish, and OCR sample assets must not
    stop the workflow; the model should generate placeholders or defer them.
    """
    if not isinstance(req, dict):
        return True
    text = " ".join([
        str(req.get("name") or ""),
        str(req.get("kind") or ""),
        str(req.get("notes") or ""),
        " ".join([str(x) for x in (req.get("preferred_formats") or [])]),
    ]).lower()
    if not _as_bool(req.get("required"), True):
        return True

    # A generic accessibility checklist/test plan can be generated by UX.
    # A company/client/legal/compliance accessibility checklist is different:
    # that is a user-owned external standard and should still be surfaced.
    accessibility_external_markers = (
        "company accessibility", "client accessibility", "official accessibility",
        "provided accessibility", "existing accessibility", "internal accessibility standard",
        "accessibility standard from", "accessibility policy", "section 508",
        "wcag", "wcag checklist", "wcag accessibility", "wcag policy", "wcag standard",
        "legal accessibility", "compliance accessibility",
        "compliance checklist", "client checklist", "company checklist",
    )
    if any(term in text for term in accessibility_external_markers):
        return False

    nonblocking_terms = (
        "optional", "nice-to-have", "nice to have", "favicon", "placeholder",
        "readme example", "readme examples", "cli example", "cli examples",
        "benchmark example", "benchmark examples",
        "generated fixture", "generated fixtures", "test fixture", "test fixtures",
        "openapi placeholder", "schema example",
        "accessibility checklist", "accessibility test plan", "test plan checklist",
        "accessibility test plan checklist", "standard accessibility",
        "internal checklist", "review checklist", "qa checklist",
    )
    if any(term in text for term in nonblocking_terms):
        return True

    # Team Lead validation: internal engineering context is never a user-owned
    # resource. These requests are auto-resolved so weak agent output cannot turn
    # repository inspection, test creation, or verification into user homework.
    internal_engineering_terms = (
        "repository working tree", "working tree contents", "repo working tree",
        "repo contents", "current repository", "current source tree", "source tree",
        "project tree", "file tree", "working directory contents",
        "makefile content", "makefile contents", "command output", "terminal output",
        "console transcript", "console transcripts", "console screenshot", "console screenshots",
        "ci run log", "ci run logs", "ci logs", "github actions run", "github actions runs",
        "pr url", "pull request url", "push test branch",
        "docker-compose up", "docker compose up", "docker-compose output", "docker compose output",
        "docker output", "docker screenshot", "docker screenshots",
        "terminal screenshot", "terminal screenshots", "log screenshot", "log screenshots",
        "ci screenshot", "ci screenshots",
        "curl output", "curl command output", "curl to /api/health", "curl /api/health", "curl http://localhost",
        "uvicorn", "fastapi testclient", "testclient", "test harness", "local verification command",
        "local verification commands", "runnable environment", "runtime environment", "ability to run",
        "run local verification", "local runtime", "live runtime", "cors proof", "timeout proof",
        "stdout log", "stdout logs", "api runtime evidence", "runtime evidence",
        "backend/dockerfile", "frontend/dockerfile", "dockerfile content",
        "minimal test file", "minimal test files", "backend test files", "frontend test files",
        "test files for backend", "test files for frontend",
        "pre-commit-config", ".pre-commit-config", "package-lock.json",
        ".env.example", "env example", "environment example",
        "frontend/src/", "backend/app/", "src/styles.css", "styles.css",
        "vite config", "tsconfig", "eslint", "pytest output", "npm run",
        "test evidence", "verification artifact", "verification artifacts",
        "browser console", "localhost screenshot", "local runtime screenshot",
        "api health output", "health endpoint output",
    )
    if any(term in text for term in internal_engineering_terms):
        return True

    # If the item is described as something the system can author itself, do not
    # force a human upload. Check this before the user-owned markers so phrases
    # like "generated sample input" or "synthetic dataset fixture" do not become
    # blocking just because they contain "sample input" or "dataset".
    generated_markers = ("generate", "generated", "fixture", "mock", "dummy", "synthetic")
    if any(term in text for term in generated_markers):
        # Real credentials/secrets are never generated placeholders.
        if not any(term in text for term in ("api key", "secret", "credential")):
            return True

    # Preserve legitimate user-owned resource requests. A sample screenshot,
    # dataset, API key, design reference, existing repo, or business rule may be
    # genuinely external even if it contains words like "sample".
    user_owned_markers = (
        "api key", "secret", "credential", "dataset", "csv", "existing repo",
        "uploaded repo", "brand", "logo", "design reference", "reference image",
        "screenshot", "sample screenshot", "sample input", "sample output",
        "business rule", "deployment account", "legal", "compliance",
    )
    if any(term in text for term in user_owned_markers):
        return False

    return False


def _wait_for_external_unblock(run_state: RunState, rec: Recorder, *, poll_seconds: float = 2.0) -> None:
    """Keep the dashboard server alive while a run is blocked."""
    if str(os.getenv("ASCENDANT_HOLD_ON_BLOCK", "1")).strip().lower() in {"0", "false", "no", "off"}:
        return
    rec.log("run_blocked_keep_alive_start", ts=_utc_now_iso(), run_state_path=str(run_state.path))
    while True:
        if (rec.out_dir / "STOP.flag").exists():
            rec.log("run_blocked_keep_alive_stop_flag", ts=_utc_now_iso())
            return
        state = _read_json_best_effort(run_state.path) or run_state.snapshot()
        if state.get("done") or not state.get("blocked"):
            rec.log("run_blocked_keep_alive_end", ts=_utc_now_iso(), done=bool(state.get("done")), blocked=bool(state.get("blocked")))
            return
        time.sleep(max(0.5, float(poll_seconds)))

def _request_items_are_nonblocking(items: Any) -> bool:
    """True when every requested item can be generated/deferred by the system."""
    if not isinstance(items, list) or not items:
        return False
    valid_items = [x for x in items if isinstance(x, dict)]
    if not valid_items:
        return False
    return all(_resource_request_is_nonblocking(x) for x in valid_items)


def _domain_lead_for_request(stage: str, agent_name: str = "") -> str:
    """Map a request origin to the internal lead that reviews before Team Lead escalation."""
    st = str(stage or "").lower()
    ag = str(agent_name or "").lower()
    if st == "ux" or "ux" in ag:
        return "UX Lead"
    if st in {"engineer", "engineering", "qa"} or "engineer" in ag or "qa" in ag:
        return "Engineering Lead"
    if st in {"eng_lead", "orchestrator"} or "engineering lead" in ag:
        return "Engineering Lead"
    if st == "pm" or "pm" in ag:
        return "Team Lead"
    return "Team Lead"


def _build_user_facing_request_packet(
    *,
    stage: str,
    agent_name: str,
    task_id: Optional[str],
    items: List[Dict[str, Any]],
    kind: str = "resource",
) -> Dict[str, Any]:
    """Create Team-Lead-mediated metadata for a user-visible request.

    Raw agent requests should not be displayed directly to the user. This packet
    records the intended escalation chain and a concise user-facing message.
    """
    domain_lead = _domain_lead_for_request(stage, agent_name)
    names = [str(x.get("name")) for x in (items or []) if isinstance(x, dict) and x.get("name")]
    title = names[0] if len(names) == 1 else (", ".join(names[:2]) + ("…" if len(names) > 2 else ""))
    if not title:
        title = "Workflow input needed"
    if kind == "resource":
        message = (
            f"{domain_lead} escalated a resource request to the Team Lead. "
            f"The team may need: {title}. Upload a file only if this is a real external asset you already have. "
            "Otherwise choose the Team Lead recommendation to proceed with internal generation, placeholders, or documented assumptions."
        )
        recommended_action = "use_team_lead_recommendation"
    else:
        message = (
            f"{domain_lead} escalated a workflow decision to the Team Lead. "
            "Review the concise question below and provide a decision or clarification."
        )
        recommended_action = "answer_or_approve"
    return {
        "title": title,
        "message": message,
        "recommended_action": recommended_action,
        "domain_lead_review": {
            "reviewed_by": domain_lead,
            "origin_agent": agent_name,
            "origin_stage": stage,
            "task_id": task_id,
            "decision": "escalate_to_team_lead",
        },
        "team_lead_review": {
            "decision": "ask_user_only_if_external_or_scope_changing",
            "hide_raw_json_by_default": True,
        },
    }


def _expand_expected_files_for_scaffold_placeholders(expected_files: List[str], wi: Dict[str, Any], code_output: Any) -> List[str]:
    """Allow deterministic .gitkeep placeholders for directories required by the task text.

    The Engineering Lead sometimes says an acceptance criterion like "repo contains
    scripts/" but forgets to include scripts/ or scripts/.gitkeep in files_expected.
    Blocking the user for that is wasteful. If the engineer writes a .gitkeep under
    a directory named in the work item text, expand the allowlist automatically.
    """
    out = [str(x).strip().replace('\\', '/') for x in (expected_files or []) if str(x).strip()]
    seen = set(out)
    try:
        wi_text = json.dumps(wi or {}, ensure_ascii=False).lower().replace('\\', '/')
    except Exception:
        wi_text = str(wi or {}).lower().replace('\\', '/')
    files = []
    if isinstance(code_output, dict) and isinstance(code_output.get("files"), list):
        files = code_output.get("files") or []
    for item in files:
        if not isinstance(item, dict):
            continue
        raw_path = str(item.get("path") or "").strip().replace('\\', '/')
        if not raw_path.endswith("/.gitkeep"):
            continue
        parent = raw_path.rsplit("/.gitkeep", 1)[0].strip("/")
        if not parent:
            continue
        parent_marker = parent.lower().rstrip("/") + "/"
        basename_marker = parent.lower().rstrip("/").split("/")[-1] + "/"
        if parent_marker in wi_text or basename_marker in wi_text:
            for candidate in (parent.rstrip("/") + "/", raw_path):
                if candidate not in seen:
                    out.append(candidate)
                    seen.add(candidate)
    return out




def _copy_tree_contents(src: Path, dst: Path, *, overwrite: bool = True) -> Dict[str, Any]:
    """Copy a directory tree into another directory and return an audit report."""
    report: Dict[str, Any] = {
        "source": str(src),
        "destination": str(dst),
        "copied_files": [],
        "copied_dirs": [],
        "errors": [],
    }
    if not src.exists() or not src.is_dir():
        report["errors"].append({"error": "source_missing_or_not_directory"})
        return report
    dst.mkdir(parents=True, exist_ok=True)
    for path in sorted(src.rglob("*")):
        rel = path.relative_to(src)
        target = dst / rel
        try:
            if path.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                report["copied_dirs"].append(str(rel).replace("\\", "/") + "/")
            elif path.is_file():
                if target.exists() and not overwrite:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                report["copied_files"].append(str(rel).replace("\\", "/"))
        except Exception as exc:
            report["errors"].append({"path": str(rel).replace("\\", "/"), "error": str(exc)})
    return report


def _count_materialized_files(path: Path) -> int:
    if not path.exists() or not path.is_dir():
        return 0
    return sum(1 for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts)


def _task_source_candidates(out_dir: Path, task_id: str) -> List[Path]:
    """Return prior same-task attempt/candidate directories, newest first.

    Only directories under the matching task id are considered. This prevents a
    resume repair for T3 from accidentally selecting a large T4/frontend snapshot
    or a scaffold-heavy candidate that happens to contain more total files.
    """
    roots = [out_dir / ".candidates" / task_id, out_dir / ".attempts" / task_id]
    found: List[Tuple[float, int, Path]] = []
    for root in roots:
        if not root.exists():
            continue
        for child in root.iterdir():
            if not (child.is_dir() and (child.name.startswith("attempt_") or child.name.startswith("resume_accept__"))):
                continue
            try:
                mtime = child.stat().st_mtime
            except Exception:
                mtime = 0.0
            count = _count_materialized_files(child)
            if count > 0:
                found.append((mtime, count, child))
    found.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [p for _, _, p in found]


def _expected_scope_match_count(path: Path, expected_files: List[str]) -> int:
    """Count expected file/dir scopes materialized under path."""
    if not path.exists() or not path.is_dir():
        return 0
    count = 0
    for raw in expected_files or []:
        item = str(raw or "").strip().replace("\\", "/")
        if not item:
            continue
        if item.endswith("/"):
            target = path / item.rstrip("/")
            if target.exists() and target.is_dir() and _count_materialized_files(target) > 0:
                count += 1
        elif (path / item).is_file():
            count += 1
    return count


def _select_task_source_for_expected_files(out_dir: Path, task_id: str, expected_files: Optional[List[str]] = None) -> Optional[Path]:
    """Select the best prior source for a task using expected-file satisfaction.

    The old selector chose the directory with the most files. That failed on
    resumed runs: a full candidate could contain many scaffold/frontend files
    while missing the actual current-task files. This selector first requires all
    expected files when possible, then falls back to partial expected-file match,
    and only uses total file count as a final tie-breaker.
    """
    expected = [str(x or "").strip().replace("\\", "/") for x in (expected_files or []) if str(x or "").strip()]
    candidates = _task_source_candidates(out_dir, task_id)
    if not candidates:
        return None

    scored: List[Tuple[int, int, int, int, float, Path]] = []
    for child in candidates:
        expected_count = _expected_scope_match_count(child, expected) if expected else 0
        all_expected = 1 if (expected and expected_count == len(expected)) else 0
        partial_expected = 1 if expected_count > 0 else 0
        file_count = _count_materialized_files(child)
        try:
            mtime = child.stat().st_mtime
        except Exception:
            mtime = 0.0
        scored.append((all_expected, partial_expected, expected_count, file_count, mtime, child))

    # Prefer task correctness over size: all expected files > partial expected
    # files > file count > mtime. This fixes the T3 resume trap where a larger
    # candidate lacked backend/app/main.py, api.py, models.py, etc.
    scored.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]), reverse=True)
    best = scored[0]
    if expected and best[2] == 0:
        # No prior source contains any expected file; using an unrelated-looking
        # folder would poison the next candidate, so return None.
        return None
    return best[5]


def _richest_existing_attempt_or_candidate(out_dir: Path, task_id: str, expected_files: Optional[List[str]] = None) -> Optional[Path]:
    """Backward-compatible wrapper for task-aware resume source selection."""
    return _select_task_source_for_expected_files(out_dir, task_id, expected_files)


def _prepare_candidate_workspace(*, out_dir: Path, task_id: str, attempt_n: int, engineer_id: str, expected_files: Optional[List[str]] = None) -> Path:
    """Create a full candidate workspace snapshot for a task attempt.

    Seed from canonical workspace, then overlay the best prior same-task source
    that satisfies this task's expected files. This keeps dependencies from the
    workspace while recovering unpromoted current-task files from old attempts.
    """
    candidate = out_dir / ".candidates" / task_id / f"attempt_{attempt_n}__{engineer_id}"
    if candidate.exists():
        shutil.rmtree(candidate)
    candidate.mkdir(parents=True, exist_ok=True)

    workspace_source = out_dir / "workspace"
    if workspace_source.exists() and workspace_source.is_dir():
        shutil.copytree(workspace_source, candidate, dirs_exist_ok=True)

    task_source = _select_task_source_for_expected_files(out_dir, task_id, expected_files or [])
    if task_source is not None and task_source.exists() and task_source.is_dir():
        try:
            if task_source.resolve() != candidate.resolve():
                shutil.copytree(task_source, candidate, dirs_exist_ok=True)
        except Exception:
            shutil.copytree(task_source, candidate, dirs_exist_ok=True)
    return candidate


def _expected_scope_has_materialized_files(workspace_dir: Path, expected_files: List[str]) -> bool:
    """Return True when at least one expected file/scope is present in workspace."""
    if not expected_files:
        return workspace_dir.exists() and any(p.is_file() for p in workspace_dir.rglob("*"))
    for raw in expected_files:
        item = str(raw or "").strip().replace("\\", "/")
        if not item:
            continue
        if item.endswith("/"):
            target = workspace_dir / item.rstrip("/")
            if target.exists() and target.is_dir() and _count_materialized_files(target) > 0:
                return True
        else:
            if (workspace_dir / item).is_file():
                return True
    return False


def _expected_scope_is_satisfied(workspace_dir: Path, expected_files: List[str]) -> bool:
    """Return True only when every declared file/dir scope is materialized.

    Resume repair must be stricter than the lightweight evidence check above:
    if T1 expected five files and only README.md was promoted, the old `any`
    check falsely treated the task as repaired and left downstream tasks broken.
    """
    normalized = [str(x or "").strip().replace("\\", "/") for x in (expected_files or []) if str(x or "").strip()]
    if not normalized:
        return workspace_dir.exists() and any(p.is_file() for p in workspace_dir.rglob("*"))
    for item in normalized:
        if item.endswith("/"):
            target = workspace_dir / item.rstrip("/")
            if not (target.exists() and target.is_dir() and _count_materialized_files(target) > 0):
                return False
        else:
            if not (workspace_dir / item).is_file():
                return False
    return True


def _safe_fs_token(value: str) -> str:
    """Return a filesystem-safe token for temporary audit paths."""
    token = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value or ""))
    return token.strip("._") or "task"


def _promote_candidate_workspace(*, candidate_dir: Path, workspace_dir: Path, task_id: str, engineer_id: str) -> Dict[str, Any]:
    """Promote a full candidate snapshot into the canonical workspace atomically-ish.

    Copy into a temp directory first, move the existing workspace to a backup,
    then replace it. If replacement fails after the old workspace moved, attempt
    rollback so a bad promotion does not destroy the user's resumable run.
    """
    report: Dict[str, Any] = {
        "task_id": task_id,
        "engineer_id": engineer_id,
        "source_candidate_dir": str(candidate_dir),
        "workspace_dir": str(workspace_dir),
        "status": "error",
        "files_written": [],
        "files_skipped": [],
        "errors": [],
        "mode": "candidate_snapshot_promotion",
    }
    tmp: Optional[Path] = None
    backup: Optional[Path] = None
    try:
        if not candidate_dir.exists() or not candidate_dir.is_dir():
            report["errors"].append({"error": "candidate_dir_missing"})
            return report
        token = f"{_safe_fs_token(task_id)}_{time.time_ns()}"
        tmp = workspace_dir.parent / f".{workspace_dir.name}.tmp_promote_{token}"
        backup = workspace_dir.parent / f".{workspace_dir.name}.backup_promote_{token}"
        if tmp.exists():
            shutil.rmtree(tmp)
        if backup.exists():
            shutil.rmtree(backup)
        shutil.copytree(candidate_dir, tmp, dirs_exist_ok=True)
        workspace_dir.parent.mkdir(parents=True, exist_ok=True)
        if workspace_dir.exists():
            workspace_dir.replace(backup)
        tmp.replace(workspace_dir)
        if backup.exists():
            shutil.rmtree(backup)
        for path in sorted(workspace_dir.rglob("*")):
            if path.is_file():
                report["files_written"].append({"path": str(path.relative_to(workspace_dir)).replace("\\", "/")})
        report["status"] = "written"
    except Exception as exc:
        report["errors"].append({"error": str(exc)})
        report["status"] = "error"
        try:
            if workspace_dir.exists() and workspace_dir.is_dir():
                shutil.rmtree(workspace_dir)
            if backup is not None and backup.exists():
                backup.replace(workspace_dir)
                report["rollback"] = "restored_previous_workspace"
        except Exception as rollback_exc:
            report["errors"].append({"error": f"rollback_failed:{rollback_exc}"})
    finally:
        for leftover in (tmp, backup):
            try:
                if leftover is not None and leftover.exists():
                    shutil.rmtree(leftover)
            except Exception:
                pass
    return report


def _repair_done_task_workspace_from_attempts(*, rec: Recorder, orch: EngineeringOrchestratorAgent) -> None:
    """Repair older resumed runs whose done tasks were not cumulatively promoted.

    This only copies artifacts for tasks already marked done by the orchestrator.
    It does not mark blocked/todo tasks complete; it merely makes the filesystem
    match the existing done statuses so downstream dependent tasks can run.
    """
    try:
        state = orch.run(agent_input={"command": "get_state"})
    except Exception as exc:
        rec.log("workspace_resume_repair_skipped", ts=_utc_now_iso(), reason=f"get_state_failed:{exc}")
        return
    if not isinstance(state, dict):
        return
    runtime = state.get("runtime") if isinstance(state.get("runtime"), dict) else {}
    work_items = state.get("work_items") if isinstance(state.get("work_items"), dict) else {}
    workspace = rec.out_dir / "workspace"
    repaired: List[Dict[str, Any]] = []
    for task_id, rt in runtime.items():
        if not isinstance(rt, dict) or rt.get("status") != "done":
            continue
        wi = work_items.get(task_id) if isinstance(work_items.get(task_id), dict) else {}
        expected = [str(x) for x in (wi.get("files_expected") or []) if str(x).strip()]
        if _expected_scope_is_satisfied(workspace, expected):
            continue
        source = _richest_existing_attempt_or_candidate(rec.out_dir, str(task_id), expected)
        if source is None:
            repaired.append({"task_id": task_id, "status": "missing_source", "expected_files": expected})
            continue
        report = _copy_tree_contents(source, workspace, overwrite=True)
        report["task_id"] = task_id
        report["expected_files"] = expected
        repaired.append(report)
    if repaired:
        rec.save_json("workspace_resume_repair.json", {"created_at_utc": _utc_now_iso(), "repairs": repaired})
        rec.log("workspace_resume_repair_done", ts=_utc_now_iso(), repaired_count=len(repaired))


def _directive_defers_verification(directive: Optional[Dict[str, Any]]) -> bool:
    """Return True when the user explicitly accepts/defer runtime/manual evidence."""
    if not isinstance(directive, dict) or _directive_blocks_task(directive):
        return False
    text = _directive_intent_text(directive)
    overrides = directive.get("overrides") if isinstance(directive.get("overrides"), dict) else {}
    explicit_override = any(bool(overrides.get(k)) for k in (
        "defer_to_v2_if_needed",
        "defer_runtime_verification",
        "accept_code_only_review",
        "qa_waiver",
        "route_around_blocked_item",
    ))
    evidence_terms = (
        "defer runtime", "defer manual", "defer verification", "runtime verification",
        "manual runtime", "code-only", "code only", "qa waiver", "accept limitation",
        "do not ask", "verification artifacts", "curl", "stdout", "screenshots", "logs",
    )
    decision = str(directive.get("decision") or "").strip().lower()
    decision_allows = decision in {"continue", "accept_limitation", "defer", "route_around", "provided", "mark_unavailable"}
    return bool(decision_allows and (explicit_override or any(term in text for term in evidence_terms)))


def _qa_block_is_evidence_only(qa_out: Dict[str, Any]) -> bool:
    """Detect QA blocks that are only missing runtime/manual evidence.

    This guard must be conservative: a human evidence waiver should not turn a
    real code/API defect into `mark_done` merely because QA also mentioned
    missing curl logs. Prefer structured issue analysis when available.
    """
    evidence_terms = (
        "runtime verification", "verification evidence", "manual verification",
        "curl", "stdout", "screenshot", "browser", "raw outputs", "logs", "testclient",
        "no files restaged", "no deliverables changed/restaged", "e2e", "playwright",
    )
    hard_defect_terms = (
        "scope_error", "path_outside_task_scope", "security", "syntax error",
        "does not compile", "missing required file", "contract mismatch",
        "incorrect", "wrong", "invalid length", "duplicate givens", "api bug",
        "not implemented", "implementation defect", "failing test",
    )

    review = qa_out.get("review") if isinstance(qa_out.get("review"), dict) else {}
    issues = review.get("issues") if isinstance(review, dict) else []
    blocking_issues = [
        issue for issue in issues
        if isinstance(issue, dict) and str(issue.get("severity") or "").lower() in {"blocker", "major"}
    ]
    if blocking_issues:
        saw_evidence_issue = False
        for issue in blocking_issues:
            issue_text = json.dumps(issue, ensure_ascii=False).lower()
            if any(term in issue_text for term in hard_defect_terms):
                return False
            if any(term in issue_text for term in evidence_terms):
                saw_evidence_issue = True
                continue
            return False
        return saw_evidence_issue

    try:
        text = json.dumps(qa_out, ensure_ascii=False).lower()
    except Exception:
        text = str(qa_out).lower()
    return any(term in text for term in evidence_terms) and not any(term in text for term in hard_defect_terms)


def _conditionally_accept_deferred_evidence_block(
    *,
    qa_out: Dict[str, Any],
    directive: Optional[Dict[str, Any]],
    candidate_dir: Path,
    expected_files: List[str],
) -> Tuple[bool, Dict[str, Any]]:
    """Apply a human-approved code-only/deferred-evidence QA waiver deterministically."""
    if not isinstance(qa_out, dict):
        return False, qa_out
    if not _directive_defers_verification(directive):
        return False, qa_out
    q = qa_out.get("queue_update") if isinstance(qa_out.get("queue_update"), dict) else {}
    if str(q.get("command") or "") != "mark_blocked":
        return False, qa_out
    if not _qa_block_is_evidence_only(qa_out):
        return False, qa_out
    if not _expected_scope_is_satisfied(candidate_dir, expected_files):
        return False, qa_out
    patched = dict(qa_out)
    review = dict(patched.get("review") or {}) if isinstance(patched.get("review"), dict) else {}
    review["verdict"] = "agree"
    suggestions = list(review.get("suggestions") or []) if isinstance(review.get("suggestions"), list) else []
    suggestions.append("Runtime/manual verification was explicitly deferred by human directive; create a follow-on integration/e2e QA task before production use.")
    review["suggestions"] = suggestions
    review["issues"] = [
        issue for issue in (review.get("issues") or [])
        if not isinstance(issue, dict) or str(issue.get("severity") or "").lower() not in {"blocker"}
    ]
    patched["review"] = review
    patched["queue_update"] = {"command": "mark_done", "blocked_reason": "done"}
    patched["notes"] = (str(patched.get("notes") or "") + "\nOperation override: human directive accepted code-only review and deferred runtime/manual evidence to a later integration gate.").strip()
    return True, patched


def _remove_pending_human_requests_for_task(resource_dir: str, task_id: str, *, reason_contains: str = "") -> int:
    try:
        path = Path(resource_dir) / "pending_human_requests.json"
        pending = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        if not isinstance(pending, list):
            return 0
        kept = []
        removed = 0
        needle = str(reason_contains or "").lower()
        for req in pending:
            if isinstance(req, dict) and str(req.get("task_id") or "") == str(task_id):
                req_text = json.dumps(req, ensure_ascii=False).lower()
                if not needle or needle in req_text:
                    removed += 1
                    continue
            kept.append(req)
        if removed:
            path.write_text(json.dumps(kept, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return removed
    except Exception:
        return 0




def _infer_human_resolution_overrides(*, decision: Any, user_message: Any, request: Optional[Dict[str, Any]] = None, existing: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Convert a resolved human answer into executable workflow flags.

    Human answers often arrive as free text (for example, "route around and do
    not ask for runtime evidence again"). The runner must not treat that as a
    passive note; it must become structured state that later gates can consume.
    Only resolved user intent is scanned here, never the original request
    options, because prompts usually contain multiple mutually exclusive choices.
    """
    overrides: Dict[str, Any] = {}
    if isinstance(existing, dict):
        overrides.update(existing)
    decision_text = str(decision or "").strip().lower()
    msg = str(user_message or "").strip().lower()
    if decision_text in {"accept_limitation", "defer", "route_around", "mark_unavailable"}:
        overrides.update({
            "route_around_blocked_item": True,
            "defer_to_v2_if_needed": True,
            "route_around_nonessential_blockers": True,
        })
    if any(term in msg for term in (
        "route around", "blocked requested item", "blocked the requested item",
        "do not ask", "don't ask", "defer", "v2", "use mock", "use mocks",
        "mock", "fixture", "fixtures", "placeholder", "fallback", "accept code-only",
        "accept code only", "code-only", "code only", "continue based on code review",
    )):
        overrides.update({
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
        overrides.update({
            "defer_runtime_verification": True,
            "accept_code_only_review": True,
            "qa_waiver": True,
        })
    if any(term in msg for term in ("unavailable", "not available", "cannot provide", "can't provide")):
        overrides.update({
            "unavailable": True,
            "use_placeholder": True,
            "generate_fixtures": True,
        })
    if any(term in msg for term in ("+3", "three additional", "3 additional", "grant 3", "grant +3")):
        overrides["extra_attempts_granted"] = max(int(overrides.get("extra_attempts_granted") or 0), 3)
    elif any(term in msg for term in ("+2", "two additional", "2 additional", "grant 2", "grant +2")):
        overrides["extra_attempts_granted"] = max(int(overrides.get("extra_attempts_granted") or 0), 2)
    elif any(term in msg for term in ("+1", "one additional", "1 additional", "grant 1", "grant +1")):
        overrides["extra_attempts_granted"] = max(int(overrides.get("extra_attempts_granted") or 0), 1)
    if _looks_like_hard_stop(msg) or decision_text in {"block", "block_task", "halt", "stop"}:
        overrides["user_blocked_task"] = True
    elif overrides.get("route_around_blocked_item"):
        overrides["user_blocked_task"] = False
    return overrides

def _auto_unblock_deferred_verification_tasks(*, rec: Recorder, orch: EngineeringOrchestratorAgent, resource_dir: str, run_state: Optional[RunState]) -> None:
    """Unblock tasks when a stored human directive already defers evidence gates."""
    try:
        blocked = orch.run(agent_input={"command": "list_blocked"}).get("blocked", [])
    except Exception:
        return
    if not isinstance(blocked, list):
        return
    changed = []
    for item in blocked:
        if not isinstance(item, dict):
            continue
        task_id = str(item.get("task_id") or "")
        if not task_id:
            continue
        directive = _load_combined_task_directive(resource_dir, task_id)
        if not (_directive_defers_verification(directive) or _directive_allows_route_around_or_acceptance(directive)):
            continue
        try:
            ok = bool(orch.run(agent_input={"command": "unblock_task", "task_id": task_id, "note": "human_deferred_verification_gate"}).get("ok", False))
        except Exception:
            ok = False
        removed = _remove_pending_human_requests_for_task(resource_dir, task_id, reason_contains="verification")
        changed.append({"task_id": task_id, "unblocked": ok, "pending_human_requests_removed": removed})
    if changed:
        rec.save_json("deferred_verification_unblocks.json", {"created_at_utc": _utc_now_iso(), "items": changed})
        rec.log("deferred_verification_tasks_unblocked", ts=_utc_now_iso(), count=len(changed))
        if run_state is not None:
            try:
                bp = run_state.state.get("block_payload") if isinstance(run_state.state, dict) else {}
                bp_task = str((bp or {}).get("task_id") or "") if isinstance(bp, dict) else ""
                if bp_task and any(x.get("task_id") == bp_task for x in changed):
                    run_state.clear_block(event="deferred_verification.human_directive_unblocked")
            except Exception:
                pass

def _task_sort_key(task_id: str) -> Tuple[int, str]:
    digits = "".join(ch for ch in str(task_id or "") if ch.isdigit())
    try:
        return (int(digits), str(task_id))
    except Exception:
        return (10**9, str(task_id))


def _repair_attempted_task_workspace_from_sources(*, rec: Recorder, orch: EngineeringOrchestratorAgent) -> None:
    """Make prior attempted task files visible in workspace before resume workers run.

    This does not mark tasks done. It only rebuilds the canonical workspace from
    task-aware prior sources so Engineer/QA can see the best existing T3/T4 files
    instead of a workspace that only contains T1/T2.
    """
    try:
        state = orch.run(agent_input={"command": "get_state"})
    except Exception as exc:
        rec.log("attempted_workspace_repair_skipped", ts=_utc_now_iso(), reason=f"get_state_failed:{exc}")
        return
    if not isinstance(state, dict):
        return
    runtime = state.get("runtime") if isinstance(state.get("runtime"), dict) else {}
    work_items = state.get("work_items") if isinstance(state.get("work_items"), dict) else {}
    workspace = rec.out_dir / "workspace"
    repaired: List[Dict[str, Any]] = []
    for task_id in sorted(work_items.keys(), key=_task_sort_key):
        rt = runtime.get(task_id) if isinstance(runtime.get(task_id), dict) else {}
        # Only recover tasks that have already been attempted or are currently
        # blocked/claimed; do not synthesize untouched future tasks.
        attempts_for_task = len(_task_source_candidates(rec.out_dir, str(task_id)))
        if attempts_for_task <= 0 and str(rt.get("status") or "") not in {"blocked", "claimed"}:
            continue
        wi = work_items.get(task_id) if isinstance(work_items.get(task_id), dict) else {}
        expected = [str(x) for x in (wi.get("files_expected") or []) if str(x).strip()]
        if not expected or _expected_scope_is_satisfied(workspace, expected):
            continue
        source = _select_task_source_for_expected_files(rec.out_dir, str(task_id), expected)
        if source is None or not _expected_scope_is_satisfied(source, expected):
            repaired.append({"task_id": task_id, "status": "no_expected_file_source", "expected_files": expected})
            continue
        report = _copy_tree_contents(source, workspace, overwrite=True)
        report["task_id"] = task_id
        report["expected_files"] = expected
        report["source"] = str(source)
        repaired.append(report)
    if repaired:
        rec.save_json("attempted_workspace_repair.json", {"created_at_utc": _utc_now_iso(), "repairs": repaired})
        rec.log("attempted_workspace_repair_done", ts=_utc_now_iso(), repaired_count=len(repaired))


def _directive_allows_route_around_or_acceptance(directive: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(directive, dict) or _directive_blocks_task(directive):
        return False
    text = _directive_intent_text(directive)
    decision = str(directive.get("decision") or "").strip().lower()
    acceptance_terms = (
        "route around", "conditionally accepting", "conditionally accept", "accept t3", "accept t4",
        "accept code-only", "accept code only", "defer runtime", "defer manual",
        "move forward", "continue based on code review", "do not authorize more",
        "use fallback", "mark unavailable", "runtime evidence", "manual evidence",
    )
    if decision in {"accept_limitation", "defer", "route_around", "provided", "mark_unavailable"}:
        return True
    if decision == "continue" and any(term in text for term in acceptance_terms):
        return True
    return any(term in text for term in acceptance_terms)


def _auto_accept_deferred_candidate_tasks(*, rec: Recorder, orch: EngineeringOrchestratorAgent, resource_dir: str, run_state: Optional[RunState], task_ids: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
    """Deterministically mark deferred-evidence tasks done when files are complete.

    This is the loop breaker for resumed T3/T4 runs: if the user already directed
    the system to route around missing runtime/manual evidence, and a prior
    same-task source contains every expected file, promote a full candidate and
    submit a done result instead of asking for another attempt or resource.
    """
    try:
        state = orch.run(agent_input={"command": "get_state"})
    except Exception as exc:
        rec.log("auto_accept_deferred_candidates_skipped", ts=_utc_now_iso(), reason=f"get_state_failed:{exc}")
        return []
    if not isinstance(state, dict):
        return []
    runtime = state.get("runtime") if isinstance(state.get("runtime"), dict) else {}
    work_items = state.get("work_items") if isinstance(state.get("work_items"), dict) else {}
    accepted: List[Dict[str, Any]] = []
    workspace = rec.out_dir / "workspace"
    target_ids = {str(x) for x in (task_ids or []) if str(x).strip()}
    for task_id in sorted(work_items.keys(), key=_task_sort_key):
        if target_ids and str(task_id) not in target_ids:
            continue
        rt = runtime.get(task_id) if isinstance(runtime.get(task_id), dict) else {}
        if str(rt.get("status") or "") == "done":
            continue
        directive = _load_combined_task_directive(resource_dir, str(task_id))
        if not (_directive_defers_verification(directive) or _directive_allows_route_around_or_acceptance(directive)):
            continue
        wi = work_items.get(task_id) if isinstance(work_items.get(task_id), dict) else {}
        expected = [str(x) for x in (wi.get("files_expected") or []) if str(x).strip()]
        if not expected:
            continue
        source = _select_task_source_for_expected_files(rec.out_dir, str(task_id), expected)
        if source is None or not _expected_scope_is_satisfied(source, expected):
            accepted.append({"task_id": task_id, "status": "not_accepted_missing_expected_source", "expected_files": expected})
            continue
        candidate = rec.out_dir / ".candidates" / str(task_id) / f"resume_accept__{int(time.time())}"
        if candidate.exists():
            shutil.rmtree(candidate)
        candidate.mkdir(parents=True, exist_ok=True)
        if workspace.exists() and workspace.is_dir():
            shutil.copytree(workspace, candidate, dirs_exist_ok=True)
        shutil.copytree(source, candidate, dirs_exist_ok=True)
        if not _expected_scope_is_satisfied(candidate, expected):
            accepted.append({"task_id": task_id, "status": "not_accepted_candidate_incomplete", "expected_files": expected, "source": str(source)})
            continue
        promo = _promote_candidate_workspace(candidate_dir=candidate, workspace_dir=workspace, task_id=str(task_id), engineer_id="resume_repair")
        if promo.get("status") != "written":
            accepted.append({"task_id": task_id, "status": "promotion_failed", "promotion": promo, "source": str(source)})
            continue
        result = {
            "task_id": str(task_id),
            "engineer_id": "resume_repair",
            "status": "done",
            "changes": {"resume_repair_promoted_prior_source": str(source)},
            "verification_run": [
                "accepted_by_resume_repair: prior same-task source contains all expected files",
                "runtime/manual evidence deferred by human directive to later integration/test-executor task",
            ],
            "notes": "Task marked done by deterministic resume repair after human directive deferred evidence gates and expected files were present.",
            "handoff_interfaces": [],
            "local_write_report": promo,
        }
        cmd = {"command": "submit_result", "result": result, "mark_done": True, "mark_blocked": False, "blocked_reason": ""}
        rec.save_json(f"orch_submissions/{task_id}.json", cmd)
        try:
            # Orchestrator only accepts submit_result for todo/claimed tasks.
            # Resume-loop tasks are often blocked, so release them before the
            # deterministic acceptance submission.
            try:
                orch.run(agent_input={"command": "unblock_task", "task_id": str(task_id), "note": "resume_repair_accepting_deferred_candidate"})
            except Exception:
                pass
            orch.run(agent_input=cmd)
        except Exception as exc:
            accepted.append({"task_id": task_id, "status": "orch_submit_failed", "error": str(exc), "source": str(source)})
            continue
        _remove_pending_human_requests_for_task(resource_dir, str(task_id))
        try:
            res_pending = Path(resource_dir) / "pending_requests.json"
            if res_pending.exists():
                pending = json.loads(res_pending.read_text(encoding="utf-8"))
                if isinstance(pending, list):
                    kept = [r for r in pending if not (isinstance(r, dict) and str(r.get("task_id") or "") == str(task_id))]
                    res_pending.write_text(json.dumps(kept, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception:
            pass
        accepted.append({"task_id": task_id, "status": "accepted", "source": str(source), "promotion_file_count": len(promo.get("files_written", []))})
    accepted_real = [x for x in accepted if x.get("status") == "accepted"]
    if accepted:
        rec.save_json("resume_deferred_candidate_acceptance.json", {"created_at_utc": _utc_now_iso(), "items": accepted})
    if accepted_real:
        rec.log("resume_deferred_candidates_accepted", ts=_utc_now_iso(), count=len(accepted_real), task_ids=[x.get("task_id") for x in accepted_real])
        if run_state is not None:
            try:
                bp = run_state.state.get("block_payload") if isinstance(run_state.state, dict) else {}
                bp_task = str((bp or {}).get("task_id") or "") if isinstance(bp, dict) else ""
                if bp_task and any(x.get("task_id") == bp_task for x in accepted_real):
                    run_state.clear_block(event="resume_repair.deferred_candidate_accepted")
            except Exception:
                pass

    return accepted


def _run_human_resolution_loop_breakers(*, rec: Recorder, orch: EngineeringOrchestratorAgent, resource_dir: str, task_id: str, directive: Dict[str, Any], run_state: Optional[RunState]) -> bool:
    """Apply deterministic post-human-decision repairs before a worker reclaims.

    This prevents the loop: human resolves max-attempt/QA evidence request -> task
    is unblocked -> worker immediately exceeds the same attempt limit -> asks the
    same question again. If the directive allows route-around/deferred evidence
    and a complete same-task candidate already exists, promote it and mark done.
    """
    if not isinstance(directive, dict):
        return False
    if not (_directive_defers_verification(directive) or _directive_allows_route_around_or_acceptance(directive)):
        return False
    accepted = _auto_accept_deferred_candidate_tasks(
        rec=rec,
        orch=orch,
        resource_dir=resource_dir,
        run_state=run_state,
        task_ids=[str(task_id)],
    )
    try:
        state = orch.run(agent_input={"command": "get_state"})
        runtime = state.get("runtime") if isinstance(state, dict) and isinstance(state.get("runtime"), dict) else {}
        status = str((runtime.get(str(task_id)) or {}).get("status") or "") if isinstance(runtime.get(str(task_id)), dict) else ""
    except Exception:
        status = ""
    done = status == "done" or any(isinstance(x, dict) and str(x.get("task_id")) == str(task_id) and x.get("status") == "accepted" for x in accepted)
    if done:
        _remove_pending_human_requests_for_task(resource_dir, str(task_id))
        rec.log("human_resolution_loop_breaker_accepted_candidate", ts=_utc_now_iso(), task_id=str(task_id), accepted_items=accepted)
    return done

def _auto_resolve_nonblocking_resource_requests(res_eval: ResourceEval, run_state: RunState, rec: Recorder) -> int:
    """Resolve stale/generated/optional resource requests so they do not block the run.

    This protects resume flows from older runs where UX/engineering asked for
    generated fixtures, README examples, OCR QA images, favicons, sample puzzles,
    or other resources that should be created by the system or deferred.
    """
    resolved = 0
    try:
        pending = res_eval.list_pending()
    except Exception as e:
        rec.log("resource_auto_resolve_scan_failed", ts=_utc_now_iso(), error=str(e))
        pending = []

    for req in list(pending or []):
        if not isinstance(req, dict):
            continue
        req_id = str(req.get("request_id") or "")
        items = req.get("items")
        if not req_id or not _request_items_are_nonblocking(items):
            continue
        names = [str(it.get("name")) for it in items if isinstance(it, dict) and it.get("name")]
        try:
            res_eval.resolve(
                request_id=req_id,
                provided_assets=[],
                decision="accept_limitation",
                user_message=(
                    "Auto-resolved as nonblocking: use placeholders, generated fixtures, "
                    "generated README/CLI/benchmark examples, or defer optional/OCR assets."
                ),
                overrides={
                    "auto_resolved_nonblocking": True,
                    "use_placeholder": True,
                    "generate_fixtures": True,
                    "do_not_request_again": names,
                },
            )
            resolved += 1
            rec.log("resource_request_auto_resolved_nonblocking", ts=_utc_now_iso(), request_id=req_id, items=names)
        except Exception as e:
            rec.log("resource_request_auto_resolve_failed", ts=_utc_now_iso(), request_id=req_id, error=str(e))

    # If the persisted run state is blocked on one of these nonblocking requests,
    # clear the block as well so the dashboard and resume logic agree.
    try:
        state = _read_json_best_effort(run_state.path) or run_state.snapshot()
        payload = state.get("block_payload") if isinstance(state, dict) and isinstance(state.get("block_payload"), dict) else {}
        blocked_items = payload.get("items") if isinstance(payload, dict) else None
        blocked_req_id = str(payload.get("request_id") or "") if isinstance(payload, dict) else ""
        if state.get("blocked") and state.get("block_type") == "resource" and _request_items_are_nonblocking(blocked_items):
            names = [str(it.get("name")) for it in blocked_items if isinstance(it, dict) and it.get("name")]
            if blocked_req_id:
                try:
                    res_eval.resolve(
                        request_id=blocked_req_id,
                        provided_assets=[],
                        decision="accept_limitation",
                        user_message="Auto-cleared nonblocking resource block.",
                        overrides={
                            "auto_resolved_nonblocking": True,
                            "use_placeholder": True,
                            "generate_fixtures": True,
                            "do_not_request_again": names,
                        },
                    )
                except Exception:
                    pass
            run_state.clear_block(event="resource.auto_resolved_nonblocking")
            rec.log("run_resource_block_auto_cleared_nonblocking", ts=_utc_now_iso(), request_id=blocked_req_id, items=names)
    except Exception as e:
        rec.log("run_resource_block_auto_clear_failed", ts=_utc_now_iso(), error=str(e))

    return resolved


def _normalize_verification_run(qa_out: Any) -> List[str]:
    """Orchestrator requires verification_run: List[str]. Accept multiple QA shapes."""
    if isinstance(qa_out, dict):
        vr = qa_out.get("verification_run")
        if isinstance(vr, list) and all(isinstance(x, str) for x in vr):
            return vr
        if isinstance(vr, str) and vr.strip():
            return [vr.strip()]
        passed = qa_out.get("passed")
        blocked = qa_out.get("blocked")
        reason = qa_out.get("blocked_reason") or qa_out.get("reason") or ""
        return [f"qa_passed={bool(passed)} blocked={bool(blocked)} reason={str(reason)}"]
    return [f"qa_out_type={type(qa_out).__name__}"]


def _run_stage_with_resources(
    *,
    agent: Any,
    agent_name: str,
    stage: str,
    agent_input: Dict[str, Any],
    res_eval: ResourceEval,
    rec: Recorder,
    shared_context: Dict[str, Any],
    previous_response_id: Optional[str] = None,
    max_loops: int = 3,
    run_state: Optional[RunState] = None,
) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Run a stage agent; if it requests user resources, create a request and wait in the same UI,
    then re-run with updated asset_manifest injected.
    """
    prev = previous_response_id
    base_inp = dict(agent_input)

    loop_count = max(1, int(max_loops))
    for loop_idx in range(loop_count):
        out, prev = agent.run(agent_input=base_inp, previous_response_id=prev)
        if not isinstance(out, dict):
            out = {"output": out}

        raw_reqs = _extract_resource_requests(out)
        reqs = _filter_resource_requests(raw_reqs, shared_context.get("resource_decision"))
        if raw_reqs and not reqs:
            # Team Lead validation rejected/auto-resolved the request as internal,
            # optional, or system-generatable. Rerun the stage once with an
            # explicit resource_decision so the agent can produce its real
            # artifact instead of surfacing a fake user-facing resource request.
            names = [str(r.get("name")) for r in raw_reqs if isinstance(r, dict) and r.get("name")]
            shared_context["resource_decision"] = {
                "request_id": "auto_nonblocking",
                "decision": "accept_limitation",
                "user_message": "Team Lead auto-resolved nonblocking/internal resource request; generate or inspect this internally and do not ask the user for it.",
                "overrides": {
                    "auto_resolved_nonblocking": True,
                    "use_placeholder": True,
                    "generate_fixtures": True,
                    "do_not_request_again": names,
                },
            }
            rec.log("resource_request_auto_resolved_stage_nonblocking", ts=_utc_now_iso(), stage=stage, agent=agent_name, items=names)
            if loop_idx >= loop_count - 1:
                rec.log("resource_request_auto_resolved_stage_exhausted", ts=_utc_now_iso(), stage=stage, agent=agent_name, items=names)
                return _strip_resource_request_fields(out), prev
            base_inp = dict(agent_input)
            base_inp["asset_manifest"] = shared_context.get("asset_manifest")
            base_inp["resource_decision"] = shared_context.get("resource_decision")
            continue
        if not reqs:
            return out, prev

        user_facing = _build_user_facing_request_packet(stage=stage, agent_name=agent_name, task_id=None, items=reqs, kind="resource")
        req_context = {
            "escalation_chain": [agent_name, user_facing["domain_lead_review"]["reviewed_by"], "Team Lead"],
            "domain_lead_review": user_facing.get("domain_lead_review"),
            "team_lead_review": user_facing.get("team_lead_review"),
            "raw_request_hidden_from_user_by_default": True,
        }
        req_id = res_eval.submit(agent=agent_name, stage=stage, items=reqs, context=req_context, user_facing=user_facing)
        rec.log("resource_requested", ts=_utc_now_iso(), stage=stage, agent=agent_name, request_id=req_id, count=len(reqs), domain_lead=user_facing["domain_lead_review"]["reviewed_by"])
        if run_state is not None:
            node_map = {"pm": NODE_PM, "ux": NODE_UX, "eng_lead": NODE_ENG_LEAD}
            run_state.mark_blocked(
                node=node_map.get(stage),
                block_type="resource",
                reason="needs_user_resources",
                payload={"stage": stage, "agent": agent_name, "request_id": req_id, "items": reqs, "context": req_context, "user_facing": user_facing},
            )

        res_eval.wait_until_resolved(req_id, poll_seconds=1.0)
        if run_state is not None:
            run_state.clear_block(event=f"{stage}.resource_resolved")
        shared_context["asset_manifest"] = res_eval.read_manifest()

        # Also inject the user's decision/correction (deny/redirect/provided)
        res_record = res_eval.read_resolution(req_id)
        if isinstance(res_record, dict):
            shared_context["resource_decision"] = {
                "request_id": req_id,
                "decision": res_record.get("decision", "provided"),
                "user_message": res_record.get("user_message", ""),
                "overrides": res_record.get("overrides") or {},
            }
            if _resource_resolution_blocks_task(res_record):
                if run_state is not None:
                    run_state.mark_blocked(
                        node={"pm": NODE_PM, "ux": NODE_UX, "eng_lead": NODE_ENG_LEAD}.get(stage, None),
                        block_type="resource",
                        reason="user_blocked_stage_resource",
                        payload={
                            "stage": stage,
                            "agent": agent_name,
                            "request_id": req_id,
                            "decision": res_record.get("decision"),
                            "user_message": res_record.get("user_message"),
                        },
                    )
                rec.log("stage_resource_block_preserved", ts=_utc_now_iso(), stage=stage, agent=agent_name, request_id=req_id, decision=res_record.get("decision"))
                raise RuntimeError(f"User blocked {stage} resource request {req_id}")
        else:
            shared_context.pop("resource_decision", None)

        base_inp = dict(agent_input)
        base_inp["asset_manifest"] = shared_context.get("asset_manifest")
        if shared_context.get("resource_decision"):
            base_inp["resource_decision"] = shared_context.get("resource_decision")

    return out, prev




def _clear_run_block_if_request_matches(
    run_state: Optional[RunState],
    *,
    request_id: str,
    allowed_block_types: Optional[List[str]] = None,
    event: str = "request.task_unblocked",
) -> bool:
    """Clear run_state.blocked only when the current block belongs to request_id.

    Background monitors tail resolved human/resource logs. A stale resolution can
    arrive after the run has already moved on to a different block. Clearing the
    run-level block unconditionally creates UI/state split-brain: the new
    request disappears from the dashboard even though the corresponding task is
    still waiting. This guard keeps request resolution scoped to the exact
    block_payload.request_id it resolves.
    """
    if run_state is None:
        return False
    try:
        state = _read_json_best_effort(run_state.path) or run_state.snapshot()
        if not isinstance(state, dict) or not state.get("blocked"):
            return False
        payload = state.get("block_payload") if isinstance(state.get("block_payload"), dict) else {}
        current_request_id = str(payload.get("request_id") or "").strip()
        if current_request_id != str(request_id or "").strip():
            return False
        if allowed_block_types:
            block_type = str(state.get("block_type") or "")
            if block_type not in set(allowed_block_types):
                return False
        run_state.clear_block(event=event)
        return True
    except Exception:
        return False


def _resource_resolution_blocks_task(record: Dict[str, Any]) -> bool:
    """True when a resolved resource request means the user intentionally blocked work.

    Resource decisions travel through the same resolved_requests.jsonl channel as
    ordinary proceed/provided decisions. Without this guard, the background
    resource monitor can accidentally release a task after the user selected
    "Block task" in the Requests & Resources UI.
    """
    if not isinstance(record, dict):
        return False
    decision = str(record.get("decision") or "").strip().lower()
    msg = str(record.get("user_message") or record.get("note") or "").strip().lower()
    overrides = record.get("overrides") if isinstance(record.get("overrides"), dict) else {}
    if overrides.get("user_blocked_task") is False or overrides.get("route_around_blocked_item"):
        return False
    if decision in {"accept_limitation", "mark_unavailable", "continue", "provided", "route_around", "defer"}:
        return False
    if any(neg in msg for neg in ("do not block", "don't block", "not block", "route around", "blocked the requested item", "requested item/resource only", "reduce scope", "defer")):
        return False
    if bool(overrides.get("user_blocked_task")):
        return True
    if decision in {"block", "block_task", "block task", "halt", "stop"}:
        return True
    return msg.startswith("block whole") or msg.startswith("stop whole") or "halt work" in msg or "stop work" in msg or "hard stop" in msg


def _resource_monitor_loop(
    *,
    orch: EngineeringOrchestratorAgent,
    res_eval: ResourceEval,
    rec: Recorder,
    shared_context: Dict[str, Any],
    poll_seconds: float = 1.0,
    run_state: Optional[RunState] = None,
) -> None:
    """
    Watch resolved_requests.jsonl and unblock engineer tasks waiting on user resources.
    Also refresh shared_context['asset_manifest'] so subsequent agent steps see assets.
    """
    log_path = Path(res_eval.resolved_log)
    try:
        seen = len(log_path.read_text(encoding="utf-8").splitlines()) if log_path.exists() else 0
    except Exception:
        seen = 0

    while True:
        try:
            lines = log_path.read_text(encoding="utf-8").splitlines() if log_path.exists() else []
            if len(lines) > seen:
                new = lines[seen:]
                seen = len(lines)

                shared_context["asset_manifest"] = res_eval.read_manifest()

                for ln in new:
                    try:
                        obj = json.loads(ln)
                    except Exception:
                        continue
                    req = obj.get("request") if isinstance(obj, dict) else None
                    if not isinstance(req, dict):
                        continue
                    if req.get("stage") == "engineer" and req.get("task_id"):
                        # Write task directive so engineer/qa can rerun with the user's correction.
                        # ResourceEval.resolve already writes a rich directive; merge instead
                        # of overwriting it so provided_assets/request metadata survive the
                        # monitor pass.
                        task_id = _canonical_task_id_for_directive(str(req["task_id"]), {"request_context": req.get("context") if isinstance(req.get("context"), dict) else {}, "task_id": req.get("task_id")})
                        td_dir = Path(res_eval.base_dir) / "task_directives"
                        td_dir.mkdir(parents=True, exist_ok=True)
                        directive_path = td_dir / f"{task_id}.json"
                        existing = _read_json_best_effort(directive_path) or {}
                        directive = dict(existing) if isinstance(existing, dict) else {}
                        existing_overrides = directive.get("overrides") if isinstance(directive.get("overrides"), dict) else {}
                        new_overrides = obj.get("overrides") if isinstance(obj.get("overrides"), dict) else {}
                        directive.update({
                            "request_id": str(req.get("request_id") or obj.get("request_id") or ""),
                            "decision": str(obj.get("decision") or "provided"),
                            "user_message": str(obj.get("user_message") or ""),
                            "overrides": {**existing_overrides, **new_overrides},
                            "provided_assets": (obj.get("provided_assets") if isinstance(obj.get("provided_assets"), list) and obj.get("provided_assets") else directive.get("provided_assets", [])),
                            "request_stage": req.get("stage") or directive.get("request_stage"),
                            "request_agent": req.get("agent") or directive.get("request_agent"),
                            "request_items": (req.get("items") if isinstance(req.get("items"), list) and req.get("items") else directive.get("request_items", [])),
                            "request_context": (req.get("context") if isinstance(req.get("context"), dict) and req.get("context") else directive.get("request_context", {})),
                            "request_user_facing": (req.get("user_facing") if isinstance(req.get("user_facing"), dict) and req.get("user_facing") else directive.get("request_user_facing", {})),
                            "resolved_at_utc": req.get("resolved_at_utc") or directive.get("resolved_at_utc"),
                            "source": "resource_monitor_loop",
                        })
                        directive_path.write_text(json.dumps(directive, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

                        if _resource_resolution_blocks_task(obj):
                            rec.log("resource_task_block_preserved", ts=_utc_now_iso(), task_id=task_id, request_id=req.get("request_id"), decision=obj.get("decision"))
                            if run_state is not None:
                                run_state.mark_blocked(
                                    node=NODE_ENGINEERING,
                                    block_type="resource",
                                    reason="user_blocked_task",
                                    payload={
                                        "task_id": task_id,
                                        "request_id": req.get("request_id"),
                                        "decision": obj.get("decision"),
                                        "user_message": obj.get("user_message"),
                                        "agent": req.get("agent"),
                                        "stage": req.get("stage"),
                                    },
                                )
                            continue

                        accepted_by_loop_breaker = _run_human_resolution_loop_breakers(
                            rec=rec,
                            orch=orch,
                            resource_dir=str(res_eval.base_dir),
                            task_id=str(task_id),
                            directive=directive,
                            run_state=run_state,
                        )
                        if accepted_by_loop_breaker:
                            rec.log("resource_resolution_loop_breaker_accepted_candidate", ts=_utc_now_iso(), task_id=str(task_id), request_id=req.get("request_id"))
                            _clear_run_block_if_request_matches(
                                run_state,
                                request_id=str(req.get("request_id") or obj.get("request_id") or ""),
                                allowed_block_types=["resource"],
                                event="resource.loop_breaker_accepted",
                            )
                            continue
                        orch.run(agent_input={"command": "unblock_task", "task_id": task_id, "note": "resources_decided"})
                        rec.log("task_unblocked", ts=_utc_now_iso(), task_id=task_id)
                        _clear_run_block_if_request_matches(
                            run_state,
                            request_id=str(req.get("request_id") or obj.get("request_id") or ""),
                            allowed_block_types=["resource"],
                            event="resource.task_unblocked",
                        )
        except Exception:
            pass

        time.sleep(max(0.5, float(poll_seconds)))




def _human_monitor_loop(
    *,
    orch: EngineeringOrchestratorAgent,
    human_desk: HumanRequestDesk,
    rec: Recorder,
    poll_seconds: float = 1.0,
    run_state: Optional[RunState] = None,
) -> None:
    """Watch resolved human requests and unblock affected engineer tasks."""
    log_path = Path(human_desk.resolved_log)
    try:
        seen = len(log_path.read_text(encoding="utf-8").splitlines()) if log_path.exists() else 0
    except Exception:
        seen = 0
    while True:
        try:
            lines = log_path.read_text(encoding="utf-8").splitlines() if log_path.exists() else []
            if len(lines) > seen:
                new = lines[seen:]
                seen = len(lines)
                for ln in new:
                    try:
                        obj = json.loads(ln)
                    except Exception:
                        continue
                    req = obj.get("request") if isinstance(obj, dict) else None
                    if not isinstance(req, dict):
                        continue
                    task_id = req.get("task_id")
                    if task_id:
                        task_id = _canonical_task_id_for_directive(str(task_id), {"request_context": req.get("context") if isinstance(req.get("context"), dict) else {}, "task_id": task_id})
                        # Persist the human decision for the next engineer retry.
                        # HumanRequestDesk.resolve already writes this for current versions,
                        # but doing it again here makes resume/old-resolution cases safe.
                        directive: Dict[str, Any] = {"decision": obj.get("decision"), "user_message": obj.get("user_message")}
                        try:
                            hd_dir = Path(human_desk.base_dir) / "human_task_directives"
                            hd_dir.mkdir(parents=True, exist_ok=True)
                            directive_path = hd_dir / f"{task_id}.json"
                            existing = _read_json_best_effort(directive_path) or {}
                            directive = dict(existing) if isinstance(existing, dict) else {}
                            existing_overrides = directive.get("overrides") if isinstance(directive.get("overrides"), dict) else {}
                            new_overrides = obj.get("overrides") if isinstance(obj.get("overrides"), dict) else {}
                            inferred_overrides = _infer_human_resolution_overrides(
                                decision=obj.get("decision"),
                                user_message=obj.get("user_message"),
                                request=req,
                                existing={**existing_overrides, **new_overrides},
                            )
                            existing_context = directive.get("request_context") if isinstance(directive.get("request_context"), dict) else {}
                            new_context = req.get("context") if isinstance(req.get("context"), dict) else {}
                            directive.update({
                                "request_id": str(req.get("request_id") or obj.get("request_id") or ""),
                                "decision": str(obj.get("decision") or "continue"),
                                "user_message": str(obj.get("user_message") or ""),
                                "overrides": inferred_overrides,
                                "request_stage": req.get("stage") or directive.get("request_stage"),
                                "request_agent": req.get("agent") or directive.get("request_agent"),
                                "request_reason": req.get("reason") or directive.get("request_reason"),
                                "request_options": (req.get("options") if isinstance(req.get("options"), list) and req.get("options") else directive.get("request_options", [])),
                                "request_context": {**existing_context, **new_context},
                                "resolved_at_utc": req.get("resolved_at_utc") or directive.get("resolved_at_utc"),
                                "source": "human_monitor_loop",
                            })
                            directive_path.write_text(json.dumps(directive, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                        except Exception as exc:
                            rec.log("human_directive_write_failed", ts=_utc_now_iso(), task_id=str(task_id), error=str(exc))
                        if _directive_blocks_task(directive if isinstance(directive, dict) else {"decision": obj.get("decision"), "user_message": obj.get("user_message")}):
                            rec.log("human_task_block_preserved", ts=_utc_now_iso(), task_id=str(task_id), request_id=req.get("request_id"), decision=obj.get("decision"))
                            if run_state is not None:
                                run_state.mark_blocked(
                                    node=NODE_ENGINEERING,
                                    block_type="human_decision",
                                    reason="user_blocked_task",
                                    payload={"task_id": str(task_id), "request_id": req.get("request_id"), "decision": obj.get("decision"), "user_message": obj.get("user_message")},
                                )
                            continue
                        accepted_by_loop_breaker = _run_human_resolution_loop_breakers(
                            rec=rec,
                            orch=orch,
                            resource_dir=str(human_desk.base_dir),
                            task_id=str(task_id),
                            directive=directive,
                            run_state=run_state,
                        )
                        if accepted_by_loop_breaker:
                            rec.log("human_task_resolved_by_loop_breaker", ts=_utc_now_iso(), task_id=str(task_id), request_id=req.get("request_id"))
                            _clear_run_block_if_request_matches(
                                run_state,
                                request_id=str(req.get("request_id") or obj.get("request_id") or ""),
                                allowed_block_types=["human_input", "human_decision", "decision"],
                                event="human_request.loop_breaker_accepted",
                            )
                            continue
                        orch.run(agent_input={"command": "unblock_task", "task_id": str(task_id), "note": "human_input_resolved"})
                        rec.log("human_task_unblocked", ts=_utc_now_iso(), task_id=str(task_id), request_id=req.get("request_id"))
                        _clear_run_block_if_request_matches(
                            run_state,
                            request_id=str(req.get("request_id") or obj.get("request_id") or ""),
                            allowed_block_types=["human_input", "human_decision", "decision"],
                            event="human_request.task_unblocked",
                        )
        except Exception:
            pass
        time.sleep(max(0.5, float(poll_seconds)))


def _submit_human_request(
    *,
    human_desk: HumanRequestDesk,
    rec: Recorder,
    task_id: str,
    stage: str,
    reason: str,
    question: str,
    options: Optional[List[str]] = None,
    context: Optional[Dict[str, Any]] = None,
    run_state: Optional[RunState] = None,
) -> str:
    req_id = human_desk.submit(
        stage=stage,
        agent="Team Lead",
        task_id=task_id,
        reason=reason,
        question=question,
        options=options or [],
        context=context or {},
    )
    rec.log("human_input_requested", ts=_utc_now_iso(), task_id=task_id, request_id=req_id, reason=reason)
    if run_state is not None:
        run_state.mark_blocked(
            node=NODE_ENGINEERING,
            block_type="human_input",
            reason=reason,
            payload={
                "task_id": task_id,
                "request_id": req_id,
                "stage": stage,
                "agent": "Team Lead",
                "question": question,
                "options": options or [],
                "context": context or {},
            },
        )
    return req_id

# -------------------------
# Engineer worker loop
# -------------------------



def _safe_component(value: str, max_len: int = 80) -> str:
    value = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in str(value))
    value = value.strip("._-")
    return (value or "item")[:max_len]


def _coordinator_workflow_decision(
    *,
    rec: Recorder,
    abnormal_state: str,
    context_pack: Dict[str, Any],
    allowed_actions: List[str],
    default_action: str = "block",
) -> Dict[str, Any]:
    """
    Call CoordinatorAgent in workflow-decision mode for abnormal states.

    This preserves the existing final-handoff Coordinator role. This helper is
    used only when normal operation/QA/resource rules hit an abnormal state.
    If the Coordinator call fails, operation.py falls back to default_action so
    the workflow does not loop or crash.
    """
    if default_action not in allowed_actions:
        allowed_actions = list(dict.fromkeys(list(allowed_actions) + [default_action]))

    coord = CoordinatorAgent()
    decision_id = f"{int(time.time())}_{_safe_component(abnormal_state)}"

    trace_event(
        "coordinator_workflow_decision_start",
        abnormal_state=abnormal_state,
        allowed_actions=allowed_actions,
        default_action=default_action,
        affected_task_id=context_pack.get("task_id"),
    )

    try:
        out, resp_id = coord.workflow_decision(
            agent_input={
                "abnormal_state": abnormal_state,
                "context_pack": context_pack,
                "allowed_actions": allowed_actions,
                "provider_stage": "coordinator.workflow_decision",
            }
        )
        out["response_id"] = resp_id
        out["fallback_used"] = False
    except Exception as e:
        out = {
            "action": default_action,
            "target_stage": "",
            "affected_task_id": str(context_pack.get("task_id", "")),
            "reason": f"Coordinator workflow decision failed; using default_action={default_action}.",
            "required_input": "",
            "max_attempt_check": context_pack.get(
                "max_attempt_check",
                {"current_attempt": 0, "max_attempts": 0, "allowed": False},
            ),
            "notes": str(e),
            "response_id": None,
            "fallback_used": True,
        }
        rec.log(
            "coordinator_workflow_decision_failed",
            ts=_utc_now_iso(),
            abnormal_state=abnormal_state,
            error_type=e.__class__.__name__,
            error=str(e),
            default_action=default_action,
        )
        trace_event(
            "coordinator_workflow_decision_error",
            abnormal_state=abnormal_state,
            error_type=e.__class__.__name__,
            error=str(e),
            default_action=default_action,
            affected_task_id=context_pack.get("task_id"),
        )

    rec.save_json(f"coordinator_workflow_decisions/{decision_id}.json", out)
    rec.log(
        "coordinator_workflow_decision",
        ts=_utc_now_iso(),
        abnormal_state=abnormal_state,
        action=out.get("action"),
        target_stage=out.get("target_stage"),
        affected_task_id=out.get("affected_task_id"),
        fallback_used=out.get("fallback_used"),
    )
    trace_event(
        "coordinator_workflow_decision_end",
        abnormal_state=abnormal_state,
        action=out.get("action"),
        target_stage=out.get("target_stage"),
        affected_task_id=out.get("affected_task_id"),
        fallback_used=out.get("fallback_used"),
        response_id=out.get("response_id"),
        decision_id=decision_id,
    )
    return out


class _Attempts:
    def __init__(self, initial: Optional[Dict[str, Any]] = None) -> None:
        self.mu = threading.Lock()
        self.n: Dict[str, int] = {}
        if isinstance(initial, dict):
            for k, v in initial.items():
                try:
                    self.n[str(k)] = int(v)
                except Exception:
                    continue

    def bump(self, task_id: str) -> int:
        with self.mu:
            self.n[task_id] = self.n.get(task_id, 0) + 1
            return self.n[task_id]

    def peek_next(self, task_id: str) -> int:
        with self.mu:
            return self.n.get(task_id, 0) + 1


class _ResponseState:
    """Task-scoped previous_response_id storage shared across worker threads."""

    def __init__(self) -> None:
        self.mu = threading.Lock()
        self.eng: Dict[str, str] = {}
        self.qa: Dict[str, str] = {}

    def get(self, role: str, task_id: str) -> Optional[str]:
        with self.mu:
            store = self.eng if role == "engineer" else self.qa
            return store.get(str(task_id))

    def set(self, role: str, task_id: str, response_id: Optional[str]) -> None:
        if not response_id:
            return
        with self.mu:
            store = self.eng if role == "engineer" else self.qa
            store[str(task_id)] = str(response_id)


def _extract_qa_issues(qa_out: Any) -> List[Dict[str, Any]]:
    if not isinstance(qa_out, dict):
        return []
    review = qa_out.get("review") if isinstance(qa_out.get("review"), dict) else {}
    issues = review.get("issues") if isinstance(review.get("issues"), list) else []
    out: List[Dict[str, Any]] = []
    for issue in issues:
        if isinstance(issue, dict):
            out.append({
                "severity": str(issue.get("severity") or ""),
                "title": str(issue.get("title") or ""),
                "detail": str(issue.get("detail") or ""),
                "evidence": str(issue.get("evidence") or ""),
                "required_action": str(issue.get("required_action") or ""),
                "rerun_verification": issue.get("rerun_verification") if isinstance(issue.get("rerun_verification"), list) else [],
            })
    return out


def _build_repair_packet(
    *,
    task_id: str,
    attempt_n: int,
    engineer_id: str,
    work_item: Dict[str, Any],
    eng_out: Dict[str, Any],
    qa_out: Dict[str, Any],
    staged_write_report: Dict[str, Any],
    expected_files: List[str],
    workflow_decision: Optional[Dict[str, Any]] = None,
    user_directive: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create the durable bridge from QA failure to the next Engineer retry.

    A QA block must not be only a complaint. This packet gives the next Engineer
    exact failure evidence, implicated files, required fixes, and verification
    commands while preserving the current workflow's internal agent structure.
    """
    issues = _extract_qa_issues(qa_out)
    required_fixes: List[str] = []
    implicated_files = set()
    verification_commands: List[str] = []

    for issue in issues:
        action = str(issue.get("required_action") or issue.get("detail") or "").strip()
        title = str(issue.get("title") or "").strip()
        if action or title:
            required_fixes.append((title + (": " if title and action else "") + action).strip())
        for cmd in issue.get("rerun_verification") or []:
            if str(cmd).strip():
                verification_commands.append(str(cmd).strip())

    for raw in expected_files or []:
        if str(raw).strip():
            implicated_files.add(str(raw).strip())
    evidence = staged_write_report.get("verification_evidence") if isinstance(staged_write_report, dict) else {}
    if isinstance(evidence, dict):
        for key in ("files_written", "scope_violations"):
            val = evidence.get(key)
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, str) and item.strip():
                        implicated_files.add(item.strip())
                    elif isinstance(item, dict) and item.get("path"):
                        implicated_files.add(str(item.get("path")).strip())

    if not required_fixes:
        note = str(qa_out.get("notes") or qa_out.get("blocked_reason") or "QA did not pass this attempt; revise using the QA output.").strip()
        required_fixes.append(note)

    if not verification_commands:
        verification_commands = [str(x) for x in (work_item.get("verification") or []) if str(x).strip()]

    code_files = []
    try:
        for item in (eng_out.get("code_output") or {}).get("files") or []:
            if isinstance(item, dict) and str(item.get("path") or "").strip():
                code_files.append({
                    "path": str(item.get("path")).strip(),
                    "write_mode": str(item.get("write_mode") or ""),
                    "content_chars": len(str(item.get("content") or "")),
                })
    except Exception:
        code_files = []

    return {
        "schema_version": "repair_packet.v1",
        "task_id": task_id,
        "created_at_utc": _utc_now_iso(),
        "source_attempt": {"attempt": attempt_n, "engineer_id": engineer_id},
        "failure_type": "qa_or_verification_failed",
        "work_item_summary": {
            "summary": work_item.get("summary"),
            "scope_in": work_item.get("scope_in"),
            "scope_out": work_item.get("scope_out"),
            "acceptance_criteria": work_item.get("acceptance_criteria") or [],
        },
        "qa_verdict": qa_out.get("review", {}).get("verdict") if isinstance(qa_out.get("review"), dict) else qa_out.get("verdict"),
        "qa_issues": issues,
        "required_fixes": required_fixes,
        "implicated_files": sorted(x for x in implicated_files if x),
        "do_not_change": [str(x) for x in (work_item.get("scope_out") if isinstance(work_item.get("scope_out"), list) else [work_item.get("scope_out")]) if str(x).strip()],
        "verification_commands": verification_commands,
        "expected_files": expected_files,
        "staged_write_report": staged_write_report,
        "previous_attempt_file_summary": code_files,
        "workflow_decision": workflow_decision or {},
        "user_directive": user_directive or {},
        "next_action": "retry_same_task_or_ask_team_lead",
    }


def _save_repair_packet(rec: Recorder, repair_packet: Dict[str, Any]) -> None:
    task_id = str(repair_packet.get("task_id") or "unknown")
    rec.save_json(f"repair_packets/{task_id}.json", repair_packet)


def _extract_task_ids_from_context(obj: Any) -> List[str]:
    from task_id_utils import extract_task_ids_from_context
    return extract_task_ids_from_context(obj)


def _canonical_task_id_for_directive(fallback_task_id: str, directive: Dict[str, Any]) -> str:
    from task_id_utils import canonical_task_id
    return canonical_task_id(fallback_task_id, directive)


def _merge_directive_dict(existing: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(existing or {})
    for key, value in (updates or {}).items():
        if value is None:
            continue
        if key == "overrides":
            old = merged.get("overrides") if isinstance(merged.get("overrides"), dict) else {}
            new = value if isinstance(value, dict) else {}
            merged[key] = {**old, **new}
        elif key == "request_context":
            old = merged.get("request_context") if isinstance(merged.get("request_context"), dict) else {}
            new = value if isinstance(value, dict) else {}
            merged[key] = {**old, **new}
        elif isinstance(value, list) and key in {"provided_assets", "request_items", "request_options"}:
            if value or key not in merged:
                merged[key] = value
        elif value != "" or key not in merged:
            merged[key] = value
    return merged


def _load_task_directive(resource_dir: str, task_id: str) -> Optional[Dict[str, Any]]:
    try:
        p = Path(resource_dir) / "human_task_directives" / f"{task_id}.json"
        if not p.exists():
            return None
        obj = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            return None
        canonical = _canonical_task_id_for_directive(str(task_id), obj)
        if canonical and canonical != str(task_id):
            # Do not feed a T4 directive into T3 or vice versa. The startup
            # quarantine/retarget pass will move it; this loader is defensive.
            return None
        return obj
    except Exception:
        return None
    return None


def _load_resource_task_directive(resource_dir: str, task_id: str) -> Optional[Dict[str, Any]]:
    """Load task_directives/<task_id>.json defensively.

    Resource decisions can also carry route-around / accept-limitation choices
    that should break the T3/T4 evidence loop. Keep this separate from
    _load_task_directive() so existing Engineer/QA inputs can still distinguish
    resource_decision from human_decision.
    """
    try:
        p = Path(resource_dir) / "task_directives" / f"{task_id}.json"
        if not p.exists():
            return None
        obj = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            return None
        canonical = _canonical_task_id_for_directive(str(task_id), obj)
        if canonical and canonical != str(task_id):
            return None
        return obj
    except Exception:
        return None


def _load_combined_task_directive(resource_dir: str, task_id: str) -> Optional[Dict[str, Any]]:
    """Merge human and resource directives for deterministic resume repair.

    Auto-accept/unblock code needs to see both sources: T3 route-around choices
    may be stored in task_directives/ after a resource fallback, while max-attempt
    choices are stored in human_task_directives/. Human directives win for top
    level fields, but text/context from both files is preserved for detection.
    """
    human = _load_task_directive(resource_dir, task_id)
    resource = _load_resource_task_directive(resource_dir, task_id)
    if not human and not resource:
        return None
    if human and not resource:
        return human
    if resource and not human:
        return resource
    merged = _merge_directive_dict(resource or {}, human or {})
    messages = []
    for label, obj in (("resource", resource), ("human", human)):
        if isinstance(obj, dict) and str(obj.get("user_message") or "").strip():
            messages.append(f"[{label}] {obj.get('user_message')}")
    if messages:
        merged["user_message"] = "\n".join(messages)
    merged["combined_directive_sources"] = [
        src for src, obj in (("task_directives", resource), ("human_task_directives", human)) if isinstance(obj, dict)
    ]
    return merged


def _quarantine_or_retarget_mismatched_directives(*, resource_dir: str, rec: Recorder) -> None:
    """Move task directives whose structured context belongs to a different task.

    Previous runs could write a T4 answer to human_task_directives/T3.json and a
    T3 answer to human_task_directives/T4.json. This function runs at startup so
    resume does not keep feeding poisoned instructions to the wrong task.
    """
    base = Path(resource_dir)
    changed: List[Dict[str, Any]] = []
    for folder_name in ("human_task_directives", "task_directives"):
        folder = base / folder_name
        if not folder.exists():
            continue
        quarantine = folder / ".quarantine_mismatched"
        for path in list(folder.glob("*.json")):
            try:
                obj = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            file_task = path.stem
            canonical = _canonical_task_id_for_directive(file_task, obj)
            if not canonical or canonical == file_task:
                continue
            quarantine.mkdir(parents=True, exist_ok=True)
            quarantine_path = quarantine / f"{file_task}__to__{canonical}__{int(time.time())}.json"
            try:
                shutil.copy2(path, quarantine_path)
            except Exception:
                pass
            target = folder / f"{canonical}.json"
            existing: Dict[str, Any] = {}
            try:
                if target.exists():
                    loaded = json.loads(target.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        existing = loaded
            except Exception:
                existing = {}
            obj["retargeted_from_task_id"] = file_task
            obj["retargeted_at_utc"] = _utc_now_iso()
            merged = _merge_directive_dict(existing, obj)
            target.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            try:
                path.unlink()
            except Exception:
                pass
            changed.append({"folder": folder_name, "from": file_task, "to": canonical, "quarantine": str(quarantine_path)})
    if changed:
        rec.save_json("directive_retarget_report.json", {"created_at_utc": _utc_now_iso(), "items": changed})
        rec.log("directive_retargeted_mismatches", ts=_utc_now_iso(), count=len(changed))


def _directive_intent_text(directive: Optional[Dict[str, Any]]) -> str:
    """Return only the user's resolved intent, excluding the original prompt.

    Max-attempt and resource prompts often include unchosen options such as
    "+2 attempts", "another engineer", or "defer runtime evidence" inside
    request_context. Intent-classification helpers must not scan that prompt
    text, or a route-around/deny answer can inherit the opposite option and
    recreate the retry loop.
    """
    if not isinstance(directive, dict):
        return ""
    parts = [
        str(directive.get("decision") or ""),
        str(directive.get("user_message") or ""),
        json.dumps(directive.get("overrides") or {}, ensure_ascii=False),
    ]
    return "\n".join(parts).lower()


def _directive_text(directive: Optional[Dict[str, Any]]) -> str:
    """Return directive plus request metadata for diagnostics/context-only checks.

    Do not use this for user intent decisions such as retry extension, reassign,
    or evidence deferral; use _directive_intent_text() for those paths.
    """
    if not isinstance(directive, dict):
        return ""
    parts = [
        _directive_intent_text(directive),
        str(directive.get("request_reason") or ""),
        str(directive.get("request_stage") or ""),
        str(directive.get("request_agent") or ""),
        json.dumps(directive.get("request_context") or {}, ensure_ascii=False),
    ]
    return "\n".join(parts).lower()


def _directive_blocks_task(directive: Optional[Dict[str, Any]]) -> bool:
    """True only when the user explicitly hard-stops the whole task.

    A user may block a requested resource/feature without blocking the whole
    engineering work item. Route-around directives must unblock the worker with
    constraints; otherwise the UI shows "Human Needed" forever after the user
    thought they chose a restriction/fallback path.
    """
    if not isinstance(directive, dict):
        return False
    decision = str(directive.get("decision") or "").strip().lower()
    msg = str(directive.get("user_message") or "").strip().lower()
    overrides = directive.get("overrides") if isinstance(directive.get("overrides"), dict) else {}
    if overrides.get("user_blocked_task") is False or overrides.get("route_around_blocked_item"):
        return False
    if decision in {"accept_limitation", "mark_unavailable", "continue", "provided", "route_around", "defer"}:
        return False
    if any(neg in msg for neg in ("do not block", "don't block", "not block", "route around", "blocked the requested item", "requested item/resource only", "reduce scope", "defer")):
        return False
    if bool(overrides.get("user_blocked_task")):
        return True
    if decision in {"block", "block_task", "block task", "halt", "stop"}:
        return True
    return msg.startswith("block whole") or msg.startswith("stop whole") or "halt work" in msg or "stop work" in msg or "hard stop" in msg


def _directive_requests_reassign(directive: Optional[Dict[str, Any]]) -> bool:
    text = _directive_intent_text(directive).strip()
    if not text or _directive_blocks_task(directive):
        return False
    return (
        "reassign" in text
        or "assign to another engineer" in text
        or "use another engineer" in text
        or text.startswith("c.")
        or text.startswith("c ")
    )


def _directive_context_candidates(directive: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return likely context dictionaries carried by human/resource directives.

    UI fallback requests may wrap the original run_state payload under
    `block_payload` and the original Team Lead decision under `last_decision`.
    Older/pending requests usually store the useful fields directly under
    `request_context`. Keeping this extraction centralized prevents reassign and
    attempt-extension logic from silently falling back to stale run_state fields.
    """
    if not isinstance(directive, dict):
        return []
    candidates: List[Dict[str, Any]] = [directive]
    for key in ("request_context", "context"):
        value = directive.get(key)
        if isinstance(value, dict):
            candidates.append(value)
            bp = value.get("block_payload")
            if isinstance(bp, dict):
                candidates.append(bp)
                bp_ctx = bp.get("context")
                if isinstance(bp_ctx, dict):
                    candidates.append(bp_ctx)
            last = value.get("last_decision")
            if isinstance(last, dict):
                candidates.append(last)
                last_ctx = last.get("context_pack")
                if isinstance(last_ctx, dict):
                    candidates.append(last_ctx)
    return candidates


def _directive_reassign_skip_engineer_id(directive: Optional[Dict[str, Any]]) -> str:
    """Return the exact engineer id a reassign directive should skip, if known.

    Reassign must be scoped to the engineer involved in the request that the
    user resolved. It must not keep following run_state.active_engineer_id,
    because that field changes when the next engineer starts and caused the
    previous version to skip the replacement engineer too.
    """
    if not isinstance(directive, dict):
        return ""
    overrides = directive.get("overrides") if isinstance(directive.get("overrides"), dict) else {}
    for key in ("skip_engineer_id", "previous_engineer_id", "engineer_id"):
        val = str(overrides.get(key) or "").strip()
        if val:
            return val
    for ctx in _directive_context_candidates(directive):
        for key in ("previous_engineer_id", "engineer_id"):
            val = str(ctx.get(key) or "").strip()
            if val:
                return val
        wf = ctx.get("workflow_decision") if isinstance(ctx.get("workflow_decision"), dict) else {}
        nested = wf.get("context_pack") if isinstance(wf.get("context_pack"), dict) else {}
        for key in ("previous_engineer_id", "engineer_id"):
            val = str(nested.get(key) or "").strip()
            if val:
                return val
    return ""


def _mark_reassign_directive_consumed(resource_dir: str, task_id: str, engineer_id: str) -> None:
    """Mark a human reassign directive consumed once it skipped its intended engineer."""
    try:
        p = Path(resource_dir) / "human_task_directives" / f"{task_id}.json"
        if not p.exists():
            return
        obj = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            return
        obj["reassign_consumed"] = True
        obj["reassign_consumed_by"] = engineer_id
        obj["reassign_consumed_at_utc"] = _utc_now_iso()
        p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception:
        return


def _directive_is_max_attempt_resolution(directive: Optional[Dict[str, Any]]) -> bool:
    """True when a human directive resolves a max-attempt Team Lead request.

    UI dropdown values such as ``accept_limitation`` or ``continue`` do not
    themselves contain words like ``rerun``. If they resolve a max-attempt
    request, however, they must still authorize at least one bounded retry;
    otherwise the task immediately re-enters the same max-attempt block.
    """
    if not isinstance(directive, dict):
        return False
    if _directive_blocks_task(directive):
        return False
    for ctx in _directive_context_candidates(directive):
        text_parts = [
            str(ctx.get("request_reason") or ""),
            str(ctx.get("reason") or ""),
            str(ctx.get("stage") or ""),
            str(ctx.get("target_stage") or ""),
        ]
        wd = ctx.get("workflow_decision") if isinstance(ctx.get("workflow_decision"), dict) else {}
        text_parts.extend([
            str(wd.get("abnormal_state") or ""),
            str(wd.get("target_stage") or ""),
            str(wd.get("reason") or ""),
        ])
        if isinstance(wd.get("max_attempt_check"), dict):
            return True
        if isinstance(ctx.get("max_attempt_check"), dict):
            return True
        combined = "\n".join(text_parts).lower()
        if "max_attempt" in combined or "max attempts" in combined or "attempts exhausted" in combined:
            return True
    return False


def _directive_extends_attempts(directive: Optional[Dict[str, Any]]) -> bool:
    """Return True only when the resolved user intent grants more attempts.

    The original max-attempt question may contain every option (A/B/C/D), so
    scanning request_context would make a route-around answer look like a +2
    retry approval. Use explicit overrides or user_message/decision only.
    """
    if not isinstance(directive, dict) or _directive_blocks_task(directive):
        return False
    overrides = directive.get("overrides") if isinstance(directive.get("overrides"), dict) else {}
    if overrides.get("extra_attempts_granted") is not None or directive.get("extra_attempts_granted") is not None:
        return True
    decision = str(directive.get("decision") or "").strip().lower()
    text = _directive_intent_text(directive)
    if decision in {"rerun", "retry", "extend_attempts", "reassign"}:
        return True
    if decision in {"accept_limitation", "mark_unavailable", "route_around", "defer", "block"}:
        return False
    return any(term in text for term in (
        "rerun",
        "retry",
        "additional attempt",
        "additional attempts",
        "approve +",
        "authorize +",
        "increase max_attempts",
        "increase max attempts",
        "set max_attempts",
        "set max attempts",
        "new max_attempts",
        "new max attempts",
        "attempt budget",
        "+1",
        "+2",
        "+3",
        "attempt_4",
        "attempt 4",
        "reassign",
        "assign to another engineer",
        "use another engineer",
    ))


def _effective_max_attempts(base_max: int, attempt_n: int, directive: Optional[Dict[str, Any]]) -> int:
    """Apply explicit human approval without silently allowing infinite retries.

    A user approval such as "+2 attempts" should create a bounded extension from
    the attempt count that triggered the human request. A generic reassign/rerun
    directive grants one additional attempt. Stale directive files therefore do
    not keep extending forever as attempt_n increases.
    """
    try:
        base = int(base_max)
    except Exception:
        base = 3
    if not _directive_extends_attempts(directive):
        return base

    text = _directive_intent_text(directive)
    extra = 1
    if isinstance(directive, dict):
        overrides = directive.get("overrides") if isinstance(directive.get("overrides"), dict) else {}
        explicit_extra = overrides.get("extra_attempts_granted")
        if explicit_extra is None:
            explicit_extra = directive.get("extra_attempts_granted")
        try:
            if explicit_extra is not None:
                extra = max(1, min(10, int(explicit_extra)))
        except Exception:
            pass
    # Apply text-derived approvals cumulatively. Do not use ``elif`` here: a
    # human note may mention more than one option, and the safest bounded
    # interpretation is the largest explicit approval.
    if "+1" in text or "one additional" in text or "1 additional" in text:
        extra = max(extra, 1)
    if "+2" in text or "two additional" in text or "2 additional" in text:
        extra = max(extra, 2)
    if "+3" in text or "three additional" in text or "3 additional" in text:
        extra = max(extra, 3)

    absolute_max: Optional[int] = None
    for pattern in (
        r"max_attempts\s*(?:to|=|:)\s*(\d{1,3})",
        r"max attempts\s*(?:to|=|:)\s*(\d{1,3})",
        r"new max_attempts\s*(?:to|=|:)\s*(\d{1,3})",
        r"new max attempts\s*(?:to|=|:)\s*(\d{1,3})",
        r"attempt budget\s*(?:to|=|:)\s*(\d{1,3})",
    ):
        m = re.search(pattern, text)
        if m:
            try:
                absolute_max = max(1, min(100, int(m.group(1))))
                break
            except Exception:
                pass

    trigger_attempt = base
    if isinstance(directive, dict):
        candidates: List[Any] = []
        for ctx in _directive_context_candidates(directive):
            candidates.append(ctx.get("attempt"))
            wd = ctx.get("workflow_decision") if isinstance(ctx.get("workflow_decision"), dict) else {}
            mac = wd.get("max_attempt_check") if isinstance(wd.get("max_attempt_check"), dict) else {}
            candidates.append(mac.get("current_attempt"))
            direct_mac = ctx.get("max_attempt_check") if isinstance(ctx.get("max_attempt_check"), dict) else {}
            candidates.append(direct_mac.get("current_attempt"))
        for candidate in candidates:
            try:
                if candidate is not None:
                    trigger_attempt = max(trigger_attempt, int(candidate))
            except Exception:
                pass
    bounded_extra_max = max(base, trigger_attempt + extra)
    if absolute_max is not None:
        return max(base, min(max(absolute_max, trigger_attempt), trigger_attempt + 10))
    return bounded_extra_max


def _worker_loop(
    *,
    engineer_id: str,
    orch: EngineeringOrchestratorAgent,
    qa: QAAgent,
    rec: Recorder,
    shared_context: Dict[str, Any],
    attempts: _Attempts,
    response_state: _ResponseState,
    max_attempts_per_task: int,
    resource_dir: str,
    human_desk: Optional[HumanRequestDesk] = None,
    run_state: Optional[RunState] = None,
) -> None:
    eng = EngineerAgent()
    res_eval = ResourceEval(resource_dir)
    human_desk = human_desk or HumanRequestDesk(resource_dir)

    while True:
        _maybe_wait_if_paused(rec)

        claim = orch.run(agent_input={
            "command": "claim_next",
            "engineer_id": engineer_id,
            "engineer_capabilities": ["generalist"],
            "allow_partial_capability_match": True,
        })
        if not isinstance(claim, dict) or not claim.get("claimed"):
            # Don't exit if the queue is blocked waiting for user-provided resources.
            try:
                blk = orch.run(agent_input={"command": "list_blocked"}).get("blocked", [])
                if isinstance(blk, list) and any(
                    ("needs_user_resources" in str(x.get("notes", "")))
                    or ("coordinator_requires_user_input" in str(x.get("notes", "")))
                    or ("team_lead_requires_user_input" in str(x.get("notes", "")))
                    or ("human_input" in str(x.get("notes", "")))
                    for x in blk if isinstance(x, dict)
                ):
                    time.sleep(1.5)
                    continue
            except Exception:
                pass
            return

        wi = claim.get("work_item") or {}
        task_id = _as_str(wi.get("task_id", "")) or f"unknown_{int(time.time())}"

        # Load any user correction/redirect for this task before consuming an
        # attempt. This lets a "reassign" decision release the previous engineer's
        # accidental/stale claim without burning another paid attempt.
        directive = None
        human_directive = None
        try:
            td_path = Path(resource_dir) / "task_directives" / f"{task_id}.json"
            if td_path.exists():
                directive = json.loads(td_path.read_text(encoding="utf-8"))
        except Exception:
            directive = None
        human_directive = _load_task_directive(resource_dir, task_id)

        if (
            _directive_requests_reassign(human_directive)
            and not bool((human_directive or {}).get("reassign_consumed"))
            and run_state is not None
        ):
            previous_engineer_id = _directive_reassign_skip_engineer_id(human_directive)
            # Legacy fallback only: if old directive files lack request context, use
            # active_engineer_id. New directives should carry request_context.engineer_id.
            if not previous_engineer_id:
                previous_engineer_id = str((run_state.state or {}).get("active_engineer_id") or "")
            if previous_engineer_id and previous_engineer_id == engineer_id:
                try:
                    released = orch.run(agent_input={
                        "command": "release_task",
                        "task_id": task_id,
                        "only_if_claimed_by": engineer_id,
                        "note": "human_requested_reassign_previous_engineer_released",
                    })
                except Exception as exc:
                    released = {"ok": False, "error": str(exc)}
                directive_consumed = bool(isinstance(released, dict) and released.get("released"))
                if directive_consumed:
                    _mark_reassign_directive_consumed(resource_dir, task_id, engineer_id)
                rec.log(
                    "task_reassign_skipped_previous_engineer",
                    ts=_utc_now_iso(),
                    task_id=task_id,
                    engineer_id=engineer_id,
                    previous_engineer_id=previous_engineer_id,
                    released=released,
                    directive_consumed=directive_consumed,
                )
                time.sleep(0.25)
                return

        # Check the max-attempt gate before consuming an engineering attempt.
        # A gate-only block writes no code and should not count as an Engineer
        # implementation attempt. This prevents the "approve more attempts" loop
        # from burning the newly approved attempt before the Engineer can act.
        next_attempt_n = attempts.peek_next(task_id)
        effective_max_attempts = _effective_max_attempts(max_attempts_per_task, next_attempt_n, human_directive)
        attempt_n = next_attempt_n
        if effective_max_attempts != max_attempts_per_task:
            rec.log(
                "attempt_limit_override_applied",
                ts=_utc_now_iso(),
                task_id=task_id,
                engineer_id=engineer_id,
                attempt=attempt_n,
                base_max_attempts=max_attempts_per_task,
                effective_max_attempts=effective_max_attempts,
            )

        if attempt_n > effective_max_attempts:
            decision = _coordinator_workflow_decision(
                rec=rec,
                abnormal_state="max_attempts_exceeded",
                context_pack={
                    "run_id": shared_context.get("run_id", ""),
                    "task_id": task_id,
                    "engineer_id": engineer_id,
                    "work_item": wi,
                    "max_attempt_check": {
                        "current_attempt": attempt_n,
                        "max_attempts": effective_max_attempts,
                        "base_max_attempts": max_attempts_per_task,
                        "allowed": False,
                    },
                    "available_context_keys": sorted(shared_context.keys()),
                    "normal_policy": "Max attempts are exhausted. Team Lead may ask_user or block; rerun is not allowed by operation.py.",
                },
                allowed_actions=["ask_user", "block"],
                default_action="block",
            )
            if run_state is not None:
                run_state.set_last_decision(decision)
            blocked_reason = "max_attempts_exceeded"
            human_request_id = None
            if decision.get("action") == "ask_user":
                blocked_reason = "team_lead_requires_user_input_after_max_attempts"
                human_request_id = _submit_human_request(
                    human_desk=human_desk,
                    rec=rec,
                    task_id=task_id,
                    stage="engineering.max_attempts",
                    reason=blocked_reason,
                    question=str(decision.get("required_input") or "The task exceeded max attempts. Provide direction, reduce scope, or approve blocking this task."),
                    options=["rerun with my clarification", "reduce scope", "accept limitation and continue", "block requested item / route around", "hard stop whole task"],
                    context={"workflow_decision": decision, "work_item": wi, "engineer_id": engineer_id, "attempt": attempt_n},
                    run_state=run_state,
                )
            cmd = {
                "command": "submit_result",
                "result": {
                    "task_id": task_id,
                    "engineer_id": engineer_id,
                    "status": "blocked",
                    "changes": None,
                    "handoff_interfaces": [],
                    "notes": f"Exceeded max attempts ({effective_max_attempts}). Team Lead workflow action: {decision.get('action')}. human_request_id={human_request_id or ''}",
                    "verification_run": ["blocked: max_attempts_exceeded"],
                    "workflow_decision": decision,
                },
                "mark_done": False,
                "mark_blocked": True,
                "blocked_reason": blocked_reason,
            }
            rec.save_json(f"orch_submissions/{task_id}.json", cmd)
            orch.run(agent_input=cmd)
            rec.log("task_blocked", task_id=task_id, engineer_id=engineer_id, attempt=attempt_n, reason=blocked_reason)
            # If Team Lead asked the user what to do, _submit_human_request()
            # already wrote a user-facing human_input block with the request_id,
            # question, options, and context. Do not overwrite it with a generic
            # engineering_task block, or the UI loses the actionable request.
            if run_state is not None and human_request_id is None:
                run_state.mark_blocked(
                    node=NODE_ENGINEERING,
                    block_type="engineering_task",
                    reason=blocked_reason,
                    payload={"task_id": task_id, "engineer_id": engineer_id, "attempt": attempt_n},
                )
            continue

        attempt_n = attempts.bump(task_id)
        if run_state is not None:
            run_state.mark_task_attempt(task_id=task_id, engineer_id=engineer_id, attempt=attempt_n)

        retry_context = None
        try:
            retry_path = rec.out_dir / "retry_contexts" / f"{task_id}.json"
            if retry_path.exists():
                retry_context = json.loads(retry_path.read_text(encoding="utf-8"))
        except Exception:
            retry_context = None

        repair_packet = None
        try:
            repair_path = rec.out_dir / "repair_packets" / f"{task_id}.json"
            if repair_path.exists():
                repair_packet = json.loads(repair_path.read_text(encoding="utf-8"))
        except Exception:
            repair_packet = None

        rec.log("engineer_start", task_id=task_id, engineer_id=engineer_id, attempt=attempt_n, ts=_utc_now_iso())
        trace_event("engineer_attempt_start", task_id=task_id, engineer_id=engineer_id, attempt=attempt_n)

        eng_repo_context = dict(shared_context)
        eng_repo_context["team_lead_directives"] = _read_team_lead_directives(
            resource_dir,
            target_agents=["Engineer(s)", "QA Agent", "Engineering Lead", "Orchestrator", "Team Lead / Intake"],
            task_id=task_id,
        )
        try:
            eng_repo_context["compiled_repo_context"] = compile_task_context(
                work_item=wi,
                workspace_dir=rec.out_dir / "workspace",
                orchestrator_results=orch.run(agent_input={"command": "get_results"}),
            )
        except Exception as exc:
            eng_repo_context["compiled_repo_context_error"] = str(exc)
        if isinstance(retry_context, dict) and retry_context:
            eng_repo_context["retry_context"] = retry_context
        if isinstance(repair_packet, dict) and repair_packet:
            eng_repo_context["repair_packet"] = repair_packet
        eng_in = {"work_item": wi, "engineer_id": engineer_id, "repo_context": eng_repo_context}
        if isinstance(directive, dict) and directive:
            eng_in["resource_decision"] = directive
        if isinstance(human_directive, dict) and human_directive:
            eng_in["human_decision"] = human_directive

        draft_for_retry = retry_context.get("previous_engineer_output") if isinstance(retry_context, dict) else None
        feedback_for_retry = retry_context.get("feedback_for_engineer") if isinstance(retry_context, dict) else None
        prev_eng = response_state.get("engineer", task_id)

        while True:
            _maybe_wait_if_paused(rec)
            try:
                eng_out, prev_eng = eng.run(
                    agent_input=eng_in,
                    draft=draft_for_retry if isinstance(draft_for_retry, dict) else None,
                    feedback=str(feedback_for_retry) if feedback_for_retry else None,
                    previous_response_id=prev_eng,
                )
                response_state.set("engineer", task_id, prev_eng)
                break
            except Exception as e:
                if _looks_like_quota_or_rate_limit_error(e):
                    trace_event("engineer_attempt_paused", task_id=task_id, engineer_id=engineer_id, reason="openai_quota_or_rate_limit", error_type=e.__class__.__name__, error=str(e))
                    _pause_run(rec, reason="openai_quota_or_rate_limit", err=e)
                    _wait_for_resume(rec)
                    continue
                trace_event("engineer_attempt_error", task_id=task_id, engineer_id=engineer_id, error_type=e.__class__.__name__, error=str(e))
                raise

        # Deterministically stage EngineerAgent code_output. The real workspace is updated only after QA passes.
        raw_code_output = eng_out.get("code_output", {}) if isinstance(eng_out, dict) else {}
        expected_files = [str(x) for x in (wi.get("files_expected") or []) if str(x).strip()]
        expected_files = _expand_expected_files_for_scaffold_placeholders(expected_files, wi, raw_code_output)
        staging_dir = rec.out_dir / ".attempts" / task_id / f"attempt_{attempt_n}__{engineer_id}"
        candidate_dir = _prepare_candidate_workspace(
            out_dir=rec.out_dir,
            task_id=task_id,
            attempt_n=attempt_n,
            engineer_id=engineer_id,
            expected_files=expected_files,
        )
        staged_write_report = write_code_output(
            code_output=raw_code_output,
            workspace_dir=candidate_dir,
            task_id=task_id,
            engineer_id=engineer_id,
            allowed_paths=expected_files,
            enforce_allowed_paths=True,
        )
        # Keep the legacy .attempts delta folder for audit/backward compatibility,
        # while QA and promotion use the full candidate snapshot.
        legacy_staged_write_report = write_code_output(
            code_output=raw_code_output,
            workspace_dir=staging_dir,
            task_id=task_id,
            engineer_id=engineer_id,
            allowed_paths=expected_files,
            enforce_allowed_paths=True,
        )
        staged_write_report["candidate_workspace_dir"] = str(candidate_dir)
        staged_write_report["legacy_attempt_dir"] = str(staging_dir)
        staged_write_report["legacy_attempt_write_report"] = legacy_staged_write_report
        eng_out["staged_write_report"] = staged_write_report
        eng_out["local_write_report"] = {
            "status": "pending_qa_approval",
            "reason": "Writes are staged into a full candidate workspace and promoted only if QA marks the task done.",
            "candidate_workspace_dir": str(candidate_dir),
            "staged_write_report": staged_write_report,
        }
        rec.save_json(f"staged_file_writes/{task_id}__attempt_{attempt_n}__{engineer_id}.json", staged_write_report)
        rec.log(
            "local_file_write_staged",
            task_id=task_id,
            engineer_id=engineer_id,
            attempt=attempt_n,
            status=staged_write_report.get("status"),
            files_written=len(staged_write_report.get("files_written", [])),
            files_skipped=len(staged_write_report.get("files_skipped", [])),
            scope_violations=len((staged_write_report.get("scope") or {}).get("violations", [])),
        )

        # Deterministic deadlock breaker: a repeated scope error is usually a bad
        # WorkItem files_expected contract, not an Engineer quality problem. Stop
        # burning retries and ask for a task-scope decision.
        if staged_write_report.get("status") == "scope_error" and attempt_n >= 2:
            human_request_id = _submit_human_request(
                human_desk=human_desk,
                rec=rec,
                task_id=task_id,
                stage="engineering.file_scope",
                reason="task_spec_scope_defect",
                question="The Engineer is trying to write files outside this task's files_expected scope. Expand the task scope, narrow the implementation, or block the task.",
                options=["expand files_expected and rerun", "narrow implementation to allowed files", "block requested item / route around", "hard stop whole task"],
                context={"work_item": wi, "staged_write_report": staged_write_report, "engineer_id": engineer_id, "attempt": attempt_n},
                run_state=run_state,
            )
            cmd = {
                "command": "submit_result",
                "result": {
                    "task_id": task_id,
                    "engineer_id": engineer_id,
                    "status": "blocked",
                    "changes": eng_out.get("changes"),
                    "code_output": eng_out.get("code_output"),
                    "staged_write_report": staged_write_report,
                    "verification_run": [f"blocked: task_spec_scope_defect human_request_id={human_request_id}"],
                    "notes": f"Repeated local-file scope error. human_request_id={human_request_id}",
                    "handoff_interfaces": eng_out.get("handoff_interfaces", []),
                },
                "mark_done": False,
                "mark_blocked": True,
                "blocked_reason": "task_spec_scope_defect",
            }
            rec.save_json(f"orch_submissions/{task_id}.json", cmd)
            orch.run(agent_input=cmd)
            rec.log("task_blocked", ts=_utc_now_iso(), task_id=task_id, engineer_id=engineer_id, attempt=attempt_n, reason="task_spec_scope_defect")
            continue

        rec.save_json(f"engineer/{task_id}__attempt_{attempt_n}__{engineer_id}.json", eng_out)
        rec.log("engineer_end", task_id=task_id, engineer_id=engineer_id, attempt=attempt_n, ts=_utc_now_iso(), response_id=prev_eng)
        trace_event("engineer_attempt_end", task_id=task_id, engineer_id=engineer_id, attempt=attempt_n, response_id=prev_eng, code_output_present=bool(eng_out.get("code_output") if isinstance(eng_out, dict) else False), local_write_status=staged_write_report.get("status"))

        rec.log("qa_start", task_id=task_id, engineer_id=engineer_id, attempt=attempt_n, ts=_utc_now_iso())
        trace_event("qa_review_start", task_id=task_id, engineer_id=engineer_id, attempt=attempt_n)
        qa_in = {
            "work_item": wi,
            "work_result": eng_out,
            "verification_artifacts": {
                "staged_write_report": staged_write_report,
                "file_scope_enforced": True,
                "file_scope_policy": "strict_files_expected_empty_means_no_file_writes",
                "expected_files": expected_files,
                "candidate_workspace_dir": str(candidate_dir),
                "candidate_workspace_file_count": _count_materialized_files(candidate_dir),
            },
        }
        if isinstance(directive, dict) and directive:
            qa_in["resource_decision"] = directive
        if isinstance(human_directive, dict) and human_directive:
            qa_in["human_decision"] = human_directive

        while True:
            _maybe_wait_if_paused(rec)
            try:
                prev_qa = response_state.get("qa", task_id)
                qa_out, prev_qa = qa.run(agent_input=qa_in, previous_response_id=prev_qa)
                response_state.set("qa", task_id, prev_qa)
                break
            except Exception as e:
                if _looks_like_quota_or_rate_limit_error(e):
                    _pause_run(rec, reason="openai_quota_or_rate_limit", err=e)
                    _wait_for_resume(rec)
                    continue
                raise

        accepted_by_deferred_evidence, qa_out = _conditionally_accept_deferred_evidence_block(
            qa_out=qa_out if isinstance(qa_out, dict) else {},
            directive=human_directive if isinstance(human_directive, dict) else directive if isinstance(directive, dict) else None,
            candidate_dir=candidate_dir,
            expected_files=expected_files,
        )
        if accepted_by_deferred_evidence:
            rec.log("qa_deferred_evidence_override_applied", ts=_utc_now_iso(), task_id=task_id, engineer_id=engineer_id, attempt=attempt_n)

        rec.save_json(f"qa/{task_id}__attempt_{attempt_n}__{engineer_id}.json", qa_out)
        rec.log("qa_end", task_id=task_id, engineer_id=engineer_id, attempt=attempt_n, ts=_utc_now_iso(), response_id=prev_qa)
        trace_event("qa_review_end", task_id=task_id, engineer_id=engineer_id, attempt=attempt_n, response_id=prev_qa, queue_command=(qa_out.get("queue_update") or {}).get("command") if isinstance(qa_out, dict) and isinstance(qa_out.get("queue_update"), dict) else None)

        # Resource gating (engineer/qa can request user-provided resources)
        reqs = _extract_resource_requests(eng_out) + _extract_resource_requests(qa_out)
        reqs = _filter_resource_requests(reqs, directive if isinstance(directive, dict) else None)
        if reqs:
            user_facing = _build_user_facing_request_packet(stage="engineer", agent_name=f"Engineer:{engineer_id}", task_id=task_id, items=reqs, kind="resource")
            req_context = {
                "escalation_chain": [f"Engineer:{engineer_id}", user_facing["domain_lead_review"]["reviewed_by"], "Team Lead"],
                "domain_lead_review": user_facing.get("domain_lead_review"),
                "team_lead_review": user_facing.get("team_lead_review"),
                "raw_request_hidden_from_user_by_default": True,
            }
            req_id = res_eval.submit(agent=f"Engineer:{engineer_id}", stage="engineer", task_id=task_id, items=reqs, context=req_context, user_facing=user_facing)
            rec.log("resource_requested", ts=_utc_now_iso(), stage="engineer", task_id=task_id, request_id=req_id, count=len(reqs), domain_lead=user_facing["domain_lead_review"]["reviewed_by"])

            cmd = {
                "command": "submit_result",
                "result": {
                    "task_id": task_id,
                    "engineer_id": engineer_id,
                    "changes": eng_out.get("changes"),
                    "code_output": eng_out.get("code_output"),
                    "staged_write_report": eng_out.get("staged_write_report"),
                    "local_write_report": eng_out.get("local_write_report"),
                    "verification_run": [f"blocked: needs_user_resources request_id={req_id}"],
                    "notes": f"needs_user_resources request_id={req_id}",
                    "handoff_interfaces": eng_out.get("handoff_interfaces", []),
                },
                "mark_done": False,
                "mark_blocked": True,
                "blocked_reason": "needs_user_resources",
            }
            rec.save_json(f"orch_submissions/{task_id}.json", cmd)
            orch.run(agent_input=cmd)
            rec.log("task_blocked", ts=_utc_now_iso(), task_id=task_id, engineer_id=engineer_id, attempt=attempt_n, reason="needs_user_resources")
            if run_state is not None:
                run_state.mark_blocked(
                    node=NODE_ENGINEERING,
                    block_type="resource",
                    reason="needs_user_resources",
                    payload={
                        "task_id": task_id,
                        "engineer_id": engineer_id,
                        "attempt": attempt_n,
                        "request_id": req_id,
                        "stage": "engineer",
                        "agent": f"Engineer:{engineer_id}",
                        "items": reqs,
                        "context": req_context,
                        "user_facing": user_facing,
                    },
                )
            continue

        # Decide status
        workflow_decision: Optional[Dict[str, Any]] = None
        q = qa_out.get("queue_update") if isinstance(qa_out, dict) else None
        q_cmd = (q.get("command") if isinstance(q, dict) else None) or ""
        q_cmd = str(q_cmd)

        passed = bool(qa_out.get("passed", False)) if isinstance(qa_out, dict) else False

        if q_cmd == "mark_done":
            mark_done = True
            mark_blocked = False
            blocked_reason = ""
        elif q_cmd == "mark_blocked":
            original_blocked_reason = _as_str((q.get("blocked_reason") if isinstance(q, dict) else "") or (qa_out.get("blocked_reason", "") if isinstance(qa_out, dict) else ""))
            effective_max_attempts_for_qa = _effective_max_attempts(max_attempts_per_task, attempt_n, human_directive)
            can_rerun = attempt_n < effective_max_attempts_for_qa
            workflow_decision = _coordinator_workflow_decision(
                rec=rec,
                abnormal_state="qa_mark_blocked",
                context_pack={
                    "run_id": shared_context.get("run_id", ""),
                    "task_id": task_id,
                    "engineer_id": engineer_id,
                    "work_item": wi,
                    "engineer_output_summary": {
                        "changes": eng_out.get("changes"),
                        "code_output_present": bool(eng_out.get("code_output")),
                        "staged_write_report": eng_out.get("staged_write_report"),
                        "local_write_report": eng_out.get("local_write_report"),
                        "questions": eng_out.get("questions", []),
                    },
                    "qa_output": qa_out,
                    "qa_blocked_reason": original_blocked_reason,
                    "max_attempt_check": {
                        "current_attempt": attempt_n,
                        "max_attempts": effective_max_attempts_for_qa,
                        "base_max_attempts": max_attempts_per_task,
                        "allowed": can_rerun,
                    },
                    "normal_policy": "If the issue is fixable by Engineer and attempts remain, rerun. If it needs user input or scope clarification, ask_user. Otherwise block.",
                },
                allowed_actions=["rerun_last_step", "rerun_specific_stage", "ask_user", "block"] if can_rerun else ["ask_user", "block"],
                default_action="block",
            )
            if run_state is not None:
                run_state.set_last_decision(workflow_decision)
            action = str(workflow_decision.get("action") or "block")
            if action in {"rerun_last_step", "rerun_specific_stage"} and can_rerun:
                mark_done = False
                mark_blocked = False
                blocked_reason = ""
            else:
                mark_done = False
                mark_blocked = True
                blocked_reason = original_blocked_reason or "qa_mark_blocked"
                if action == "ask_user":
                    blocked_reason = "team_lead_requires_user_input"
                    human_request_id = _submit_human_request(
                        human_desk=human_desk,
                        rec=rec,
                        task_id=task_id,
                        stage="engineering.qa_blocked",
                        reason=blocked_reason,
                        question=str(workflow_decision.get("required_input") or original_blocked_reason or "QA blocked this task. Provide a concrete correction or decide whether to block/continue."),
                        options=["rerun with my clarification", "reduce scope", "accept limitation and continue", "block requested item / route around", "hard stop whole task"],
                        context={"workflow_decision": workflow_decision, "work_item": wi, "qa_output": qa_out, "engineer_id": engineer_id, "attempt": attempt_n},
                        run_state=run_state,
                    )
        elif q_cmd == "return_to_queue":
            mark_done = False
            mark_blocked = False
            blocked_reason = ""
        else:
            # legacy QA shape
            mark_done = passed
            mark_blocked = bool(qa_out.get("blocked", False)) and not passed if isinstance(qa_out, dict) else False
            blocked_reason = _as_str(qa_out.get("blocked_reason", "")) if (mark_blocked and isinstance(qa_out, dict)) else ""

        # Deterministic override: out-of-scope file writes can never be marked done.
        if mark_done and staged_write_report.get("status") == "scope_error":
            mark_done = False
            mark_blocked = False
            blocked_reason = ""
            qa_out = dict(qa_out or {})
            qa_out["queue_update"] = {"command": "return_to_queue", "blocked_reason": "path_outside_task_scope"}
            qa_out["notes"] = (str(qa_out.get("notes") or "") + "\nOperation override: code_output attempted to write outside files_expected scope.").strip()

        # Durable QA-to-Engineer bridge: whenever QA/operation does not mark the
        # task done, save a Repair Packet so the next attempt receives concrete
        # evidence, implicated files, required fixes, and verification commands.
        if not mark_done:
            try:
                repair_packet_out = _build_repair_packet(
                    task_id=task_id,
                    attempt_n=attempt_n,
                    engineer_id=engineer_id,
                    work_item=wi,
                    eng_out=eng_out if isinstance(eng_out, dict) else {},
                    qa_out=qa_out if isinstance(qa_out, dict) else {},
                    staged_write_report=staged_write_report if isinstance(staged_write_report, dict) else {},
                    expected_files=expected_files,
                    workflow_decision=workflow_decision,
                    user_directive=human_directive if isinstance(human_directive, dict) else directive if isinstance(directive, dict) else None,
                )
                _save_repair_packet(rec, repair_packet_out)
                rec.log("repair_packet_saved", ts=_utc_now_iso(), task_id=task_id, attempt=attempt_n, engineer_id=engineer_id)
            except Exception as exc:
                rec.log("repair_packet_save_failed", ts=_utc_now_iso(), task_id=task_id, attempt=attempt_n, error=str(exc))

        final_write_report = None
        if mark_done:
            final_write_report = _promote_candidate_workspace(
                candidate_dir=candidate_dir,
                workspace_dir=rec.out_dir / "workspace",
                task_id=task_id,
                engineer_id=engineer_id,
            )
            eng_out["local_write_report"] = final_write_report
            rec.save_json(f"file_writes/{task_id}__attempt_{attempt_n}__{engineer_id}.json", final_write_report)
            rec.log(
                "local_file_write_promoted",
                task_id=task_id,
                engineer_id=engineer_id,
                attempt=attempt_n,
                status=final_write_report.get("status"),
                files_written=len(final_write_report.get("files_written", [])),
                files_skipped=len(final_write_report.get("files_skipped", [])),
            )
            if final_write_report.get("status") in {"error", "scope_error", "partial"}:
                mark_done = False
                mark_blocked = False
                blocked_reason = ""
                qa_out = dict(qa_out or {})
                qa_out["queue_update"] = {"command": "return_to_queue", "blocked_reason": "final_write_failed"}
                qa_out["notes"] = (str(qa_out.get("notes") or "") + "\nOperation override: final workspace promotion failed.").strip()

        if mark_done:
            try:
                (rec.out_dir / "retry_contexts" / f"{task_id}.json").unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                pass
        elif not mark_blocked:
            review = qa_out.get("review") if isinstance(qa_out, dict) else {}
            issues = review.get("issues") if isinstance(review, dict) else []
            required_fixes = []
            if isinstance(issues, list):
                for issue in issues:
                    if isinstance(issue, dict):
                        title = str(issue.get("title") or "").strip()
                        action = str(issue.get("required_action") or issue.get("detail") or "").strip()
                        if title or action:
                            required_fixes.append((title + ": " + action).strip(": "))
            if not required_fixes:
                required_fixes.append(str(qa_out.get("notes") or "QA returned the task to the queue; revise the previous attempt."))
            retry_payload = {
                "task_id": task_id,
                "created_at_utc": _utc_now_iso(),
                "attempt": attempt_n,
                "previous_engineer_output": eng_out,
                "qa_review": qa_out,
                "required_fixes": required_fixes,
                "failed_verification": _normalize_verification_run(qa_out),
                "staged_write_report": staged_write_report,
                "feedback_for_engineer": "\n".join(required_fixes),
            }
            rec.save_json(f"retry_contexts/{task_id}.json", retry_payload)

        cmd = {
            "command": "submit_result",
            "result": {
                "task_id": task_id,
                "engineer_id": engineer_id,
                "status": "done" if mark_done else ("blocked" if mark_blocked else "needs_fix"),
                "changes": eng_out.get("changes"),
                "code_output": eng_out.get("code_output"),
                "staged_write_report": eng_out.get("staged_write_report"),
                "local_write_report": eng_out.get("local_write_report"),
                "final_write_report": final_write_report,
                "workflow_decision": workflow_decision,
                "verification_run": _normalize_verification_run(qa_out),
                "notes": _as_str(eng_out.get("notes", "")),
                "handoff_interfaces": eng_out.get("handoff_interfaces", []),
            },
            "mark_done": mark_done,
            "mark_blocked": mark_blocked,
            "blocked_reason": blocked_reason,
        }
        rec.save_json(f"orch_submissions/{task_id}.json", cmd)
        orch.run(agent_input=cmd)
        rec.log("task_submitted", task_id=task_id, engineer_id=engineer_id, attempt=attempt_n, status=cmd["result"]["status"])
        trace_event("task_submitted", task_id=task_id, engineer_id=engineer_id, attempt=attempt_n, status=cmd["result"]["status"], mark_done=mark_done, mark_blocked=mark_blocked, blocked_reason=blocked_reason, workflow_decision_action=(workflow_decision or {}).get("action") if isinstance(workflow_decision, dict) else None)


# -------------------------
# Main
# -------------------------


def main() -> None:
    _ensure_openai_key()

    # Auto-resume the most recent incomplete run unless FORCE_NEW_RUN is set.
    resume_dir: Optional[Path] = None
    if str(os.getenv("FORCE_NEW_RUN", "")).strip().lower() not in {"1", "true", "yes"}:
        resume_dir = _find_latest_incomplete_run(outputs_dir="outputs")

    if resume_dir is not None:
        rec = _make_recorder_for_run(resume_dir, logs_dir="logs")
        rec.log("run_resuming", ts=_utc_now_iso())
    else:
        rec = Recorder.new()
        rec.log("run_started", ts=_utc_now_iso())

    # Enable lightweight trace observability for modules that do not receive Recorder directly.
    os.environ["ASCENDANT_RUN_ID"] = rec.run_id
    os.environ["ASCENDANT_TRACE_PATH"] = str(rec.out_dir / "trace.jsonl")
    trace_event(
        "run_observability_initialized",
        output_dir=str(rec.out_dir),
        log_path=str(rec.log_path),
        resumed=bool(resume_dir is not None),
        run_mode=os.getenv("ASCENDANT_RUN_MODE", "development"),
    )

    resource_dir = str(rec.out_dir / "resources")
    run_state = RunState(rec.out_dir, run_id=rec.run_id)
    run_state.initialize(
        run_mode=os.getenv("ASCENDANT_RUN_MODE", "development"),
        pointers={
            "output_dir": str(rec.out_dir),
            "log_path": str(rec.log_path),
            "trace_path": str(rec.out_dir / "trace.jsonl"),
            "resource_dir": resource_dir,
        },
    )
    trace_event("run_state_initialized", path=str(run_state.path), current_node=run_state.state.get("current_node"), resume_from=run_state.state.get("resume_from"))

    res_eval = ResourceEval(resource_dir)
    human_desk = HumanRequestDesk(resource_dir)
    _quarantine_or_retarget_mismatched_directives(resource_dir=resource_dir, rec=rec)
    _auto_resolve_nonblocking_resource_requests(res_eval, run_state, rec)

    # Fail fast on provider/runtime setup before long-running agent work.
    preflight_report = run_preflight(
        output_dir=rec.out_dir,
        strict=os.getenv("ASCENDANT_PREFLIGHT_STRICT", "0").strip().lower() in {"1", "true", "yes", "on"},
    )
    rec.save_json("preflight_report.json", preflight_report)
    run_state.set_artifact("preflight_report", str(rec.out_dir / "preflight_report.json"))
    if not preflight_report.get("ok", False):
        run_state.mark_blocked(
            node=None,
            block_type="preflight",
            reason="preflight_failed",
            payload=preflight_report,
        )
        rec.log("preflight_failed", ts=_utc_now_iso(), error_count=preflight_report.get("error_count"), warning_count=preflight_report.get("warning_count"))
        _wait_for_external_unblock(run_state, rec)
        return
    rec.log("preflight_passed", ts=_utc_now_iso(), warning_count=preflight_report.get("warning_count"))

    # ------------------------
    # 1) Intake (only if needed)
    # ------------------------
    initial_input_path = rec.out_dir / "initial_input.json"
    initial_input: Optional[Dict[str, Any]] = _read_json_best_effort(initial_input_path)

    if isinstance(initial_input, dict) and initial_input:
        # Resume/dashboard mode (intake already done)
        try:
            supervise_ui(
                agent=None,
                agent_input={},
                title="WORKSPACE — Ascendant Path",
                description_md=(
                    "Keep this tab open. It shows status/logs/artifacts and the Resources Desk."
                ),
                max_rounds=0,
                session_id=f"workspace_{rec.run_id}",
                log_file=str(rec.log_path),
                port=int(os.getenv("WORKSPACE_UI_PORT", os.getenv("INTAKE_UI_PORT", "7860"))),
                open_browser=True,
                keep_open=True,
                enable_resource_desk=True,
                resource_dir=resource_dir,
                run_id=rec.run_id,
                allow_edit_agent_input=False,
                block_until_approved=False,
                watch_output_dir=str(rec.out_dir),
            )
        except Exception:
            pass

    if not isinstance(initial_input, dict) or not initial_input:
        seed_initial = _read_json_best_effort(Path("initial_input.json")) or {}

        def _intake_ready(out: Dict[str, Any]) -> bool:
            return (
                isinstance(out, dict)
                and out.get("mode") == "FINAL"
                and isinstance(out.get("initial_input"), dict)
            )

        def _intake_assistant(out: Dict[str, Any]) -> Optional[str]:
            if not isinstance(out, dict):
                return "Describe what you want to build."
            if _intake_ready(out):
                return "Draft looks complete. Review the JSON and click **Approve** to start the pipeline."
            if out.get("mode") == "ASK":
                qs = out.get("questions") or []
                if isinstance(qs, list) and qs:
                    return "Answer these:\n" + "\n".join([f"{i+1}. {q}" for i, q in enumerate(qs)])
                return "Add more details so I can finalize initial_input."
            return "Draft is not approvable yet. Check Latest Output and answer any remaining questions, then click Run / Generate again."

        team_lead = TeamLeadAgent()
        run_state.mark_node_started(NODE_INTAKE)
        rec.log("team_lead_intake_start", ts=_utc_now_iso())

        sup = supervise_ui(
            agent=team_lead,
            agent_input={"user_message": "", "seed_initial_input": seed_initial},
            title="TEAM LEAD — Intake & Run Control",
            description_md=("Team Lead owns intake and user communication. Iterate until **Initial Input** is complete enough to send through the existing workflow: PM → UX → Engineering Lead → Engineering/QA.\n\n"
                           "Use **Run / Generate** to iterate, and **Approve** when ready."),
            max_rounds=int(os.getenv("INTAKE_MAX_ROUNDS", "12")),
            session_id=f"intake_{rec.run_id}",
            log_file=str(rec.log_path),
            port=int(os.getenv("INTAKE_UI_PORT", "7860")),
            open_browser=True,
            keep_open=True,
            enable_resource_desk=True,
            resource_dir=resource_dir,
            run_id=rec.run_id,
            allow_edit_agent_input=False,
            approvable=_intake_ready,
            assistant_message=_intake_assistant,
            output_transform=lambda out: (
                {"initial_input": out.get("initial_input")}
                if isinstance(out, dict) and out.get("mode") == "FINAL" and isinstance(out.get("initial_input"), dict)
                else {}
            ),
            block_until_approved=True,
            watch_output_dir=str(rec.out_dir),
        )
        rec.log("team_lead_intake_end", ts=_utc_now_iso())

        intake_out = sup.final_output
        if not isinstance(intake_out, dict) or not isinstance(intake_out.get("initial_input"), dict):
            raise RuntimeError("Intake did not produce final initial_input dict.")

        initial_input = intake_out["initial_input"]
        _write_pretty_json(Path("initial_input.json"), initial_input)
        rec.save_json("initial_input.json", initial_input)
        run_state.mark_node_completed(NODE_INTAKE, artifact_name="initial_input", artifact_path=str(rec.out_dir / "initial_input.json"))

    if not isinstance(initial_input, dict) or not initial_input:
        raise RuntimeError("Missing initial_input.json; cannot proceed.")

    run_state.mark_node_completed(NODE_INTAKE, artifact_name="initial_input", artifact_path=str(initial_input_path if initial_input_path.exists() else Path("initial_input.json")))

    # Shared context (available to all downstream stages/workers)
    shared_context: Dict[str, Any] = {
        "run_id": rec.run_id,
        "outputs_dir": str(rec.out_dir),
        "initial_input": initial_input,
        "asset_manifest": res_eval.read_manifest(),
        "team_lead_directives": _read_team_lead_directives(resource_dir),
    }

    # Helper: run a stage with pause/retry on quota issues and optional cache hit.
    def _run_or_load_stage_json(
        *,
        cache_relpath: str,
        stage_name: str,
        fn,
        node_name: Optional[str] = None,
        artifact_name: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        cached = _read_json_best_effort(rec.out_dir / cache_relpath)
        if isinstance(cached, dict) and cached:
            rec.log(f"{stage_name}_cache_hit", ts=_utc_now_iso(), path=cache_relpath)
            trace_event("stage_cache_hit", stage=stage_name, path=cache_relpath)
            if node_name:
                run_state.mark_node_skipped(node_name, reason="cached_artifact_present")
            if artifact_name:
                run_state.set_artifact(artifact_name, str(rec.out_dir / cache_relpath))
            return cached, None

        while True:
            _maybe_wait_if_paused(rec)
            try:
                if node_name:
                    run_state.mark_node_started(node_name)
                trace_event("stage_execution_start", stage=stage_name, cache_path=cache_relpath)
                out, resp_id = fn()
                if isinstance(out, dict):
                    rec.save_json(cache_relpath, out)
                    if artifact_name:
                        run_state.set_artifact(artifact_name, str(rec.out_dir / cache_relpath))
                if node_name:
                    run_state.mark_node_completed(node_name, artifact_name=artifact_name, artifact_path=str(rec.out_dir / cache_relpath) if artifact_name else None)
                trace_event("stage_execution_end", stage=stage_name, cache_path=cache_relpath, response_id=resp_id, output_is_dict=isinstance(out, dict))
                return out if isinstance(out, dict) else {"output": out}, resp_id
            except Exception as e:
                if _looks_like_quota_or_rate_limit_error(e):
                    _pause_run(rec, reason="openai_quota_or_rate_limit", err=e)
                    _wait_for_resume(rec)
                    continue
                raise

    # ------------------------
    # 2) PM
    # ------------------------
    brief = _as_str(initial_input.get("brief", ""))
    if not brief:
        raise RuntimeError("initial_input missing 'brief'.")

    pm = PMAgent()
    rec.log("pm_start", ts=_utc_now_iso())

    def _pm_fn():
        return _run_stage_with_resources(
            agent=pm,
            agent_name="PM",
            stage="pm",
            agent_input={
                "brief": brief,
                "initial_input": initial_input,
                "asset_manifest": shared_context.get("asset_manifest"),
                "resource_decision": shared_context.get("resource_decision"),
                "team_lead_directives": _read_team_lead_directives(resource_dir, target_agents=["PM Agent", "Team Lead / Intake"]),
            },
            res_eval=res_eval,
            rec=rec,
            shared_context=shared_context,
            run_state=run_state,
        )

    pm_out, pm_resp = _run_or_load_stage_json(cache_relpath="pm_output.json", stage_name="pm", fn=_pm_fn, node_name=NODE_PM, artifact_name="pm_output")
    rec.log("pm_end", ts=_utc_now_iso(), response_id=pm_resp)

    # ------------------------
    # 3) UX
    # ------------------------
    ux = UXDesignerAgent()
    rec.log("ux_start", ts=_utc_now_iso())

    def _ux_fn():
        return _run_stage_with_resources(
            agent=ux,
            agent_name="UX",
            stage="ux",
            agent_input={
                "design_brief": {"brief": brief, "initial_input": initial_input, "pm_output": pm_out},
                "repo_context": {
                    "asset_manifest": shared_context.get("asset_manifest"),
                    "resource_decision": shared_context.get("resource_decision"),
                },
                "asset_manifest": shared_context.get("asset_manifest"),
                "resource_decision": shared_context.get("resource_decision"),
                "team_lead_directives": _read_team_lead_directives(resource_dir, target_agents=["UX Agent", "Team Lead / Intake"]),
            },
            res_eval=res_eval,
            rec=rec,
            shared_context=shared_context,
            run_state=run_state,
        )

    ux_out, ux_resp = _run_or_load_stage_json(cache_relpath="ux_output.json", stage_name="ux", fn=_ux_fn, node_name=NODE_UX, artifact_name="ux_output")
    rec.log("ux_end", ts=_utc_now_iso(), response_id=ux_resp)

    # ------------------------
    # 4) Eng Lead
    # ------------------------
    eng_lead = EngineeringLeadAgent()
    rec.log("eng_lead_start", ts=_utc_now_iso())

    def _eng_lead_fn():
        return _run_stage_with_resources(
            agent=eng_lead,
            agent_name="EngLead",
            stage="eng_lead",
            agent_input={
                "task": "Produce an engineering execution plan and concrete tasks for engineers based on initial_input + PM + UX outputs.",
                "upstream": {
                    "initial_input": initial_input,
                    "pm_output": pm_out,
                    "ux_output": ux_out,
                    "asset_manifest": shared_context.get("asset_manifest"),
                    "resource_decision": shared_context.get("resource_decision"),
                },
                "asset_manifest": shared_context.get("asset_manifest"),
                "resource_decision": shared_context.get("resource_decision"),
                "team_lead_directives": _read_team_lead_directives(resource_dir, target_agents=["Engineering Lead", "Orchestrator", "Team Lead / Intake"]),
            },
            res_eval=res_eval,
            rec=rec,
            shared_context=shared_context,
            run_state=run_state,
        )

    eng_lead_out, eng_lead_resp = _run_or_load_stage_json(cache_relpath="eng_lead_output.json", stage_name="eng_lead", fn=_eng_lead_fn, node_name=NODE_ENG_LEAD, artifact_name="eng_lead_output")
    rec.log("eng_lead_end", ts=_utc_now_iso(), response_id=eng_lead_resp)

    eng_lead_out, mvp_gate_report = _enforce_mvp_engineering_plan(initial_input, eng_lead_out)
    if mvp_gate_report.get("applied"):
        rec.save_json("eng_lead_output.json", eng_lead_out)
        rec.save_json("mvp_scope_gate_report.json", mvp_gate_report)
        run_state.set_artifact("mvp_scope_gate_report", str(rec.out_dir / "mvp_scope_gate_report.json"))
        rec.log("mvp_scope_gate_applied", ts=_utc_now_iso(), **mvp_gate_report)

    shared_context.update({"pm_output": pm_out, "ux_output": ux_out, "eng_lead_output": eng_lead_out})

    # ------------------------
    # 5) Orchestrator ingest plan (and replay previous submissions if resuming)
    # ------------------------
    task_graph_report = validate_task_graph(eng_lead_out)
    rec.save_json("task_graph_validation.json", task_graph_report)
    run_state.set_artifact("task_graph_validation", str(rec.out_dir / "task_graph_validation.json"))
    if not task_graph_report.get("ok", False):
        rec.log("task_graph_validation_failed_before_repair", ts=_utc_now_iso(), error_count=task_graph_report.get("error_count"), warning_count=task_graph_report.get("warning_count"))
        repair_result = repair_task_graph(eng_lead_out)
        rec.save_json("task_graph_repair_report.json", repair_result)
        run_state.set_artifact("task_graph_repair_report", str(rec.out_dir / "task_graph_repair_report.json"))
        if repair_result.get("ok") and isinstance(repair_result.get("eng_lead_output"), dict):
            eng_lead_out = repair_result["eng_lead_output"]
            shared_context["eng_lead_output"] = eng_lead_out
            rec.save_json("eng_lead_output.json", eng_lead_out)
            task_graph_report = repair_result.get("validation_report") or validate_task_graph(eng_lead_out)
            rec.save_json("task_graph_validation.json", task_graph_report)
            run_state.clear_block(event="task_graph.auto_repaired")
            rec.log("task_graph_auto_repaired", ts=_utc_now_iso(), change_count=len(repair_result.get("changes") or []), warning_count=task_graph_report.get("warning_count"))
        else:
            payload = repair_result.get("validation_report") if isinstance(repair_result, dict) else task_graph_report
            run_state.mark_blocked(node=NODE_ORCHESTRATOR_INGEST, block_type="task_graph_validation", reason="invalid_engineering_task_graph", payload=payload)
            rec.log("task_graph_validation_failed", ts=_utc_now_iso(), error_count=(payload or {}).get("error_count"), warning_count=(payload or {}).get("warning_count"))
            _wait_for_external_unblock(run_state, rec)
            return
    rec.log("task_graph_validation_passed", ts=_utc_now_iso(), warning_count=task_graph_report.get("warning_count"))

    orch_state_path = rec.out_dir / "orchestrator_state.json"
    orch = EngineeringOrchestratorAgent(state_path=str(orch_state_path))
    run_state.set_artifact("orchestrator_state", str(orch_state_path))
    run_state.mark_node_started(NODE_ORCHESTRATOR_INGEST)
    rec.log("orch_ingest_start", ts=_utc_now_iso())
    orch.run(agent_input={
        "command": "ingest_plan",
        "eng_lead_output": eng_lead_out,
        "preserve_existing_state": bool(orch_state_path.exists()),
    })
    rec.log("orch_ingest_end", ts=_utc_now_iso())
    run_state.mark_node_completed(NODE_ORCHESTRATOR_INGEST)

    # Replay previously submitted task results so resume works after restarts.
    subs_dir = rec.out_dir / "orch_submissions"
    if subs_dir.exists():
        for p in sorted(subs_dir.glob("*.json")):
            try:
                cmd = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(cmd, dict):
                    orch.run(agent_input=cmd)
            except Exception:
                continue
        rec.log("orch_replay_done", ts=_utc_now_iso())

    _repair_done_task_workspace_from_attempts(rec=rec, orch=orch)
    _repair_attempted_task_workspace_from_sources(rec=rec, orch=orch)
    _auto_accept_deferred_candidate_tasks(rec=rec, orch=orch, resource_dir=resource_dir, run_state=run_state)
    _auto_unblock_deferred_verification_tasks(rec=rec, orch=orch, resource_dir=resource_dir, run_state=run_state)

    # Resource monitor: unblocks tasks once you resolve requests in Resources Desk
    resource_monitor = threading.Thread(
        target=_resource_monitor_loop,
        kwargs=dict(orch=orch, res_eval=res_eval, rec=rec, shared_context=shared_context, poll_seconds=1.0, run_state=run_state),
        daemon=True,
    )
    resource_monitor.start()
    human_monitor = threading.Thread(
        target=_human_monitor_loop,
        kwargs=dict(orch=orch, human_desk=human_desk, rec=rec, poll_seconds=1.0, run_state=run_state),
        daemon=True,
    )
    human_monitor.start()

    # ------------------------
    # 6) Engineers + QA
    # ------------------------
    num_engineers = int(os.getenv("NUM_ENGINEERS", "3"))
    max_attempts_per_task = int(os.getenv("MAX_ATTEMPTS_PER_TASK", "3"))

    qa = QAAgent()
    attempts = _Attempts(run_state.state.get("task_attempts") if isinstance(run_state.state, dict) else None)
    response_state = _ResponseState()
    todo_only_no_progress_loops = 0

    while True:
        run_state.mark_node_started(NODE_ENGINEERING, payload={"num_engineers": num_engineers, "max_attempts_per_task": max_attempts_per_task})
        rec.log("engineers_start", ts=_utc_now_iso(), num_engineers=num_engineers)

        threads: List[threading.Thread] = []
        for i in range(num_engineers):
            tid = f"eng_{i+1}"
            t = threading.Thread(
                target=_worker_loop,
                kwargs=dict(
                    engineer_id=tid,
                    orch=orch,
                    qa=qa,
                    rec=rec,
                    shared_context=shared_context,
                    attempts=attempts,
                    response_state=response_state,
                    max_attempts_per_task=max_attempts_per_task,
                    resource_dir=resource_dir,
                    human_desk=human_desk,
                    run_state=run_state,
                ),
                daemon=True,
            )
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        engineering_state = orch.run(agent_input={"command": "get_state"})
        engineering_summary = engineering_state.get("summary", {}) if isinstance(engineering_state, dict) else {}
        counts = engineering_summary.get("counts", {}) if isinstance(engineering_summary, dict) else {}
        unfinished = int(counts.get("todo", 0) or 0) + int(counts.get("claimed", 0) or 0) + int(counts.get("blocked", 0) or 0)
        rec.log("engineers_end", ts=_utc_now_iso(), summary=engineering_summary)

        # If all worker threads have joined but the queue still contains claimed
        # tasks, those claims are orphaned. This happened when an abnormal
        # max-attempt submission crashed before submit_result cleared the claim.
        # Recover immediately instead of blocking until claim TTL or requiring a
        # manual state edit.
        todo_count = int(counts.get("todo", 0) or 0)
        claimed_count = int(counts.get("claimed", 0) or 0)
        blocked_count = int(counts.get("blocked", 0) or 0)
        if claimed_count > 0:
            try:
                recovered = orch.run(agent_input={
                    "command": "recover_orphaned_claims",
                    "note": "workers_joined_with_no_active_worker",
                })
            except Exception as exc:
                recovered = {"ok": False, "error": str(exc), "recovered_count": 0, "recovered": []}
            if int(recovered.get("recovered_count", 0) or 0) > 0:
                rec.log(
                    "resume_recovered_stale_claim",
                    ts=_utc_now_iso(),
                    recovered_count=recovered.get("recovered_count"),
                    recovered=recovered.get("recovered"),
                    reason="engineer_threads_joined_but_queue_had_claimed_tasks",
                )
                continue

        if todo_count > 0 and claimed_count == 0 and blocked_count == 0:
            todo_only_no_progress_loops += 1
            rec.log(
                "engineering_todo_available_after_workers_joined",
                ts=_utc_now_iso(),
                summary=engineering_summary,
                recovery_loop=todo_only_no_progress_loops,
            )
            trace_event("engineering_todo_available_not_blocked", summary=engineering_summary, recovery_loop=todo_only_no_progress_loops)
            if todo_only_no_progress_loops <= 3:
                run_state.clear_block(event="engineering.todo_available_continue")
                continue
            # After repeated no-progress loops, block with a precise scheduler
            # reason instead of the misleading generic engineering_incomplete.
            run_state.mark_blocked(
                node=NODE_ENGINEERING,
                block_type="engineering_scheduler",
                reason="todo_tasks_available_but_workers_made_no_progress",
                payload={"summary": engineering_summary, "state": engineering_state},
            )
            rec.save_json("ENGINEERING_BLOCKED.json", {"summary": engineering_summary, "state": engineering_state})
            trace_event("engineering_scheduler_no_progress_blocked", summary=engineering_summary)
            _wait_for_external_unblock(run_state, rec)
            post_state = _read_json_best_effort(run_state.path) or run_state.snapshot()
            if post_state.get("done") or post_state.get("blocked"):
                return
            rec.log("engineering_resuming_after_scheduler_unblock", ts=_utc_now_iso(), summary=engineering_summary)
            todo_only_no_progress_loops = 0
            continue
        else:
            todo_only_no_progress_loops = 0

        if unfinished:
            # Preserve a precise user-facing human/resource block if a worker
            # already wrote one. The old generic engineering_queue block erased
            # request_id/question/items from run_state and left the UI unable to
            # show the actionable request.
            existing_state = _read_json_best_effort(run_state.path) or run_state.snapshot()
            existing_block_type = str(existing_state.get("block_type") or "") if isinstance(existing_state, dict) else ""
            preserve_precise_block = bool(
                isinstance(existing_state, dict)
                and existing_state.get("blocked")
                and existing_block_type in {"resource", "human_input", "human_decision", "decision"}
            )
            if not preserve_precise_block:
                run_state.mark_blocked(
                    node=NODE_ENGINEERING,
                    block_type="engineering_queue",
                    reason="engineering_incomplete",
                    payload={"summary": engineering_summary, "blocked": engineering_state.get("runtime", {}) if isinstance(engineering_state, dict) else {}},
                )
            else:
                rec.log(
                    "engineering_preserved_precise_block",
                    ts=_utc_now_iso(),
                    block_type=existing_block_type,
                    summary=engineering_summary,
                )
            rec.save_json("ENGINEERING_BLOCKED.json", {"summary": engineering_summary, "state": engineering_state})
            trace_event("engineering_blocked_before_runbook", summary=engineering_summary, preserved_precise_block=preserve_precise_block)
            _wait_for_external_unblock(run_state, rec)
            post_state = _read_json_best_effort(run_state.path) or run_state.snapshot()
            if post_state.get("done") or post_state.get("blocked"):
                return
            rec.log("engineering_resuming_after_unblock", ts=_utc_now_iso(), summary=engineering_summary)
            continue
        run_state.mark_node_completed(NODE_ENGINEERING)
        break

    # ------------------------
    # 6.4) Deterministic project executor gate
    # ------------------------
    run_state.mark_node_started(NODE_EXECUTOR)
    rec.log("project_executor_start", ts=_utc_now_iso())
    executor_report = run_project_executor(
        workspace_dir=rec.out_dir / "workspace",
        output_dir=rec.out_dir / "executor",
    )
    rec.save_json("executor_report.json", executor_report)
    run_state.set_artifact("executor_report", str(rec.out_dir / "executor_report.json"))
    run_state.set_artifact("executor_report_md", str(rec.out_dir / "executor" / "EXECUTOR_REPORT.md"))
    rec.log("project_executor_end", ts=_utc_now_iso(), status=executor_report.get("status"), ok=executor_report.get("ok"))
    if not executor_report.get("ok", False):
        run_state.mark_blocked(
            node=NODE_EXECUTOR,
            block_type="project_executor",
            reason="deterministic_project_checks_failed",
            payload=executor_report,
        )
        trace_event("project_executor_failed", status=executor_report.get("status"), summary=executor_report.get("summary"))
        _wait_for_external_unblock(run_state, rec)
        return
    run_state.mark_node_completed(NODE_EXECUTOR, artifact_name="executor_report", artifact_path=str(rec.out_dir / "executor_report.json"), payload={"status": executor_report.get("status"), "summary": executor_report.get("summary")})

    # ------------------------
    # 6.5) Eng Lead run instructions (post-build)
    # ------------------------
    try:
        run_state.mark_node_started(NODE_RUNBOOK)
        rec.log("run_instructions_start", ts=_utc_now_iso())
        runbook, _ = eng_lead.runbook(
            agent_input={
                "context_pack": {
                    "initial_input": initial_input,
                    "pm_output": pm_out,
                    "ux_output": ux_out,
                    "eng_lead_output": eng_lead_out,
                    "orchestrator_state": orch.run(agent_input={"command": "get_state"}),
                    "orchestrator_results": orch.run(agent_input={"command": "get_results"}),
                    "outputs_dir": str(rec.out_dir),
                }
            }
        )
        if isinstance(runbook, dict) and isinstance(runbook.get("run_instructions_md"), str):
            (rec.out_dir / "RUN_INSTRUCTIONS.md").write_text(runbook["run_instructions_md"], encoding="utf-8")
            rec.save_json("run_instructions.json", runbook)
            run_state.set_artifact("run_instructions_md", str(rec.out_dir / "RUN_INSTRUCTIONS.md"))
            run_state.set_artifact("run_instructions_json", str(rec.out_dir / "run_instructions.json"))
        run_state.mark_node_completed(NODE_RUNBOOK)
        rec.log("run_instructions_end", ts=_utc_now_iso())
    except Exception as e:
        run_state.mark_blocked(node=NODE_RUNBOOK, block_type="runbook_error", reason=str(e))
        rec.log("run_instructions_failed", ts=_utc_now_iso(), error=str(e))
        _wait_for_external_unblock(run_state, rec)
        return

    # ------------------------
    # 7) Coordinator summary
    # ------------------------
    coordinator = CoordinatorAgent()
    state = orch.run(agent_input={"command": "get_state"})
    results = orch.run(agent_input={"command": "get_results"})

    rec.log("coordinator_start", ts=_utc_now_iso())

    def _coord_fn():
        return _run_stage_with_resources(
            agent=coordinator,
            agent_name="Coordinator",
            stage="coordinator",
            agent_input={
                "objective": "Summarize the execution, outstanding risks, and next steps. Produce a coherent handoff for a human.",
                "context_pack": {
                    "initial_input": initial_input,
                    "pm_output": pm_out,
                    "ux_output": ux_out,
                    "eng_lead_output": eng_lead_out,
                    "orchestrator_state": state,
                    "orchestrator_results": results,
                    "outputs_dir": str(rec.out_dir),
                },
                "asset_manifest": shared_context.get("asset_manifest"),
            },
            res_eval=res_eval,
            rec=rec,
            shared_context=shared_context,
            run_state=run_state,
        )

    coord_out, coord_resp = _run_or_load_stage_json(cache_relpath="coordinator_output.json", stage_name="coordinator", fn=_coord_fn, node_name=NODE_COORDINATOR_FINAL_HANDOFF, artifact_name="coordinator_output")
    rec.log("coordinator_end", ts=_utc_now_iso(), response_id=coord_resp)

    # Mark done for UI / resume logic
    try:
        (rec.out_dir / "DONE.flag").write_text("done\n", encoding="utf-8")
    except Exception:
        pass

    run_state.mark_done()
    rec.log("run_finished", ts=_utc_now_iso())

    print(f"Run id: {rec.run_id}")
    print(f"Log: {rec.log_path}")
    print(f"Output dir: {rec.out_dir}")
    print("Wrote: initial_input.json")


if __name__ == "__main__":
    main()
