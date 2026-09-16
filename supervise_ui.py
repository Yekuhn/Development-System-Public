# supervise_ui.py
"""
Generic supervision UI for supervising any agent.

Design goals:
- Works with Gradio builds where Chatbot expects *messages* format
  (list of {"role": "...", "content": "..."} dicts) and where Chatbot does NOT
  accept the `type=` keyword argument.
- Keeps supervision logic generic: the caller controls what's shown and
  when approval is allowed via hook functions.

Agent contract (duck-typed):
    agent.run(
        agent_input=...,
        draft=...,
        feedback=...,
        previous_response_id=...
    ) -> (output_dict, response_id)  OR output_dict
"""

from __future__ import annotations

import json
import inspect
import re
import threading
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import gradio as gr

from resource_eval import ResourceEval
from human_requests import HumanRequestDesk


# -------------------------
# UI styling (modal-like approval panel)
# -------------------------
_APPROVE_MODAL_CSS = r"""
#approve_backdrop {
  display: none !important;
  pointer-events: none !important;
}

/* Compact, regular-sized panels. Large content scrolls inside the box instead of expanding the page. */
#approve_modal {
  width: 100%;
  max-width: 520px;
  min-height: 0 !important;
  max-height: 360px;
  overflow: auto !important;
  padding: 10px 12px;
  border-radius: 12px;
  border: 1px solid rgba(128,128,128,0.35);
  background: var(--background-fill-primary, #fff);
  margin: 0 0 12px 0;
}
#run_block_banner {
  max-height: 160px;
  overflow: auto !important;
  padding: 10px 12px;
  border-radius: 10px;
  border: 1px solid rgba(255, 180, 0, 0.55);
  background: rgba(255, 180, 0, 0.12);
  margin: 8px 0 12px 0;
}
#resource_desk_banner {
  max-height: 140px;
  overflow: auto !important;
  padding: 8px 10px;
  border-radius: 10px;
  border: 1px solid rgba(128,128,128,0.35);
  margin: 6px 0 10px 0;
}
.compact-scroll,
.compact-scroll > div,
.compact-scroll .block,
.compact-scroll .wrap,
.compact-scroll .contain {
  min-height: 0 !important;
  max-height: 360px !important;
  overflow: auto !important;
}
.compact-scroll textarea,
.compact-scroll pre,
.compact-scroll code,
.compact-scroll .cm-editor,
.compact-scroll .cm-scroller,
.compact-scroll .prose,
.compact-scroll .markdown-body {
  max-height: 320px !important;
  overflow: auto !important;
  white-space: pre !important;
  overflow-wrap: normal !important;
}
.compact-scroll .cm-content {
  white-space: pre !important;
}
.compact-scroll-small,
.compact-scroll-small > div,
.compact-scroll-small .block,
.compact-scroll-small .wrap,
.compact-scroll-small .contain {
  min-height: 0 !important;
  max-height: 220px !important;
  overflow: auto !important;
}
.compact-scroll-small textarea,
.compact-scroll-small pre,
.compact-scroll-small code,
.compact-scroll-small .cm-editor,
.compact-scroll-small .cm-scroller {
  max-height: 200px !important;
  overflow: auto !important;
  white-space: pre !important;
  overflow-wrap: normal !important;
}
.gradio-container textarea {
  overflow: auto !important;
}
"""



# -------------------------
# Types / containers
# -------------------------

Agent = Any  # duck-typed


@dataclass
class OutputView:
    title: str
    getter: Callable[[Dict[str, Any]], Any]


@dataclass
class SupervisionResult:
    final_output: Dict[str, Any]
    rounds: int
    previous_response_id: Optional[str]
    log_file: Optional[str]
    session_id: Optional[str]


@dataclass
class UIState:
    # JSON string shown in the Agent Input code box
    agent_input_json: str = "{}"
    # Last agent output (dict) + pretty JSON render
    last_output: Dict[str, Any] = None  # type: ignore[assignment]
    last_output_json: str = ""
    # Final/approved output JSON
    final_output_json: str = ""
    # Progress
    rounds: int = 0
    approved: bool = False
    previous_response_id: Optional[str] = None


# -------------------------
# Logging (JSONL)
# -------------------------

def _utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _log_jsonl(path: Optional[Path], session_id: str, event: str, payload: Dict[str, Any]) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "ts_utc": _utc_now_iso(),
        "session_id": session_id,
        "event": event,
        "payload": payload,
    }
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        # UI logging should never crash the app
        return


def _sanitize_md(s: str) -> str:
    s = s or ""
    # prevent accidental markdown code fences from breaking layout
    s = re.sub(r"```+", "```", s)
    return s


def _default_views() -> List[OutputView]:
    return [
        OutputView("Raw output", lambda out: out),
    ]


def _as_messages(chat_state: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Ensure Chatbot receives messages list."""
    if not chat_state:
        return []
    out: List[Dict[str, str]] = []
    for m in chat_state:
        if isinstance(m, dict) and "role" in m and "content" in m:
            out.append({"role": str(m["role"]), "content": str(m["content"])})
    return out


# -------------------------
# Main
# -------------------------


# Keep references to launched demos so the server doesn't get garbage-collected.
_LIVE_DEMOS: List[Any] = []


def supervise_ui(
    *,
    agent: Optional[Agent] = None,
    agent_input: Any = None,
    title: str = "Supervise Agent",
    description_md: Optional[str] = None,
    output_views: Optional[List[OutputView]] = None,
    log_file: Optional[str] = "logs/supervise_ui.jsonl",
    session_id: Optional[str] = None,
    max_rounds: int = 10,
    host: str = "127.0.0.1",
    port: int = 7860,
    open_browser: bool = True,
    keep_open: bool = False,
    enable_resource_desk: bool = False,
    resource_dir: Optional[str] = None,
    resource_poll_seconds: float = 1.5,
    run_id: Optional[str] = None,
    allow_edit_agent_input: bool = False,
    output_transform: Optional[Callable[[Dict[str, Any]], Any]] = None,
    approvable: Optional[Callable[[Dict[str, Any]], bool]] = None,
    assistant_message: Optional[Callable[[Dict[str, Any]], Optional[str]]] = None,
    block_until_approved: bool = True,
    watch_output_dir: Optional[str] = None,
) -> SupervisionResult:
    """
    Single-tab workspace UI that can:
      - supervise an agent (optional),
      - allow an approval gate (optional),
      - act as a persistent dashboard (logs, artifacts),
      - handle user-provided resources for any agent.
    """
    output_views = output_views or _default_views()
    sid = session_id or f"ui_{int(time.time())}"
    log_path = Path(log_file) if log_file else None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)

    out_dir: Optional[Path] = None
    if watch_output_dir:
        out_dir = Path(watch_output_dir)
    elif run_id:
        out_dir = Path("outputs") / run_id

    if enable_resource_desk and resource_dir is None:
        if out_dir is not None:
            resource_dir = str(out_dir / "resources")

    res_eval = ResourceEval(resource_dir) if (enable_resource_desk and resource_dir) else None
    human_desk = HumanRequestDesk(resource_dir) if (enable_resource_desk and resource_dir) else None

    state = UIState(
        agent_input_json=json.dumps(agent_input or {}, ensure_ascii=False, indent=2),
        last_output={},
        last_output_json="",
        final_output_json="",
        rounds=0,
        approved=False,
        previous_response_id=None,
    )

    approved_event = threading.Event()
    holder: Dict[str, Any] = {}

    def _log(event: str, **payload: Any) -> None:
        _log_jsonl(log_path, sid, event, dict(payload))

    def _tail_text(path: Path, *, max_bytes: int = 9000) -> str:
        try:
            if not path.exists():
                return ""
            size = path.stat().st_size
            with path.open("rb") as f:
                if size > max_bytes:
                    f.seek(size - max_bytes)
                data = f.read()
            text = data.decode("utf-8", errors="replace")
            # keep last ~120 lines
            lines = text.splitlines()
            if len(lines) > 120:
                lines = lines[-120:]
            return "\n".join(lines)
        except Exception as e:
            return f"[log read error] {e}"

    def _scan_artifacts() -> List[str]:
        if out_dir is None:
            return []
        try:
            if not out_dir.exists():
                return []
            keep_ext = {".json", ".md", ".txt", ".py", ".zip"}
            items: List[str] = []
            for p in out_dir.rglob("*"):
                if not p.is_file():
                    continue
                if p.name in {"PAUSED.json", "DONE.flag", "RESUME.flag"}:
                    pass
                if p.suffix.lower() in keep_ext:
                    rel = str(p.relative_to(out_dir)).replace("\\", "/")
                    items.append(rel)
            items.sort()
            return items
        except Exception:
            return []

    def _load_artifact(rel: str) -> Tuple[str, Optional[str]]:
        if out_dir is None:
            return "", None
        try:
            p = out_dir / rel
            if not p.exists() or not p.is_file():
                return "", None
            # Avoid dumping huge binaries into the preview
            if p.stat().st_size > 300_000:
                return f"[{rel}] is large ({p.stat().st_size} bytes). Use Download.", str(p)
            txt = p.read_text(encoding="utf-8", errors="replace")
            return txt, str(p)
        except Exception as e:
            return f"[artifact read error] {e}", None

    def _run_status() -> Tuple[str, bool, bool]:
        """returns (status_md, paused?, done?)"""
        paused = bool(out_dir and (out_dir / "PAUSED.json").exists())
        done = bool(out_dir and (out_dir / "DONE.flag").exists())
        blocked = False
        stale_blocked = False
        block_reason = ""
        block_type = ""
        current_node = ""
        state_done = False
        payload: Any = None
        awaiting_last_decision = False
        last_decision_reason = ""
        if out_dir is not None:
            try:
                rs_path = out_dir / "run_state.json"
                if rs_path.exists():
                    rs = json.loads(rs_path.read_text(encoding="utf-8"))
                    if isinstance(rs, dict):
                        blocked = bool(rs.get("blocked"))
                        block_type = str(rs.get("block_type") or "")
                        block_reason = str(rs.get("block_reason") or "")
                        current_node = str(rs.get("current_node") or "")
                        state_done = bool(rs.get("done"))
                        payload = rs.get("block_payload")
                        last_decision = rs.get("last_decision") if isinstance(rs.get("last_decision"), dict) else None
                        if last_decision and str(last_decision.get("action") or "") == "ask_user":
                            awaiting_last_decision = True
                            last_decision_reason = str(last_decision.get("reason") or last_decision.get("target_stage") or "user decision needed")
                        node_status = rs.get("node_status") if isinstance(rs.get("node_status"), dict) else {}
                        stale_blocked = (not blocked) and any(str(v) == "blocked" for v in node_status.values()) and not bool(rs.get("done"))
                        if stale_blocked and not block_reason:
                            block_reason = "node_status_contains_blocked_but_run_state_block_flag_is_false"
                            block_type = "stale_or_resolved_block"
            except Exception:
                pass
        done = done or state_done
        status = "RUNNING"
        if blocked:
            status = f"BLOCKED ({block_type or block_reason or 'needs attention'})"
        elif stale_blocked:
            status = "STALE/BLOCKED STATE — restart or resume needed"
        elif awaiting_last_decision:
            status = "WAITING FOR USER DECISION"
        if paused:
            status = "PAUSED (needs Resume)"
        if done:
            status = "DONE"
        rid = run_id or (out_dir.name if out_dir else "n/a")
        od = str(out_dir) if out_dir else "n/a"
        extra = f"  \n**Current node:** `{current_node}`" if current_node else ""
        md = f"**Run:** `{rid}`  \n**Status:** {status}{extra}  \n**Output dir:** `{od}`"
        if blocked or stale_blocked or awaiting_last_decision:
            action = "Open **Requests & Resources**, click **Refresh requests**, then follow the Team Lead request chat. Upload files only if a real external resource is needed."
            if awaiting_last_decision:
                action = "Open **Requests & Resources**, click **Refresh requests**, then answer the synthesized Team Lead request from the latest workflow decision."
            elif block_type in {"human_input", "human_decision", "decision"}:
                action = "Open **Requests & Resources**, click **Refresh requests**, answer the Team Lead request chat, then keep the app open."
            elif stale_blocked:
                action = "The run has an inconsistent blocked marker. Restart with FORCE_NEW_RUN unset; if it remains blocked, inspect ENGINEERING_BLOCKED.json."
            elif block_type != "resource":
                action = "Check the Log tail and run_state.json artifact for the required action."
            brief_payload = ""
            if isinstance(payload, dict):
                req_id = payload.get("request_id")
                items = payload.get("items")
                item_names = []
                if isinstance(items, list):
                    for it in items[:5]:
                        if isinstance(it, dict) and it.get("name"):
                            item_names.append(str(it.get("name")))
                if req_id or item_names:
                    brief_payload = f"  \n**Request:** `{req_id or ''}`"
                    if item_names:
                        brief_payload += "  \n**Items:** " + "; ".join(item_names)
            md += f"\n\n<div id='run_block_banner'>⚠️ <b>Human action needed.</b><br>Block reason: <code>{block_reason or block_type}</code>{brief_payload}<br>{action}</div>"
        if stale_blocked and not blocked:
            md += f"\n\n<div id='run_block_banner'>⚠️ <b>Workflow state needs attention.</b><br>Reason: <code>{block_reason}</code><br>The server may have stopped after a human/resource unblock. Restart with <code>FORCE_NEW_RUN</code> unset so the engineering queue can continue.</div>"
        return md, paused, done

    def _write_resume_flag() -> str:
        if out_dir is None:
            return "No output dir."
        try:
            (out_dir / "RESUME.flag").write_text("resume\n", encoding="utf-8")
            return "Resume requested."
        except Exception as e:
            return f"Resume write failed: {e}"

    def _read_run_state() -> Dict[str, Any]:
        if out_dir is None:
            return {}
        try:
            p = out_dir / "run_state.json"
            if p.exists():
                obj = json.loads(p.read_text(encoding="utf-8"))
                return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
        return {}


    # -------------------------
    # Dynamic Agent Inspector / Team Lead directive routing
    # -------------------------
    _AGENT_INSPECTOR_CHOICES = [
        "Team Lead / Intake",
        "PM Agent",
        "UX Agent",
        "Engineering Lead",
        "Orchestrator",
        "Engineer(s)",
        "QA Agent",
        "Project Executor",
        "Runbook / Final Handoff",
    ]

    def _safe_json_file(rel: str) -> Any:
        if out_dir is None:
            return None
        try:
            p = out_dir / rel
            if p.exists() and p.is_file():
                return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
        return None

    def _recent_files_under(rel_dir: str, *, limit: int = 8) -> List[str]:
        if out_dir is None:
            return []
        try:
            d = out_dir / rel_dir
            if not d.exists():
                return []
            files = [p for p in d.rglob("*") if p.is_file()]
            files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return [str(p.relative_to(out_dir)).replace("\\", "/") for p in files[:limit]]
        except Exception:
            return []

    def _agent_status_payload(agent_name: str) -> Dict[str, Any]:
        rs = _read_run_state()
        node_status = rs.get("node_status") if isinstance(rs.get("node_status"), dict) else {}
        active_task = rs.get("active_task_id")
        active_engineer = rs.get("active_engineer_id")
        payload: Dict[str, Any] = {
            "selected_agent": agent_name,
            "run_id": run_id or (out_dir.name if out_dir else None),
            "current_node": rs.get("current_node"),
            "resume_from": rs.get("resume_from"),
            "run_blocked": bool(rs.get("blocked")),
            "block_type": rs.get("block_type"),
            "block_reason": rs.get("block_reason"),
            "active_task_id": active_task,
            "active_engineer_id": active_engineer,
        }
        if agent_name == "Team Lead / Intake":
            payload.update({
                "role": "User-facing coordinator and intake owner",
                "status": node_status.get("intake", "pending"),
                "input_artifact": "user messages / targeted directives",
                "output_artifact": "initial_input.json",
                "artifact": _safe_json_file("initial_input.json"),
            })
        elif agent_name == "PM Agent":
            payload.update({
                "role": "Product requirements and scope handoff",
                "status": node_status.get("pm", "pending"),
                "input_artifact": "initial_input.json",
                "output_artifact": "pm_output.json",
                "artifact": _safe_json_file("pm_output.json"),
            })
        elif agent_name == "UX Agent":
            payload.update({
                "role": "UX flows, UI behavior, user experience constraints",
                "status": node_status.get("ux", "pending"),
                "input_artifact": "initial_input.json + pm_output.json",
                "output_artifact": "ux_output.json",
                "artifact": _safe_json_file("ux_output.json"),
            })
        elif agent_name == "Engineering Lead":
            payload.update({
                "role": "Task graph, dependencies, files_expected, engineering plan",
                "status": node_status.get("eng_lead", "pending"),
                "input_artifact": "initial_input.json + pm_output.json + ux_output.json",
                "output_artifact": "eng_lead_output.json / task_graph_validation.json",
                "artifact": _safe_json_file("eng_lead_output.json"),
                "validation": _safe_json_file("task_graph_validation.json"),
            })
        elif agent_name == "Orchestrator":
            payload.update({
                "role": "Task queue and workflow state controller",
                "status": node_status.get("orchestrator_ingest", "pending"),
                "input_artifact": "eng_lead_output.json task graph",
                "output_artifact": "orchestrator_state.json",
                "artifact": _safe_json_file("orchestrator_state.json"),
            })
        elif agent_name == "Engineer(s)":
            payload.update({
                "role": "Implementation agent(s) writing staged changes",
                "status": node_status.get("engineering", "pending"),
                "current_task": active_task,
                "current_engineer": active_engineer,
                "task_attempts": rs.get("task_attempts"),
                "recent_engineer_outputs": _recent_files_under("engineer"),
                "recent_staged_writes": _recent_files_under("staged_file_writes"),
                "repair_packets": _recent_files_under("repair_packets"),
            })
        elif agent_name == "QA Agent":
            payload.update({
                "role": "Review, verification, and repair evidence",
                "status": node_status.get("engineering", "pending"),
                "current_task": active_task,
                "recent_qa_outputs": _recent_files_under("qa"),
                "repair_packets": _recent_files_under("repair_packets"),
            })
        elif agent_name == "Project Executor":
            payload.update({
                "role": "Final workspace/build application step after engineering passes",
                "status": node_status.get("project_executor", "pending"),
                "recent_outputs": _recent_files_under("file_writes"),
            })
        elif agent_name == "Runbook / Final Handoff":
            payload.update({
                "role": "Final instructions, known limits, and handoff",
                "status": node_status.get("runbook", node_status.get("coordinator_final_handoff", "pending")),
                "recent_docs": _recent_files_under("docs"),
            })
        return payload

    def _agent_status_markdown(agent_name: str) -> str:
        info = _agent_status_payload(agent_name)
        status = str(info.get("status") or "pending")
        role = str(info.get("role") or "")
        blocked = "yes" if info.get("run_blocked") else "no"
        active = str(info.get("active_task_id") or "n/a")
        eng = str(info.get("active_engineer_id") or "n/a")
        md = f"### {agent_name}\n"
        if role:
            md += f"**Role:** {role}  \n"
        md += f"**Status:** `{status}`  \n**Run blocked:** `{blocked}`  \n**Active task:** `{active}`  \n**Active engineer:** `{eng}`"
        if info.get("block_reason"):
            md += f"  \n**Block reason:** `{info.get('block_reason')}`"
        if info.get("output_artifact"):
            md += f"  \n**Output artifact:** `{info.get('output_artifact')}`"
        return md

    def _inspect_agent(agent_name: str) -> Tuple[str, str]:
        agent_name = agent_name or "Team Lead / Intake"
        payload = _agent_status_payload(agent_name)
        return _agent_status_markdown(agent_name), json.dumps(payload, ensure_ascii=False, indent=2)

    def _write_agent_inspector_directive(agent_name: str, message: str) -> Tuple[str, str]:
        agent_name = agent_name or "Team Lead / Intake"
        msg = (message or "").strip()
        if not msg:
            return "No directive sent. Write a message first.", json.dumps(_agent_status_payload(agent_name), ensure_ascii=False, indent=2)
        rs = _read_run_state()
        active_task = str(rs.get("active_task_id") or "").strip()
        event = {
            "schema_version": "team_lead_directive.v1",
            "created_at_utc": _utc_now_iso(),
            "run_id": run_id or (out_dir.name if out_dir else None),
            "target_agent": agent_name,
            "target_task_id": active_task or None,
            "received_by": "Team Lead",
            "routing_rule": "User-selected agent messages are routed through Team Lead before any internal agent sees them.",
            "user_message": msg,
            "status": "recorded",
        }
        try:
            base = Path(resource_dir) if resource_dir else (out_dir / "resources" if out_dir else None)
            if base is None:
                return "Directive not saved: no resource/project state directory is available yet.", json.dumps(event, ensure_ascii=False, indent=2)
            d = base / "team_lead_directives"
            d.mkdir(parents=True, exist_ok=True)
            safe_agent = re.sub(r"[^A-Za-z0-9_.-]+", "_", agent_name).strip("_") or "agent"
            fname = f"{int(time.time())}_{uuid.uuid4().hex[:8]}_{safe_agent}.json"
            (d / fname).write_text(json.dumps(event, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            with (base / "team_lead_directives.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")

            # If there is an active engineering task, also expose this as a task
            # directive so the next Engineer/QA retry receives it. It is still
            # explicitly Team Lead-routed, not a direct user-to-agent bypass.
            if active_task and agent_name in {"Engineer(s)", "QA Agent", "Engineering Lead", "Orchestrator"}:
                hd = base / "human_task_directives"
                hd.mkdir(parents=True, exist_ok=True)
                p = hd / f"{active_task}.json"
                existing: Dict[str, Any] = {}
                try:
                    if p.exists():
                        obj = json.loads(p.read_text(encoding="utf-8"))
                        existing = obj if isinstance(obj, dict) else {}
                except Exception:
                    existing = {}
                # Preserve any existing request-resolution directive (for
                # example max-attempt approval, reassignment, or resource
                # decision). User guidance from the Agent Inspector is appended
                # as Team Lead-routed context instead of overwriting the active
                # decision semantics needed by the scheduler.
                history = existing.get("team_lead_user_directives") if isinstance(existing.get("team_lead_user_directives"), list) else []
                history.append(event)
                overrides = existing.get("overrides") if isinstance(existing.get("overrides"), dict) else {}
                tl_overrides = overrides.get("team_lead_user_directives") if isinstance(overrides.get("team_lead_user_directives"), list) else []
                tl_overrides.append({
                    "created_at_utc": event["created_at_utc"],
                    "target_agent": agent_name,
                    "user_message": msg,
                    "source": "agent_inspector_team_lead_directive",
                })
                overrides["team_lead_user_directives"] = tl_overrides
                existing.update({
                    "team_lead_user_directives": history,
                    "latest_team_lead_user_message": msg,
                    "latest_team_lead_target_agent": agent_name,
                    "target_task_id": active_task,
                    "received_by": "Team Lead",
                    "source": existing.get("source") or "agent_inspector_team_lead_directive",
                    "overrides": overrides,
                    "updated_at_utc": event["created_at_utc"],
                })
                if not existing.get("decision"):
                    existing["decision"] = "targeted_directive"
                if not existing.get("user_message"):
                    existing["user_message"] = msg
                p.write_text(json.dumps(existing, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return f"Directive recorded for Team Lead routing to {agent_name}.", json.dumps(event, ensure_ascii=False, indent=2)
        except Exception as exc:
            event["status"] = "error"
            event["error"] = str(exc)
            return f"Directive save failed: {exc}", json.dumps(event, ensure_ascii=False, indent=2)

    def _write_team_lead_chat_directive(message: str, *, source: str = "team_lead_chat") -> Dict[str, Any]:
        """Record a Team Lead chat message as a structured directive.

        After intake is approved, or in dashboard-only mode, the main chat must
        remain useful without re-running intake. Messages are therefore routed
        through Team Lead into the same directive channel used by Agent
        Inspector. Downstream agents read these via ProjectState/resources.
        """
        msg = (message or "").strip()
        rs = _read_run_state()
        active_task = str(rs.get("active_task_id") or "").strip()
        event: Dict[str, Any] = {
            "schema_version": "team_lead_directive.v1",
            "created_at_utc": _utc_now_iso(),
            "run_id": run_id or (out_dir.name if out_dir else None),
            # Keep this name so existing downstream filters that always include
            # Team Lead / Intake continue to receive global Team Lead guidance.
            "target_agent": "Team Lead / Intake",
            "target_task_id": active_task or None,
            "received_by": "Team Lead",
            "routing_rule": "Main chat messages after intake/dashboard mode become Team Lead-routed directives, not direct agent messages.",
            "user_message": msg,
            "status": "recorded",
            "source": source,
        }
        if not msg:
            event["status"] = "ignored_empty"
            return event
        try:
            base = Path(resource_dir) if resource_dir else (out_dir / "resources" if out_dir else None)
            if base is None:
                event["status"] = "error"
                event["error"] = "no resource/project state directory is available yet"
                return event
            d = base / "team_lead_directives"
            d.mkdir(parents=True, exist_ok=True)
            fname = f"{int(time.time())}_{uuid.uuid4().hex[:8]}_team_lead_chat.json"
            (d / fname).write_text(json.dumps(event, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            with (base / "team_lead_directives.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")

            # If a concrete task is active, also expose the directive to the
            # per-task human directive channel. This lets the next Engineer/QA
            # attempt consume the user's guidance without bypassing Team Lead.
            if active_task:
                hd = base / "human_task_directives"
                hd.mkdir(parents=True, exist_ok=True)
                hp = hd / f"{active_task}.json"
                existing: Dict[str, Any] = {}
                try:
                    if hp.exists():
                        obj = json.loads(hp.read_text(encoding="utf-8"))
                        existing = obj if isinstance(obj, dict) else {}
                except Exception:
                    existing = {}
                # Preserve any existing request-resolution directive. Main
                # Team Lead chat guidance should augment the task context, not
                # erase max-attempt approvals, reassignments, resource decisions,
                # or block/continue semantics already stored for the task.
                history = existing.get("team_lead_user_directives") if isinstance(existing.get("team_lead_user_directives"), list) else []
                history.append(event)
                overrides = existing.get("overrides") if isinstance(existing.get("overrides"), dict) else {}
                tl_overrides = overrides.get("team_lead_user_directives") if isinstance(overrides.get("team_lead_user_directives"), list) else []
                tl_overrides.append({
                    "created_at_utc": event["created_at_utc"],
                    "target_agent": "Team Lead / Intake",
                    "user_message": msg,
                    "source": source,
                })
                overrides["team_lead_user_directives"] = tl_overrides
                existing.update({
                    "team_lead_user_directives": history,
                    "latest_team_lead_user_message": msg,
                    "latest_team_lead_target_agent": "Team Lead / Intake",
                    "target_task_id": active_task,
                    "received_by": "Team Lead",
                    "source": existing.get("source") or source,
                    "overrides": overrides,
                    "updated_at_utc": event["created_at_utc"],
                })
                if not existing.get("decision"):
                    existing["decision"] = "targeted_directive"
                if not existing.get("user_message"):
                    existing["user_message"] = msg
                hp.write_text(json.dumps(existing, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return event
        except Exception as exc:
            event["status"] = "error"
            event["error"] = str(exc)
            return event

    def _send_team_lead_dashboard_message(user_msg: str, chat_history: List[Dict[str, Any]]) -> Tuple[List[Dict[str, str]], str]:
        chat_history = _as_messages(chat_history or [])
        msg = (user_msg or "").strip()
        if not msg:
            return chat_history, "Write a message first."
        event = _write_team_lead_chat_directive(msg, source="team_lead_chat_dashboard")
        reply = "Recorded as a Team Lead directive. I will route it into the workflow state for the relevant downstream agents."
        if event.get("status") == "error":
            reply = f"I could not save the Team Lead directive: {event.get('error')}"
        return chat_history + [{"role": "user", "content": msg}, {"role": "assistant", "content": reply}], ""

    def _resource_request_from_run_state() -> Optional[Dict[str, Any]]:
        rs = _read_run_state()
        if not (isinstance(rs, dict) and rs.get("blocked") and rs.get("block_type") == "resource"):
            return None
        payload = rs.get("block_payload")
        if not isinstance(payload, dict):
            return None
        req_id = payload.get("request_id")
        if not req_id:
            return None
        context = payload.get("context") if isinstance(payload.get("context"), dict) else {}
        user_facing = payload.get("user_facing") if isinstance(payload.get("user_facing"), dict) else {}
        return {
            "request_id": str(req_id),
            "created_at_utc": str(rs.get("updated_at_utc") or ""),
            "agent": str(payload.get("agent") or payload.get("stage") or "agent"),
            "stage": str(payload.get("stage") or "stage"),
            "task_id": payload.get("task_id"),
            "items": payload.get("items") if isinstance(payload.get("items"), list) else [],
            "context": context,
            "user_facing": user_facing,
            "reason": str(payload.get("reason") or rs.get("block_reason") or "resource_required"),
            "status": "pending",
            "resolved_at_utc": None,
            "source": "run_state.block_payload",
        }

    def _human_request_from_run_state() -> Optional[Dict[str, Any]]:
        """Return a visible human request reconstructed from run_state.

        Normal pending human requests live in pending_human_requests.json. Older
        or interrupted runs can also preserve the Team Lead decision only inside
        run_state.block_payload or run_state.last_decision. The request deck must
        surface both; otherwise the workflow can wait for a user decision while
        the UI says there are no requests.
        """
        rs = _read_run_state()
        if not isinstance(rs, dict):
            return None

        if rs.get("blocked") and str(rs.get("block_type") or "") in {"human_input", "human_decision", "decision"}:
            payload = rs.get("block_payload")
            if not isinstance(payload, dict):
                return None
            req_id = payload.get("request_id")
            if not req_id:
                return None
            # Preserve the original payload context directly, while also keeping the
            # fallback wrapper for traceability. This lets downstream Team Lead logic
            # recover engineer_id/attempt/max_attempt_check from fallback-resolved UI
            # requests instead of falling back to stale run_state.active_engineer_id.
            context: Dict[str, Any] = {}
            payload_context = payload.get("context")
            if isinstance(payload_context, dict):
                context.update(payload_context)
            for key in ("engineer_id", "attempt", "previous_engineer_id", "stage"):
                if key in payload and key not in context:
                    context[key] = payload.get(key)
            context.update({"source": "run_state.block_payload", "block_payload": payload, "last_decision": rs.get("last_decision")})
            return {
                "request_id": str(req_id),
                "created_at_utc": str(rs.get("updated_at_utc") or ""),
                "stage": str(payload.get("stage") or rs.get("current_node") or "workflow"),
                "agent": str(payload.get("agent") or "Team Lead"),
                "task_id": payload.get("task_id"),
                "reason": str(payload.get("reason") or rs.get("block_reason") or "human_input_required"),
                "question": str(payload.get("question") or "Human input required."),
                "options": payload.get("options") if isinstance(payload.get("options"), list) else [],
                "context": context,
                "status": "pending",
                "resolved_at_utc": None,
                "source": "run_state.block_payload",
            }

        # Critical fallback: the coordinator can set last_decision.action=ask_user
        # without a block_payload/pending_human_requests entry. Surface that as a
        # synthetic request so the user has a visible way to answer it.
        last = rs.get("last_decision") if isinstance(rs.get("last_decision"), dict) else None
        if not last or str(last.get("action") or "") != "ask_user":
            return None
        task_id = str(last.get("affected_task_id") or rs.get("active_task_id") or "").strip() or None
        stage = str(last.get("target_stage") or rs.get("current_node") or "workflow")
        response_id = str(last.get("response_id") or "").strip()
        safe_stage = re.sub(r"[^A-Za-z0-9_.-]+", "_", stage).strip("_") or "workflow"
        safe_task = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(task_id or "task")).strip("_") or "task"
        req_id = f"run_state_last_decision_{response_id}" if response_id else f"run_state_last_decision_{safe_task}_{safe_stage}"
        context: Dict[str, Any] = {
            "source": "run_state.last_decision",
            "workflow_decision": last,
            "last_decision": last,
            "max_attempt_check": last.get("max_attempt_check") if isinstance(last.get("max_attempt_check"), dict) else {},
            "engineer_id": rs.get("active_engineer_id"),
            "attempt": (last.get("max_attempt_check") or {}).get("current_attempt") if isinstance(last.get("max_attempt_check"), dict) else None,
        }
        return {
            "request_id": req_id,
            "created_at_utc": str(rs.get("updated_at_utc") or ""),
            "stage": stage,
            "agent": "Team Lead",
            "task_id": task_id,
            "reason": str(last.get("reason") or "human_input_required"),
            "question": str(last.get("required_input") or "The workflow needs your decision before it can continue."),
            "options": ["rerun with my clarification", "reduce scope", "accept limitation and continue", "block requested item / route around", "hard stop whole task"],
            "context": context,
            "status": "pending",
            "resolved_at_utc": None,
            "source": "run_state.last_decision",
        }

    def _pending_human_requests() -> List[Dict[str, Any]]:
        pending: List[Dict[str, Any]] = []
        if human_desk is not None:
            try:
                raw = human_desk.list_pending()
                if isinstance(raw, list):
                    pending.extend([x for x in raw if isinstance(x, dict) and x.get("request_id")])
            except Exception:
                pass
        fallback = _human_request_from_run_state()
        if fallback and not _request_already_resolved("human", str(fallback.get("request_id") or "")) and not any(str(x.get("request_id")) == str(fallback.get("request_id")) for x in pending):
            pending.append(fallback)
        return pending

    def _get_human_request(req_id: str) -> Optional[Dict[str, Any]]:
        if req_id and human_desk is not None:
            try:
                req = human_desk.get_request(req_id)
                if isinstance(req, dict):
                    return req
            except Exception:
                pass
        fallback = _human_request_from_run_state()
        if fallback and (not req_id or str(fallback.get("request_id")) == str(req_id)):
            if not _request_already_resolved("human", str(fallback.get("request_id") or "")):
                return fallback
        return None

    def _default_human_request_id() -> str:
        pending = _pending_human_requests()
        if pending:
            return str(pending[0].get("request_id") or "")
        fallback = _human_request_from_run_state()
        return str(fallback.get("request_id") or "") if fallback else ""

    def _pending_resource_requests() -> List[Dict[str, Any]]:
        pending: List[Dict[str, Any]] = []
        if res_eval is not None:
            try:
                raw = res_eval.list_pending()
                if isinstance(raw, list):
                    pending.extend([x for x in raw if isinstance(x, dict) and x.get("request_id")])
            except Exception:
                pass
        fallback = _resource_request_from_run_state()
        if fallback and not _request_already_resolved("resource", str(fallback.get("request_id") or "")) and not any(str(x.get("request_id")) == str(fallback.get("request_id")) for x in pending):
            pending.append(fallback)
        return pending

    def _get_resource_request(req_id: str) -> Optional[Dict[str, Any]]:
        if req_id and res_eval is not None:
            try:
                req = res_eval.get_request(req_id)
                if isinstance(req, dict):
                    return req
            except Exception:
                pass
        fallback = _resource_request_from_run_state()
        if fallback and (not req_id or str(fallback.get("request_id")) == str(req_id)):
            if not _request_already_resolved("resource", str(fallback.get("request_id") or "")):
                return fallback
        return None

    def _default_resource_request_id() -> str:
        pending = _pending_resource_requests()
        if pending:
            return str(pending[0].get("request_id") or "")
        fallback = _resource_request_from_run_state()
        return str(fallback.get("request_id") or "") if fallback else ""

    def _resource_desk_banner() -> str:
        pending = _pending_resource_requests()
        fallback = _resource_request_from_run_state()
        if pending:
            req = pending[0]
            names = []
            for it in req.get("items") or []:
                if isinstance(it, dict) and it.get("name"):
                    names.append(str(it.get("name")))
            return (
                "<div id='resource_desk_banner'>⚠️ <b>Resource action available.</b> "
                f"Request <code>{req.get('request_id','')}</code>. "
                + ("Items: " + "; ".join(names[:5]) if names else "")
                + "<br>Upload files or click <b>Proceed without these resources</b>.</div>"
            )
        if fallback:
            return "<div id='resource_desk_banner'>⚠️ Resource block detected in run_state. Click Refresh or Proceed without these resources.</div>"
        return "<div id='resource_desk_banner'>No pending resource request.</div>"

    def _clear_run_block_if_matching(req_id: str) -> None:
        if out_dir is None:
            return
        try:
            rs_path = out_dir / "run_state.json"
            if not rs_path.exists():
                return
            rs = json.loads(rs_path.read_text(encoding="utf-8"))
            if not isinstance(rs, dict):
                return
            changed = False
            payload = rs.get("block_payload") if isinstance(rs.get("block_payload"), dict) else {}
            if rs.get("blocked") and str(payload.get("request_id") or "") == str(req_id):
                old_type = str(rs.get("block_type") or "")
                current = rs.get("current_node")
                node_status = rs.get("node_status") if isinstance(rs.get("node_status"), dict) else {}
                if current and node_status.get(current) == "blocked":
                    node_status[current] = "running"
                    rs["node_status"] = node_status
                rs["blocked"] = False
                rs["block_type"] = None
                rs["block_reason"] = None
                rs["block_payload"] = None
                rs["last_event"] = f"{old_type or 'block'}.ui_resolved"
                changed = True

            last = rs.get("last_decision") if isinstance(rs.get("last_decision"), dict) else None
            if last and str(last.get("action") or "") == "ask_user":
                response_id = str(last.get("response_id") or "").strip()
                task_id = str(last.get("affected_task_id") or rs.get("active_task_id") or "task")
                stage = str(last.get("target_stage") or rs.get("current_node") or "workflow")
                safe_stage = re.sub(r"[^A-Za-z0-9_.-]+", "_", stage).strip("_") or "workflow"
                safe_task = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id).strip("_") or "task"
                synthetic_id = f"run_state_last_decision_{response_id}" if response_id else f"run_state_last_decision_{safe_task}_{safe_stage}"
                if str(req_id) == synthetic_id:
                    ack = dict(last)
                    ack["resolved_via_ui_at_utc"] = _utc_now_iso()
                    rs["last_decision"] = {"action": "resolved", "previous_decision": ack}
                    rs["last_event"] = "last_decision.ui_resolved"
                    changed = True

            if changed:
                rs["updated_at_utc"] = _utc_now_iso()
                rs_path.write_text(json.dumps(rs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception:
            pass

    def _human_decision_blocks_task(decision: str, user_message: str = "") -> bool:
        decision_text = str(decision or "").strip().lower()
        msg = str(user_message or "").strip().lower()
        if decision_text in {"accept_limitation", "mark_unavailable", "continue", "provided", "route_around", "defer"}:
            return False
        if any(neg in msg for neg in ("do not block", "don't block", "not block", "route around", "blocked the requested item", "requested item/resource only", "reduce scope", "defer")):
            return False
        if decision_text in {"block", "block_task", "block task", "halt", "stop"}:
            return True
        return msg.startswith("block whole") or msg.startswith("stop whole") or "halt work" in msg or "stop work" in msg or "hard stop" in msg

    def _mark_run_blocked_by_user(req: Dict[str, Any], *, decision: str, user_message: str) -> None:
        if out_dir is None:
            return
        try:
            rs_path = out_dir / "run_state.json"
            if not rs_path.exists():
                return
            rs = json.loads(rs_path.read_text(encoding="utf-8"))
            if not isinstance(rs, dict):
                return
            current = rs.get("current_node") or "engineering"
            node_status = rs.get("node_status") if isinstance(rs.get("node_status"), dict) else {}
            node_status[str(current)] = "blocked"
            rs["node_status"] = node_status
            rs["blocked"] = True
            rs["block_type"] = "human_decision"
            rs["block_reason"] = "user_blocked_task"
            rs["block_payload"] = {
                "task_id": req.get("task_id"),
                "request_id": req.get("request_id"),
                "decision": decision,
                "user_message": user_message,
            }
            rs["last_event"] = "human_decision.user_blocked_task"
            rs["updated_at_utc"] = _utc_now_iso()
            rs_path.write_text(json.dumps(rs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception:
            pass

    def _append_fallback_resolution_log(kind: str, record: Dict[str, Any]) -> None:
        """Append a synthetic resolution event when resolving a run_state fallback.

        The background monitor unblocks the in-memory EngineeringOrchestrator by
        tailing resolved_*.jsonl. If the UI reconstructed the request from
        run_state rather than pending_requests.json, ResourceEval/HumanRequestDesk
        returns already=True and would not write a log line. Without this, the UI
        can clear run_state while the orchestrator task remains blocked.
        """
        try:
            if not resource_dir:
                return
            base = Path(resource_dir)
            if kind == "human":
                path = base / "resolved_human_requests.jsonl"
            else:
                path = base / "resolved_requests.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _unblock_orchestrator_state_task(req: Dict[str, Any], *, note: str) -> None:
        """Best-effort file-level unblock for resume-after-UI fallback cases.

        If the backend process is running, the monitor handles the in-memory
        orchestrator. If the user resolves a fallback request and restarts later,
        this prevents orchestrator_state.json from preserving a blocked/claimed
        task that can never be claimed again.
        """
        if out_dir is None or not isinstance(req, dict):
            return
        task_id = str(req.get("task_id") or "").strip()
        if not task_id:
            return
        try:
            p = out_dir / "orchestrator_state.json"
            if not p.exists():
                return
            state = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(state, dict):
                return
            runtime = state.get("runtime") if isinstance(state.get("runtime"), dict) else {}
            rt = runtime.get(task_id) if isinstance(runtime.get(task_id), dict) else None
            if not isinstance(rt, dict) or rt.get("status") == "done":
                return
            rt["status"] = "todo"
            rt["claimed_by"] = None
            rt["claimed_at"] = None
            rt["claim_expires_at"] = None
            rt["notes"] = str(note or "ui_fallback_unblocked")
            locks = state.get("locks") if isinstance(state.get("locks"), dict) else {}
            if locks:
                state["locks"] = {str(k): v for k, v in locks.items() if str(v) != task_id}
            meta = state.get("meta") if isinstance(state.get("meta"), dict) else {}
            meta["ui_fallback_unblocked_at"] = time.time()
            state["meta"] = meta
            p.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception:
            pass

    def _structured_task_ids(obj: Any) -> List[str]:
        found: List[str] = []

        def add(value: Any) -> None:
            text = str(value or "").strip()
            if text and text not in found:
                found.append(text)

        def walk(x: Any) -> None:
            if not isinstance(x, dict):
                return
            add(x.get("task_id"))
            add(x.get("target_task_id"))
            add(x.get("affected_task_id"))
            wi = x.get("work_item")
            if isinstance(wi, dict):
                add(wi.get("task_id"))
            wd = x.get("workflow_decision")
            if isinstance(wd, dict):
                add(wd.get("affected_task_id"))
                if isinstance(wd.get("context_pack"), dict):
                    walk(wd["context_pack"])
            bp = x.get("block_payload")
            if isinstance(bp, dict):
                walk(bp)
                if isinstance(bp.get("context"), dict):
                    walk(bp["context"])
            if isinstance(x.get("context"), dict):
                walk(x["context"])
            if isinstance(x.get("request_context"), dict):
                walk(x["request_context"])

        walk(obj)
        return found

    def _canonical_task_id_for_updates(fallback_task_id: str, updates: Dict[str, Any]) -> str:
        ids = _structured_task_ids(updates)
        return str((ids[-1] if ids else fallback_task_id) or fallback_task_id or "").strip()

    def _merge_task_directive(task_id: str, directive_dir_name: str, updates: Dict[str, Any]) -> None:
        """Merge UI-written directives instead of overwriting monitor/desk context."""
        try:
            if not task_id or not resource_dir:
                return
            task_id = _canonical_task_id_for_updates(str(task_id), updates)
            if not task_id:
                return
            base = Path(resource_dir)
            d = base / directive_dir_name
            d.mkdir(parents=True, exist_ok=True)
            path = d / f"{task_id}.json"
            existing: Dict[str, Any] = {}
            if path.exists():
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    existing = loaded
            merged = dict(existing)
            # Do not discard richer previous context. New values win only when
            # they are meaningful, while provided_assets/metadata are preserved.
            for key, value in updates.items():
                if value is None:
                    continue
                if key in {"provided_assets", "request_items", "request_options"}:
                    # Preserve richer context already written by ResourceEval,
                    # HumanRequestDesk, or backend monitor loops. UI fallback
                    # resolutions often carry an empty list simply because the
                    # request was reconstructed from run_state; an empty list
                    # must not erase previously uploaded assets or original
                    # request items/options. Non-empty lists intentionally
                    # replace stale values.
                    if isinstance(value, list) and value:
                        merged[key] = value
                    elif key not in merged:
                        merged[key] = []
                elif key == "overrides":
                    old = merged.get("overrides") if isinstance(merged.get("overrides"), dict) else {}
                    new = value if isinstance(value, dict) else {}
                    merged[key] = {**old, **new}
                elif key == "request_context":
                    old = merged.get("request_context") if isinstance(merged.get("request_context"), dict) else {}
                    new = value if isinstance(value, dict) else {}
                    merged[key] = {**old, **new}
                elif value != "":
                    merged[key] = value
                elif key not in merged:
                    merged[key] = value
            path.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception:
            pass

    def _write_fallback_resource_directive(req: Dict[str, Any], *, decision: str, user_message: str, provided_assets: Optional[List[Dict[str, Any]]] = None, overrides: Optional[Dict[str, Any]] = None) -> None:
        """Write/merge task_directives when the UI resolves a resource request.

        For normal pending requests, ResourceEval.resolve() already writes a
        directive and resolved log; this function only merges UI fallback/context.
        For run_state fallback requests, it also writes a synthetic resolved log
        and releases the persisted orchestrator task so resume can continue.
        """
        try:
            if not isinstance(req, dict) or not resource_dir:
                return
            task_id = str(req.get("task_id") or "").strip()
            if not task_id:
                return
            resolved_at = _utc_now_iso()
            is_run_state_fallback = req.get("source") == "run_state.block_payload"
            _merge_task_directive(task_id, "task_directives", {
                "request_id": str(req.get("request_id") or ""),
                "decision": str(decision or "provided"),
                "user_message": str(user_message or ""),
                "overrides": overrides or {},
                "provided_assets": provided_assets or [],
                "source": ("supervise_ui.run_state_fallback" if is_run_state_fallback else "supervise_ui.ui_resolution"),
                "request_stage": req.get("stage"),
                "request_agent": req.get("agent"),
                "request_items": req.get("items") if isinstance(req.get("items"), list) else [],
                "request_context": req.get("context") if isinstance(req.get("context"), dict) else {},
                "request_user_facing": req.get("user_facing") if isinstance(req.get("user_facing"), dict) else {},
                "resolved_at_utc": resolved_at,
            })
            req2 = dict(req)
            req2["status"] = "resolved"
            req2["resolved_at_utc"] = resolved_at
            if is_run_state_fallback:
                _append_fallback_resolution_log("resource", {
                    "event": "resolved",
                    "request": req2,
                    "note": str(user_message or ""),
                    "decision": str(decision or "provided"),
                    "user_message": str(user_message or ""),
                    "overrides": overrides or {},
                    "provided_assets": provided_assets or [],
                    "source": "supervise_ui.run_state_fallback",
                })
            if _human_decision_blocks_task(decision, user_message):
                _mark_run_blocked_by_user(req2, decision=str(decision or "block"), user_message=str(user_message or ""))
            else:
                _unblock_orchestrator_state_task(req2, note=f"resource_ui_resolved_{decision or 'resolved'}")
        except Exception:
            pass

    def _write_fallback_human_directive(req: Dict[str, Any], *, decision: str, user_message: str, overrides: Optional[Dict[str, Any]] = None) -> None:
        """Write/merge human_task_directives when the UI resolves a human request.

        Normal pending requests are resolved by HumanRequestDesk.resolve(). This
        helper preserves/merges UI context and handles run_state fallback requests
        whose pending request record may no longer exist.
        """
        try:
            if not isinstance(req, dict) or not resource_dir:
                return
            task_id = str(req.get("task_id") or "").strip()
            if not task_id:
                return
            resolved_at = _utc_now_iso()
            is_run_state_fallback = req.get("source") in {"run_state.block_payload", "run_state.last_decision"}
            _merge_task_directive(task_id, "human_task_directives", {
                "request_id": str(req.get("request_id") or ""),
                "decision": str(decision or "continue"),
                "user_message": str(user_message or ""),
                "overrides": overrides or {},
                "source": ("supervise_ui.run_state_fallback" if is_run_state_fallback else "supervise_ui.ui_resolution"),
                "request_stage": req.get("stage"),
                "request_agent": req.get("agent"),
                "request_reason": req.get("reason"),
                "request_options": req.get("options") if isinstance(req.get("options"), list) else [],
                "request_context": req.get("context") if isinstance(req.get("context"), dict) else {},
                "resolved_at_utc": resolved_at,
            })
            req2 = dict(req)
            req2["status"] = "resolved"
            req2["resolved_at_utc"] = resolved_at
            if is_run_state_fallback:
                _append_fallback_resolution_log("human", {
                    "event": "resolved",
                    "request": req2,
                    "decision": str(decision or "continue"),
                    "user_message": str(user_message or ""),
                    "overrides": overrides or {},
                    "source": "supervise_ui.run_state_fallback",
                })
            if _human_decision_blocks_task(decision, user_message):
                _mark_run_blocked_by_user(req2, decision=str(decision or "block"), user_message=str(user_message or ""))
            else:
                _unblock_orchestrator_state_task(req2, note=f"human_ui_resolved_{decision or 'resolved'}")
        except Exception:
            pass

    # -------------------------
    # Agent supervision callbacks
    # -------------------------
    def _call_agent(user_msg: str, chat_history: List[Dict[str, Any]], agent_input_json: str) -> Tuple:
        if chat_history is None:
            chat_history = []
        chat_history = _as_messages(chat_history)
        user_msg = (user_msg or "").strip()

        if agent is None:
            if user_msg:
                event = _write_team_lead_chat_directive(user_msg, source="team_lead_chat_dashboard")
                reply = "Recorded as a Team Lead directive for the running workflow."
                if event.get("status") == "error":
                    reply = f"I could not save the Team Lead directive: {event.get('error')}"
                chat_history = chat_history + [{"role": "user", "content": user_msg}, {"role": "assistant", "content": reply}]
            return (
                chat_history,
                state.last_output_json,
                state.final_output_json,
                gr.update(visible=False),
                gr.update(),
                gr.update(interactive=False),
                gr.update(visible=False),
            )

        # After initial_input approval, this same UI may remain open while the
        # pipeline runs. Do not keep calling the intake model for new messages;
        # route post-approval messages into Team Lead directives instead.
        if state.approved:
            if user_msg:
                event = _write_team_lead_chat_directive(user_msg, source="team_lead_chat_after_intake_approval")
                reply = "Recorded as a Team Lead directive for the running workflow."
                if event.get("status") == "error":
                    reply = f"I could not save the Team Lead directive: {event.get('error')}"
                chat_history = chat_history + [{"role": "user", "content": user_msg}, {"role": "assistant", "content": reply}]
            return (
                chat_history,
                state.last_output_json,
                state.final_output_json,
                gr.update(visible=False),
                gr.update(value=state.final_output_json, visible=False),
                gr.update(interactive=False),
                gr.update(visible=False),
            )

        # Reflect the user turn in the Team Lead chat before calling the agent.
        # This keeps the UI conversation aligned with the actual intake/request
        # history instead of showing only assistant replies.
        if user_msg:
            chat_history = chat_history + [{"role": "user", "content": user_msg}]

        # Update agent_input if editable
        if allow_edit_agent_input:
            try:
                state.agent_input_json = agent_input_json or "{}"
                agent_in = json.loads(state.agent_input_json)
            except Exception:
                agent_in = agent_input
        else:
            agent_in = agent_input

        # Build agent input
        payload = agent_in or {}
        if isinstance(payload, dict):
            payload = dict(payload)
            if user_msg:
                payload["user_message"] = user_msg

        _log("agent_call", round=state.rounds + 1, user_message=user_msg)

        try:
            run_kwargs = {"agent_input": payload, "previous_response_id": state.previous_response_id}
            try:
                sig = inspect.signature(agent.run)
                params = sig.parameters
                if "draft" in params:
                    run_kwargs["draft"] = state.last_output if isinstance(state.last_output, dict) else None
                if "feedback" in params:
                    run_kwargs["feedback"] = user_msg
            except Exception:
                # Conservative fallback for older agent wrappers.
                pass
            out = agent.run(**run_kwargs)
            if isinstance(out, tuple) and len(out) == 2:
                output, prev_id = out
            else:
                output, prev_id = out, state.previous_response_id
            state.previous_response_id = prev_id
        except Exception as e:
            chat_history = chat_history + [{"role": "assistant", "content": f"[agent error] {e}"}]
            _log("agent_error", error=str(e))
            return (
                chat_history,
                state.last_output_json,
                state.final_output_json,
                gr.update(visible=True),
                gr.update(value="", visible=False),
                gr.update(interactive=False, visible=True),
                gr.update(visible=False),
            )

        state.rounds += 1
        state.last_output = output if isinstance(output, dict) else {"output": output}
        try:
            state.last_output_json = json.dumps(state.last_output, ensure_ascii=False, indent=2)
        except Exception:
            state.last_output_json = _pretty(state.last_output)

        assistant = None
        if assistant_message is not None:
            try:
                assistant = assistant_message(state.last_output)
            except Exception:
                assistant = None

        if assistant is None:
            assistant = "Output updated."

        chat_history = chat_history + [{"role": "assistant", "content": assistant}]

        can_approve = False
        transformed = None
        if approvable is not None:
            try:
                can_approve = bool(approvable(state.last_output))
            except Exception:
                can_approve = False

        if can_approve and output_transform is not None:
            try:
                transformed = output_transform(state.last_output)
            except Exception:
                transformed = None

        approve_panel_update = gr.update(visible=True)
        approve_backdrop_update = gr.update(visible=False)
        approve_payload_update = gr.update(
            value=json.dumps(transformed, ensure_ascii=False, indent=2) if transformed is not None else state.last_output_json,
            visible=bool(can_approve),
        )
        approve_btn_update = gr.update(interactive=bool(can_approve), visible=True)

        _log("agent_output", round=state.rounds, can_approve=can_approve)

        return (
            chat_history,
            state.last_output_json,
            state.final_output_json,
            approve_panel_update,
            approve_payload_update,
            approve_btn_update,
            approve_backdrop_update,
        )

    def _approve(approve_payload: str) -> Tuple:
        try:
            payload = json.loads(approve_payload)
        except Exception:
            payload = {}

        state.approved = True
        holder["final_output"] = payload if isinstance(payload, dict) else {"output": payload}
        try:
            state.final_output_json = json.dumps(holder["final_output"], ensure_ascii=False, indent=2)
        except Exception:
            state.final_output_json = _pretty(holder["final_output"])

        _log("approved", final_output=holder["final_output"])
        approved_event.set()

        return (
            gr.update(value="Approved. Pipeline is running — keep this tab open."),
            state.final_output_json,
            gr.update(visible=False),
            gr.update(visible=False),
        )

    # -------------------------
    # Team Lead-mediated Requests & Resources UI helpers
    # -------------------------

    def _request_key(kind: str, request_id: str) -> str:
        return f"{kind}:{request_id}" if request_id else ""

    def _parse_request_key(key: str) -> Tuple[str, str]:
        text = str(key or "")
        if ":" in text:
            k, rid = text.split(":", 1)
            return k.strip(), rid.strip()
        # Backward fallback: prefer resource if present.
        if _get_resource_request(text):
            return "resource", text
        if _get_human_request(text):
            return "human", text
        return "", text

    def _request_items_text(req: Dict[str, Any]) -> str:
        items = req.get("items") if isinstance(req.get("items"), list) else []
        names: List[str] = []
        for it in items:
            if isinstance(it, dict):
                name = str(it.get("name") or "").strip()
                if name:
                    names.append(name)
            elif str(it).strip():
                names.append(str(it).strip())
        return ", ".join(names[:8]) if names else "the requested item"

    def _request_fingerprint(req: Dict[str, Any], kind: str) -> str:
        """Stable-ish fingerprint used to suppress duplicate UI surfacing.

        request_id alone is not enough: run_state fallback can resurrect a
        recently resolved request, while two agents can also ask the same raw
        question with different ids. This fingerprint is deliberately based on
        user-visible meaning, not volatile timestamps.
        """
        try:
            base = {
                "kind": kind,
                "task_id": req.get("task_id"),
                "stage": req.get("stage"),
                "reason": req.get("reason"),
                "question": req.get("question"),
                "items": req.get("items"),
                "user_facing": req.get("user_facing"),
            }
            raw = json.dumps(base, ensure_ascii=False, sort_keys=True, default=str)
        except Exception:
            raw = f"{kind}|{req.get('task_id')}|{req.get('stage')}|{req.get('reason')}|{req.get('question')}|{_request_items_text(req)}"
        return str(abs(hash(raw)))

    def _request_already_resolved(kind: str, req_id: str) -> bool:
        req_id = str(req_id or "").strip()
        if not req_id:
            return False
        try:
            if kind == "resource" and res_eval is not None:
                return isinstance(res_eval.read_resolution(req_id), dict)
            if kind == "human" and human_desk is not None:
                return isinstance(human_desk.read_resolution(req_id), dict)
        except Exception:
            return False
        return False

    def _looks_like_explanation_request(text: str) -> bool:
        msg = str(text or "").strip().lower()
        if not msg:
            return False
        markers = (
            "i don't understand", "i dont understand", "do not understand",
            "what is this", "what does this mean", "what are you asking",
            "can you explain", "explain this", "explain it", "in simple", "simple way",
            "plain english", "plain language", "why do you need", "why is this needed",
            "i'm confused", "im confused", "confused", "what should i do",
        )
        return any(m in msg for m in markers)

    def _looks_like_hard_stop(text: str) -> bool:
        msg = str(text or "").strip().lower()
        if not msg:
            return False
        hard_markers = (
            "stop the whole task", "stop this whole task", "cancel the task",
            "cancel this task", "halt everything", "halt the whole task",
            "do not continue this task", "kill this task", "hard stop",
        )
        return any(m in msg for m in hard_markers)

    def _plain_language_request_explanation(req: Dict[str, Any], kind: str) -> str:
        title = _request_title(req, kind)
        task_id = str(req.get("task_id") or "").strip()
        task_line = f" It is attached to task `{task_id}`." if task_id else ""
        if kind == "resource":
            items = _request_items_text(req)
            uf = req.get("user_facing") if isinstance(req.get("user_facing"), dict) else {}
            rec = str(uf.get("recommended_action") or "use fallback / proceed").replace("_", " ")
            return (
                f"Plain English: the internal team is asking whether you can provide **{items}**.{task_line} "
                "This does not automatically mean you personally must upload something. If these are logs, screenshots, generated files, test evidence, or anything the program can create/check by itself, the Team Lead should route around it internally. "
                f"Recommended route: **{rec}**. You can choose **Mark unavailable / use fallback** or **Block requested item / route around** and the workflow should continue with a placeholder, deferral, or reduced scope instead of asking you again."
            )
        q = str(req.get("question") or req.get("reason") or title)
        return (
            f"Plain English: the workflow is asking for a decision: **{q}**.{task_line} "
            "You do not need to translate this into agent language. Reply with what you want in normal words. "
            "If you block the requested item, the Team Lead should convert that into a workaround, reduced scope, or V2 deferral instead of freezing the whole project."
        )

    def _request_messages_with_user_reply(key_or_label: str, user_text: str, assistant_text: str) -> List[Dict[str, str]]:
        msgs = _team_lead_request_messages(key_or_label)
        if user_text:
            msgs.append({"role": "user", "content": user_text})
        if assistant_text:
            msgs.append({"role": "assistant", "content": assistant_text})
        return msgs

    def _request_title(req: Optional[Dict[str, Any]], kind: str) -> str:
        if not isinstance(req, dict):
            return "No active request"
        uf = req.get("user_facing") if isinstance(req.get("user_facing"), dict) else {}
        if uf.get("title"):
            return str(uf.get("title"))
        if kind == "human":
            q = str(req.get("question") or req.get("reason") or "Decision needed").strip()
            return q[:80] + ("…" if len(q) > 80 else "")
        items = req.get("items")
        if isinstance(items, list):
            names = [str(x.get("name")) for x in items if isinstance(x, dict) and x.get("name")]
            if names:
                return "; ".join(names[:2]) + ("…" if len(names) > 2 else "")
        return str(req.get("reason") or req.get("agent") or "Resource needed")

    def _combined_pending_requests() -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        seen_ids: set[str] = set()
        seen_fingerprints: set[str] = set()

        def add(kind: str, r: Dict[str, Any]) -> None:
            if not isinstance(r, dict) or not r.get("request_id"):
                return
            rid = str(r.get("request_id"))
            id_key = f"{kind}:{rid}"
            if id_key in seen_ids or _request_already_resolved(kind, rid):
                return
            fp = f"{kind}:{_request_fingerprint(r, kind)}"
            if fp in seen_fingerprints:
                return
            seen_ids.add(id_key)
            seen_fingerprints.add(fp)
            key = _request_key(kind, rid)
            out.append({"key": key, "kind": kind, "request_id": rid, "request": r})

        for r in _pending_resource_requests():
            add("resource", r)
        for r in _pending_human_requests():
            add("human", r)
        return out

    def _request_choices() -> Tuple[List[str], Optional[str]]:
        pending = _combined_pending_requests()
        choices = []
        for x in pending:
            req = x.get("request") if isinstance(x, dict) else {}
            label = f"{x.get('key')} — {_request_title(req, str(x.get('kind') or ''))}"
            choices.append(label)
        return choices, (choices[0] if choices else None)

    def _lookup_combined_request(key_or_label: str) -> Tuple[str, str, Optional[Dict[str, Any]]]:
        key = str(key_or_label or "").split(" — ", 1)[0].strip()
        kind, req_id = _parse_request_key(key)
        if kind == "resource":
            return kind, req_id, _get_resource_request(req_id)
        if kind == "human":
            return kind, req_id, _get_human_request(req_id)
        return "", req_id, None

    def _team_lead_request_messages(key_or_label: str = "") -> List[Dict[str, str]]:
        kind, req_id, req = _lookup_combined_request(key_or_label)
        if not req:
            return [{"role": "assistant", "content": "No pending requests need user input. If the workflow is blocked, click refresh or inspect run_state/logs."}]
        title = _request_title(req, kind)
        plain = _plain_language_request_explanation(req, kind)
        if kind == "resource":
            uf = req.get("user_facing") if isinstance(req.get("user_facing"), dict) else {}
            ctx = req.get("context") if isinstance(req.get("context"), dict) else {}
            chain = ctx.get("escalation_chain") if isinstance(ctx.get("escalation_chain"), list) else []
            msg = str(uf.get("message") or "The team may need an external resource. Upload it only if you already have it; otherwise use the Team Lead recommendation to proceed with internal generation/placeholders.")
            rec = str(uf.get("recommended_action") or "use_team_lead_recommendation")
            chain_text = f"\n\nEscalation: {' → '.join(str(x) for x in chain)}" if chain else ""
            content = (
                f"**{title}**\n\n{msg}{chain_text}\n\nRecommended action: **{rec.replace('_', ' ')}**."
                f"\n\n{plain}"
                "\n\nUse **Submit uploaded file(s)** only for a real file you already have. Otherwise use **Mark unavailable / use fallback** or **Block requested item / route around**."
            )
        else:
            q = str(req.get("question") or "The workflow needs your decision.")
            opts = req.get("options") if isinstance(req.get("options"), list) else []
            opts_txt = "\n" + "\n".join(f"- {x}" for x in opts) if opts else ""
            content = (
                f"**{title}**\n\n{q}{opts_txt}"
                f"\n\n{plain}"
                "\n\nReply below in normal language. If you choose to block the requested item, Team Lead will route around it rather than freeze the run."
            )
        return [{"role": "assistant", "content": content}]

    def _request_raw_details(key_or_label: str = "") -> str:
        kind, _req_id, req = _lookup_combined_request(key_or_label)
        if not req:
            return ""
        payload = {"kind": kind, "request": req}
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _request_status_markdown() -> str:
        pending = _combined_pending_requests()
        if not pending:
            return "✅ **No pending user-facing requests.**"
        return f"⚠️ **{len(pending)} request(s) need review.** Team Lead has translated the active request below."

    # -------------------------
    # Dashboard tick
    # -------------------------
    def _tick():
        status_md, paused, done = _run_status()

        request_choices: List[str] = []
        request_val: Optional[str] = None
        request_chat: List[Dict[str, str]] = []
        request_raw = ""
        request_status = ""
        if res_eval is not None:
            try:
                request_choices, request_val = _request_choices()
                request_status = _request_status_markdown()
                request_chat = _team_lead_request_messages(request_val or "")
                request_raw = _request_raw_details(request_val or "")
            except Exception as e:
                request_status = f"[request refresh error] {e}"
                request_chat = [{"role": "assistant", "content": request_status}]

        artifacts = _scan_artifacts()
        log_tail = _tail_text(log_path) if log_path else ""

        return (
            status_md,
            gr.update(visible=paused),
            gr.update(visible=done),
            gr.update(visible=paused),
            gr.update(choices=artifacts),
            log_tail,
            request_status,
            gr.update(choices=request_choices, value=request_val),
            request_chat,
            request_raw,
        )

    # -------------------------
    # Build UI
    # -------------------------
    try:
        demo = gr.Blocks(title=title, css=_APPROVE_MODAL_CSS)
    except TypeError:
        demo = gr.Blocks(title=title)
    with demo as demo:
        gr.Markdown(f"# {title}")
        if description_md:
            gr.Markdown(description_md)

        status_md = gr.Markdown()
        paused_banner = gr.Markdown("⚠️ **Paused** (API quota/rate limit). Refill quota, then click **Resume**.", visible=False)
        done_banner = gr.Markdown("✅ **Finished** — outputs are in the artifacts list below.", visible=False)
        resume_btn = gr.Button("Resume", visible=False)

        approve_backdrop = gr.HTML("", visible=False, elem_id="approve_backdrop")

        with gr.Row():
            with gr.Column(scale=7):
                gr.Markdown("## Team Lead Chat")
                gr.Markdown("Main user-facing channel. Intake, clarifications, status explanations, and broad workflow guidance all route through the Team Lead.")
                initial_chat = [{"role": "assistant", "content": "I’m the Team Lead. Tell me what you want to build, and I’ll turn it into initial_input.json before routing it through PM → UX → Engineering Lead → Engineering/QA."}]
                # Team Lead chat / intake panel
                if agent is None:
                    gr.Markdown("Agent interaction is disabled for this session. Use this as a dashboard + request/action desk.")
                    try:
                        chat = gr.Chatbot(value=initial_chat, height=300, type="messages", elem_classes=["compact-scroll-small"])
                    except TypeError:
                        chat = gr.Chatbot(value=[], height=300, elem_classes=["compact-scroll-small"])
                    user_msg = gr.Textbox(label="Message to Team Lead", placeholder="Send a Team Lead directive for the running workflow…", lines=3, max_lines=6, interactive=True)
                    run_btn = gr.Button("Send to Team Lead", interactive=True)
                    clear_btn = gr.Button("Clear", interactive=True)
                    agent_input_box = gr.Code(value="{}", language="json", label="Internal Agent Input", interactive=False, visible=False)
                else:
                    # Chatbot compatibility across Gradio versions
                    try:
                        chat = gr.Chatbot(value=initial_chat, height=300, type="messages", elem_classes=["compact-scroll-small"])
                    except TypeError:
                        chat = gr.Chatbot(value=[], height=300, elem_classes=["compact-scroll-small"])

                    user_msg = gr.Textbox(
                        label="Message to Team Lead",
                        placeholder="Describe what you want to build, answer Team Lead questions, or add constraints…",
                        lines=3,
                        max_lines=6,
                        elem_classes=["compact-scroll-small"],
                    )
                    with gr.Row():
                        run_btn = gr.Button("Send / Generate", variant="primary")
                        clear_btn = gr.Button("Clear")

                    with gr.Accordion("Internal Team Lead input / latest output", open=False):
                        agent_input_box = gr.Code(
                            value=state.agent_input_json,
                            language="json",
                            label="Internal Agent Input",
                            interactive=bool(allow_edit_agent_input),
                            lines=8,
                            max_lines=12,
                            elem_classes=["compact-scroll"],
                        )
                        out_json = gr.Code(value="", language="json", label="Latest Team Lead Output", lines=8, max_lines=12, elem_classes=["compact-scroll"])
                        final_json = gr.Code(value="", language="json", label="Approved Output", lines=8, max_lines=12, elem_classes=["compact-scroll"])
                if agent is None:
                    with gr.Accordion("Internal Team Lead input / latest output", open=False):
                        out_json = gr.Code(value="", language="json", label="Latest Output", lines=8, max_lines=12, elem_classes=["compact-scroll"])
                        final_json = gr.Code(value="", language="json", label="Approved Output", lines=8, max_lines=12, elem_classes=["compact-scroll"])

                # Requests & Resources now belongs directly under Team Lead Chat.
                # It is an action desk/chat, not a raw JSON request wall.
                if enable_resource_desk and res_eval is not None:
                    gr.Markdown("## Requests & Resources Desk")
                    gr.Markdown("Team Lead translates agent requests here. Upload files only when the Team Lead says an actual external resource is needed.")
                    request_status = gr.Markdown(_request_status_markdown())
                    request_dd = gr.Dropdown(choices=[], label="Active request", value=None)
                    try:
                        request_chat = gr.Chatbot(value=[{"role": "assistant", "content": "No pending requests yet."}], height=220, type="messages", elem_classes=["compact-scroll-small"], label="Requests chat")
                    except TypeError:
                        request_chat = gr.Chatbot(value=[], height=220, elem_classes=["compact-scroll-small"], label="Requests chat")
                    uploader = gr.File(label="Upload resource file(s), only if needed", file_count="multiple")
                    request_note = gr.Textbox(label="Reply / note to Team Lead", placeholder="Answer the Team Lead request, approve the recommendation, or explain why no resource is available…", lines=2, max_lines=5, elem_classes=["compact-scroll-small"])
                    request_action = gr.Dropdown(
                        choices=[
                            "Use Team Lead recommendation / proceed",
                            "Submit uploaded file(s)",
                            "Answer / approve / explain",
                            "Mark unavailable / use fallback",
                            "Block requested item / route around",
                        ],
                        value="Use Team Lead recommendation / proceed",
                        label="Action",
                    )
                    with gr.Row():
                        request_submit_btn = gr.Button("Submit to Team Lead", variant="primary")
                        request_refresh_btn = gr.Button("Refresh requests", variant="secondary")
                    request_notice = gr.Markdown("")
                    with gr.Accordion("Raw request details for debugging", open=False):
                        request_raw_json = gr.Code(value="", language="json", label="Raw request", lines=8, max_lines=12, elem_classes=["compact-scroll"])

            with gr.Column(scale=5):
                gr.Markdown("## Agent Inspector")
                gr.Markdown("Select one internal agent/stage. Messages here are still routed through the Team Lead, then attached as structured directives for the selected agent/task.")
                inspector_agent = gr.Dropdown(choices=_AGENT_INSPECTOR_CHOICES, value="Team Lead / Intake", label="Selected agent / stage")
                inspector_status = gr.Markdown(_agent_status_markdown("Team Lead / Intake"))
                inspector_json = gr.Code(value=json.dumps(_agent_status_payload("Team Lead / Intake"), ensure_ascii=False, indent=2), language="json", label="Selected agent status / task packet", lines=8, max_lines=12, elem_classes=["compact-scroll"])
                inspector_msg = gr.Textbox(label="Guidance to selected agent (routed through Team Lead)", lines=2, max_lines=5, elem_classes=["compact-scroll-small"])
                with gr.Row():
                    inspector_send_btn = gr.Button("Send directive via Team Lead", variant="secondary")
                    inspector_refresh_btn = gr.Button("Refresh selected agent", variant="secondary")
                inspector_notice = gr.Markdown("")

                # Approval panel: compact right-column card. Always visible; button is enabled only when intake is truly final.
                with gr.Column(visible=True, elem_id="approve_modal") as approve_panel:
                    gr.Markdown("### ✅ Initial input approval")
                    gr.Markdown("Review appears here when Team Lead intake is final. The button enables only when approval is valid.")
                    approve_payload = gr.Code(value="", language="json", label="Initial input (review)", lines=8, max_lines=12, visible=False, elem_classes=["compact-scroll"])
                    approve_btn = gr.Button("Approve and start pipeline", variant="primary", interactive=False)

                approve_notice = gr.Markdown("", visible=True)

                # Artifacts stay on the right column. They are secondary to Team Lead chat and Requests Desk.
                gr.Markdown("## Artifacts")
                artifacts_dd = gr.Dropdown(choices=[], label="Select an artifact")
                artifact_preview = gr.Code(value="", language=None, label="Preview", lines=10, max_lines=14, elem_classes=["compact-scroll"])
                artifact_file = gr.File(label="Download")

        # Full-width log tail at the bottom: one long merged-cell style panel below all other UI boxes.
        gr.Markdown("## Log tail")
        log_box = gr.Textbox(value="", lines=12, max_lines=18, interactive=False, show_label=False, elem_classes=["compact-scroll"])

        # events: Team Lead chat / intake / dashboard directives
        run_btn.click(
            _call_agent,
            inputs=[user_msg, chat, agent_input_box],
            outputs=[chat, out_json, final_json, approve_panel, approve_payload, approve_btn, approve_backdrop],
        )
        clear_btn.click(lambda: ([], "", "", gr.update(visible=True), gr.update(value="", visible=False), gr.update(interactive=False, visible=True), gr.update(visible=False)), inputs=[], outputs=[chat, out_json, final_json, approve_panel, approve_payload, approve_btn, approve_backdrop])

        # approval
        approve_btn.click(_approve, inputs=[approve_payload], outputs=[approve_notice, final_json, approve_panel, approve_backdrop])

        # artifacts selection
        def _on_select_artifact(rel: str):
            txt, p = _load_artifact(rel)
            # try to set file for download
            return txt, p

        artifacts_dd.change(_on_select_artifact, inputs=[artifacts_dd], outputs=[artifact_preview, artifact_file])

        # dynamic Agent Inspector
        inspector_agent.change(_inspect_agent, inputs=[inspector_agent], outputs=[inspector_status, inspector_json])
        inspector_refresh_btn.click(_inspect_agent, inputs=[inspector_agent], outputs=[inspector_status, inspector_json])
        inspector_send_btn.click(_write_agent_inspector_directive, inputs=[inspector_agent, inspector_msg], outputs=[inspector_notice, inspector_json])

        # resume
        def _resume():
            msg = _write_resume_flag()
            return msg

        resume_btn.click(lambda: _resume(), inputs=[], outputs=[approve_notice])

        # compact Team Lead-mediated request/resource handlers
        if enable_resource_desk and res_eval is not None:
            def _refresh_requests_ui(selected_key: str = ""):
                choices, val = _request_choices()
                if selected_key:
                    selected_prefix = str(selected_key).split(" — ", 1)[0]
                    for c in choices:
                        if c.startswith(selected_prefix):
                            val = c
                            break
                chat_msgs = _team_lead_request_messages(val or "")
                raw = _request_raw_details(val or "")
                return _request_status_markdown(), gr.update(choices=choices, value=val), chat_msgs, raw

            def _load_request_ui(selected_key: str):
                return _team_lead_request_messages(selected_key or ""), _request_raw_details(selected_key or "")

            def _copy_uploaded_files_for_request(req_id: str, files: Any) -> List[Dict[str, Any]]:
                if not files:
                    return []
                import shutil as _shutil
                file_paths: List[str]
                if isinstance(files, list):
                    file_paths = [getattr(f, "name", None) or str(f) for f in files]
                else:
                    file_paths = [getattr(files, "name", None) or str(files)]

                base = Path(resource_dir) if resource_dir else None
                if base is None:
                    return []
                inbox = base / "inbox"
                inbox.mkdir(parents=True, exist_ok=True)

                desired: List[str] = []
                try:
                    req = _get_resource_request(req_id) or {}
                    items = req.get("items") or []
                    if isinstance(items, list):
                        desired = [str(it.get("name")) for it in items if isinstance(it, dict) and it.get("name")]
                except Exception:
                    desired = []

                provided_assets: List[Dict[str, Any]] = []
                for fp in file_paths:
                    try:
                        src = Path(fp)
                        if not src.exists():
                            continue
                        dst = inbox / src.name
                        if dst.exists():
                            dst = inbox / f"{dst.stem}_{int(time.time())}{dst.suffix}"
                        _shutil.copy2(str(src), str(dst))
                        asset_name = dst.name
                        for dn in desired:
                            if dn and dn.lower() in dst.name.lower():
                                asset_name = dn
                                break
                        provided_assets.append({
                            "name": asset_name,
                            "type": "file",
                            "path": f"inbox/{dst.name}",
                            "original_filename": src.name,
                        })
                    except Exception:
                        continue
                return provided_assets

            def _infer_user_note_overrides(note_text: str) -> Dict[str, Any]:
                msg = str(note_text or "").strip().lower()
                inferred: Dict[str, Any] = {}
                if not msg:
                    return inferred
                if any(x in msg for x in ("+3", "3 additional", "three additional", "grant 3", "grant +3")):
                    inferred["extra_attempts_granted"] = 3
                elif any(x in msg for x in ("+2", "2 additional", "two additional", "grant 2", "grant +2")):
                    inferred["extra_attempts_granted"] = 2
                elif any(x in msg for x in ("+1", "1 additional", "one additional", "grant 1", "grant +1")):
                    inferred["extra_attempts_granted"] = 1
                if any(x in msg for x in ("scope trim", "reduce scope", "local-only", "local only", "mvp", "defer", "v2")):
                    inferred.update({
                        "scope_trim_to_local_mvp": True,
                        "defer_enterprise_items_to_v2": True,
                        "route_around_nonessential_blockers": True,
                    })
                if any(x in msg for x in ("do not ask", "don't ask", "no ci", "github actions", "ci runner", "docker", "playwright", "ocr", "tesseract", "sse", "benchmark", "runbook")):
                    inferred["do_not_request_ci_runner_or_github_actions"] = True
                    inferred["defer_ocr_tesseract_ci_docker_playwright_sse_benchmarks_docs_to_v2"] = True
                return inferred

            def _submit_request_ui(selected_key: str, action: str, files: Any, note_text: str):
                kind, req_id, req = _lookup_combined_request(selected_key or "")
                if not req_id or not req:
                    return (
                        "No pending request found. Click Refresh requests and check the run status.",
                        _request_status_markdown(),
                        gr.update(choices=[], value=None),
                        _team_lead_request_messages(""),
                        "",
                    )
                action = str(action or "Use Team Lead recommendation / proceed")
                note_msg = str(note_text or "").strip()
                inferred_note_overrides = _infer_user_note_overrides(note_msg)
                try:
                    if _looks_like_explanation_request(note_msg) and action != "Submit uploaded file(s)":
                        explanation = _plain_language_request_explanation(req, kind) + "\n\nI kept the request pending and did not forward your clarification question to the internal agents."
                        return (
                            "Team Lead answered your clarification; no workflow decision was submitted.",
                            _request_status_markdown(),
                            gr.update(),
                            _request_messages_with_user_reply(selected_key, note_msg, explanation),
                            _request_raw_details(selected_key),
                        )

                    if kind == "resource":
                        provided_assets: List[Dict[str, Any]] = []
                        decision = "accept_limitation"
                        overrides: Dict[str, Any] = {"use_placeholder": True, "generate_fixtures": True, "team_lead_mediated": True}
                        user_message = note_msg or "Team Lead recommendation accepted: proceed without user upload unless a real external resource was provided."

                        if action == "Submit uploaded file(s)":
                            provided_assets = _copy_uploaded_files_for_request(req_id, files)
                            if not provided_assets:
                                return (
                                    "No valid uploaded files found. Upload a file or choose a proceed/fallback action.",
                                    _request_status_markdown(),
                                    gr.update(),
                                    _team_lead_request_messages(selected_key),
                                    _request_raw_details(selected_key),
                                )
                            decision = "provided"
                            overrides = {"team_lead_mediated": True}
                            user_message = note_msg or "User provided file(s) through Team Lead request desk."
                        elif action in {"Answer / approve", "Answer / approve / explain"}:
                            decision = "continue"
                            overrides = {"team_lead_mediated": True, "proceed_with_user_answer": True}
                            user_message = note_msg or "User approved/provided answer through Team Lead request desk."
                        elif action == "Mark unavailable / use fallback":
                            decision = "mark_unavailable"
                            blocked_names = [str(it.get("name")) for it in (req.get("items") or []) if isinstance(it, dict) and it.get("name")]
                            overrides = {"unavailable": True, "use_placeholder": True, "generate_fixtures": True, "defer_to_v2_if_needed": True, "do_not_request_again": blocked_names, "team_lead_mediated": True}
                            user_message = note_msg or "Requested resource is unavailable; proceed with fallback/placeholders or defer the affected part to V2."
                        elif action in {"Block task", "Block requested item / route around"}:
                            blocked_names = [str(it.get("name")) for it in (req.get("items") or []) if isinstance(it, dict) and it.get("name")]
                            if _looks_like_hard_stop(note_msg):
                                decision = "block"
                                overrides = {"user_blocked_task": True, "team_lead_mediated": True}
                                user_message = note_msg or "User explicitly requested a hard stop for the whole task."
                            else:
                                decision = "accept_limitation"
                                overrides = {
                                    "user_blocked_requested_item": True,
                                    "user_blocked_task": False,
                                    "route_around_blocked_item": True,
                                    "defer_to_v2_if_needed": True,
                                    "use_placeholder": True,
                                    "generate_fixtures": True,
                                    "do_not_request_again": blocked_names,
                                    "team_lead_mediated": True,
                                }
                                user_message = note_msg or "User blocked the requested item/resource only. Continue by routing around it, using a placeholder/mock, reducing scope, or deferring that part to V2."

                        overrides = {**overrides, **inferred_note_overrides}
                        res_eval.resolve(
                            request_id=req_id,
                            provided_assets=provided_assets,
                            decision=decision,
                            user_message=user_message,
                            overrides=overrides,
                        )
                        _write_fallback_resource_directive(req, decision=decision, user_message=user_message, provided_assets=provided_assets, overrides=overrides)
                        if decision == "block":
                            # Only a hard stop should keep the whole run blocked.
                            pass
                        else:
                            _clear_run_block_if_matching(req_id)
                        notice = f"Team Lead recorded resource action for {req_id}: {decision}."
                    else:
                        decision = "continue"
                        user_message = note_msg or "Team Lead recommendation accepted."
                        overrides_human: Dict[str, Any] = {"team_lead_mediated": True}
                        if action in {"Block task", "Block requested item / route around"}:
                            if _looks_like_hard_stop(note_msg):
                                decision = "block"
                                overrides_human = {"team_lead_mediated": True, "user_blocked_task": True}
                                user_message = note_msg or "User explicitly requested a hard stop for the whole task."
                            else:
                                decision = "accept_limitation"
                                overrides_human = {
                                    "team_lead_mediated": True,
                                    "user_blocked_requested_item": True,
                                    "user_blocked_task": False,
                                    "route_around_blocked_item": True,
                                    "defer_to_v2_if_needed": True,
                                    "requires_team_lead_alternative_route": True,
                                }
                                user_message = note_msg or "User blocked the requested item/decision only. Team Lead should propose or apply an alternate route, reduce scope, use a safe fallback, or defer the affected part to V2."
                        elif action == "Mark unavailable / use fallback":
                            decision = "accept_limitation"
                            overrides_human = {"team_lead_mediated": True, "unavailable": True, "use_placeholder": True, "route_around_blocked_item": True, "defer_to_v2_if_needed": True}
                            user_message = note_msg or "Requested item unavailable; proceed with fallback, reduced scope, or V2 deferral if safe."
                        elif action in {"Answer / approve", "Answer / approve / explain"}:
                            decision = "continue"
                            overrides_human = {"team_lead_mediated": True, "proceed_with_user_answer": True}
                            user_message = note_msg or "User approved/provided clarification."

                        if human_desk is None:
                            return (
                                "Human request desk is unavailable.",
                                _request_status_markdown(),
                                gr.update(),
                                _team_lead_request_messages(selected_key),
                                _request_raw_details(selected_key),
                            )
                        overrides_human = {**overrides_human, **inferred_note_overrides}
                        human_desk.resolve(request_id=req_id, decision=decision, user_message=user_message, overrides=overrides_human)
                        _write_fallback_human_directive(req, decision=decision, user_message=user_message, overrides=overrides_human)
                        if _human_decision_blocks_task(decision, user_message):
                            _mark_run_blocked_by_user(req, decision=decision, user_message=user_message)
                        else:
                            _clear_run_block_if_matching(req_id)
                        notice = f"Team Lead recorded decision for {req_id}: {decision}."

                    status, dd, chat_msgs, raw = _refresh_requests_ui("")
                    return notice, status, dd, chat_msgs, raw
                except Exception as e:
                    return (
                        f"Request submit failed: {e}",
                        _request_status_markdown(),
                        gr.update(),
                        _team_lead_request_messages(selected_key),
                        _request_raw_details(selected_key),
                    )

            request_refresh_btn.click(_refresh_requests_ui, inputs=[request_dd], outputs=[request_status, request_dd, request_chat, request_raw_json])
            request_dd.change(_load_request_ui, inputs=[request_dd], outputs=[request_chat, request_raw_json])
            request_submit_btn.click(_submit_request_ui, inputs=[request_dd, request_action, uploader, request_note], outputs=[request_notice, request_status, request_dd, request_chat, request_raw_json])

        # tick / auto-refresh
        tick_outputs = [status_md, paused_banner, done_banner, resume_btn, artifacts_dd, log_box]
        tick_outputs += [request_status, request_dd, request_chat, request_raw_json] if (enable_resource_desk and res_eval is not None) else []

        def _tick_adapter():
            tup = _tick()
            if enable_resource_desk and res_eval is not None:
                return tup[0], tup[1], tup[2], tup[3], tup[4], tup[5], tup[6], tup[7], tup[8], tup[9]
            return tup[0], tup[1], tup[2], tup[3], tup[4], tup[5]

        # Initial load only. High-frequency Gradio polling can keep buttons in a
        # disabled/running state during long agent runs. Use the explicit
        # Refresh status/resources button for resource/human intervention checks.
        demo.load(_tick_adapter, inputs=[], outputs=tick_outputs)


        # launch (compat across Gradio versions)
        try:
            demo.launch(
                server_name=host,
                server_port=port,
                prevent_thread_lock=True,
                inbrowser=bool(open_browser),
            )
        except TypeError:
            # Fallback: some versions don't accept prevent_thread_lock and/or inbrowser.
            def _bg_launch():
                # Try a few launch signatures for compatibility across Gradio versions
                launch_attempts = [
                    dict(server_name=host, server_port=port, inbrowser=bool(open_browser)),
                    dict(server_name=host, server_port=port),
                ]
                for kwargs in launch_attempts:
                    try:
                        demo.launch(**kwargs)
                        return
                    except TypeError:
                        continue
                    except Exception:
                        return

            threading.Thread(target=_bg_launch, daemon=True).start()
    _LIVE_DEMOS.append(demo)

    if block_until_approved and agent is not None and approvable is not None:
        approved_event.wait()

    final_output = holder.get("final_output") or {}
    rounds = int(state.rounds or 0)
    prev_id = state.previous_response_id

    if not keep_open:
        try:
            demo.close()
        except Exception:
            pass

    return SupervisionResult(
        final_output=final_output if isinstance(final_output, dict) else {"output": final_output},
        rounds=rounds,
        previous_response_id=prev_id,
        log_file=str(log_path) if log_path else None,
        session_id=sid,
    )
