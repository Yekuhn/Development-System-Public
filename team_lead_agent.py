# team_lead_agent.py
"""
Team Lead agent for Ascendant Path.

This agent is the single user-facing coordinator. For v2 it owns the intake
phase directly while preserving the existing downstream workflow order:

    Team Lead intake -> PM -> UX -> Engineering Lead -> Orchestrator ->
    Engineer(s) -> QA -> Project Executor -> Runbook / final handoff

The Team Lead does not replace PM, UX, Engineering Lead, Engineer, or QA.
It converts the user's request into the same initial_input.json contract the
existing pipeline already expects, then hands that artifact to the current
internal agent chain.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from intake_agent import INTAKE_SCHEMA
from model_provider_router import create_response_for_stage
from response_parser import parse_json_response_dict


DEFAULT_MODEL = os.getenv("TEAM_LEAD_MODEL", os.getenv("INTAKE_MODEL", "gpt-4.1-mini"))


TEAM_LEAD_INTAKE_SYSTEM_PROMPT = """You are the Team Lead in an agentic software-building system.

You are the single user-facing coordinator for the run. In the intake phase,
you personally perform the previous Intake Agent role: clarify the user's
software request, maintain an evolving initial_input draft, and decide when the
draft is ready for the internal pipeline.

Your responsibilities during intake and workflow coordination:
- Communicate with the user directly and concretely.
- Ask only high-leverage questions that materially affect scope, architecture,
  acceptance criteria, resources, or constraints.
- Convert the user's messy intent into a structured `initial_input` object.
- Keep the workflow order intact: once initial_input is FINAL, operation.py will
  send it to PM, then UX, then Engineering Lead, then Engineering/QA.
- Do not perform PM, UX, engineering planning, coding, or QA in this output.
  You may note downstream concerns, but do not create build plans or pm_to_* keys.
- Treat targeted user comments as Team Lead directives that must be preserved in
  initial_input.meta.freeform_notes when they affect downstream work.
- Act as the human-facing interpreter of internal requests. Internal agents may
  ask for resources or decisions, but those requests should first be reviewed by
  the relevant domain lead and then by you. Ask the user only when the request is
  a real external asset, credential, dataset, business rule, or scope-changing
  decision. If the request can be generated, inspected, tested, or decided
  internally, do not burden the user with raw agent JSON.
- When a user-facing request is necessary, translate it into plain language:
  what is needed, why, what the recommended action is, and what happens if the
  user does not provide it.
- Never blindly forward a user reply such as "I don't understand" or "what is
  this?" to internal agents. Treat it as a clarification request, answer simply,
  and keep the original workflow request pending until the user chooses an
  actual decision.
- Treat "block this request/resource/item" as a route-around constraint by
  default, not as a stop-the-whole-project command. Propose or encode an
  alternative route: mock it, generate it internally, reduce scope, or defer that
  feature to V2. Only hard-stop the whole task when the user explicitly says to
  stop/cancel/halt the entire task.
- Do not ask the user for internal engineering evidence such as CI logs, curl
  output, local screenshots, missing source files, package-manager output,
  pre-commit output, or generated fixtures. Route those internally.

Protocol:
1. Receive the user message.
2. Update the initial_input draft.
3. If key information is missing, set mode="ASK" and ask the smallest useful set
   of questions.
4. If the draft is actionable, set mode="FINAL" with questions=[].
5. If this is the first round and the draft appears ready, prefer asking one
   final confirmation question before FINAL unless the user explicitly says to
   proceed.

`brief` must be a single concise paragraph describing what to build.
`meta` fields can be null if unknown, but fill them when implied.
Return only the JSON object matching the schema.
"""


def _make_user_prompt(user_message: str, draft: Optional[Dict[str, Any]]) -> str:
    draft_txt = json.dumps(draft.get("initial_input"), ensure_ascii=False, indent=2) if isinstance(draft, dict) else "null"
    return f"""User message:
{user_message}

Current draft initial_input (may be null):
{draft_txt}

Update the draft and decide ASK vs FINAL. Remember: you are the Team Lead doing
intake only; downstream PM/UX/Engineering Lead agents will run after FINAL.
"""


def team_lead_intake_execute(
    *,
    user_message: str,
    draft: Optional[Dict[str, Any]],
    previous_response_id: Optional[str],
    model: str = DEFAULT_MODEL,
) -> Tuple[Dict[str, Any], Optional[str]]:
    req: Dict[str, Any] = {
        "model": model,
        "input": [
            {"role": "system", "content": TEAM_LEAD_INTAKE_SYSTEM_PROMPT},
            {"role": "user", "content": _make_user_prompt(user_message, draft)},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "team_lead_intake_output",
                "strict": True,
                "schema": INTAKE_SCHEMA,
            }
        },
    }
    if previous_response_id:
        req["previous_response_id"] = previous_response_id

    # Reuse the existing intake provider route so the current provider_config.py
    # remains compatible. The role has changed; the output contract has not.
    # Team Lead intake should be a normal structured intake interaction, not a
    # visible GPTWeb role-prompt dump. Use the OpenAI-backed power route by
    # default; set TEAM_LEAD_INTAKE_RUN_MODE=development only if you explicitly
    # want GPTWeb for intake.
    resp = create_response_for_stage(
        "intake.generate",
        req,
        run_mode=os.getenv("TEAM_LEAD_INTAKE_RUN_MODE", "power"),
        model_override=model,
    )

    out_text = getattr(resp, "output_text", None)
    parsed: Optional[Dict[str, Any]] = None
    if isinstance(out_text, str) and out_text.strip():
        try:
            parsed = parse_json_response_dict(out_text)
        except Exception:
            parsed = None

    if parsed is None:
        try:
            nested_text = resp.output[0].content[0].text  # type: ignore[attr-defined]
            parsed = parse_json_response_dict(nested_text)
        except Exception:
            parsed = None

    if not isinstance(parsed, dict):
        try:
            meta_props = INTAKE_SCHEMA["properties"]["initial_input"]["properties"]["meta"]["properties"]
            meta_keys = list(meta_props.keys())
        except Exception:
            meta_keys = [
                "product_goal", "target_users", "platforms", "constraints",
                "success_metrics", "non_goals", "timeline", "tech_stack",
                "integrations", "analytics_events", "security_privacy",
                "repo", "design_refs", "freeform_notes",
            ]
        parsed = {
            "mode": "ASK",
            "readiness_score": 0,
            "initial_input": {"brief": "", "meta": {k: None for k in meta_keys}},
            "questions": ["I could not parse the Team Lead intake output. Please restate what you want to build in one sentence."],
            "assumptions": [],
            "risks": [],
            "approval_request": "Please answer the question above.",
            "notes_for_workflow_coordinator": "Team Lead intake parse fallback was used.",
        }

    return parsed, getattr(resp, "id", None)


@dataclass
class TeamLeadAgent:
    """Supervise-UI compatible Team Lead.

    In this version, run() is the Team Lead intake procedure. Later UI messages
    targeted at a selected agent should still be routed to Team Lead first and
    stored as structured directives, but this class preserves the existing
    operation.py intake contract.
    """

    name: str = "team_lead"
    model: str = DEFAULT_MODEL

    def run(
        self,
        *,
        agent_input: Any,
        draft: Optional[Dict[str, Any]] = None,
        feedback: Optional[str] = None,
        previous_response_id: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        if feedback and feedback.strip():
            user_message = feedback.strip()
        elif isinstance(agent_input, str):
            user_message = agent_input.strip()
        elif isinstance(agent_input, dict):
            user_message = str(agent_input.get("user_message") or "").strip()
        else:
            user_message = ""

        if draft is None and isinstance(agent_input, dict):
            seed = agent_input.get("seed_initial_input")
            if isinstance(seed, dict) and seed:
                draft = {"initial_input": seed}

        # Do not burn a model call or send the Team Lead role definition to the
        # browser/provider before the user has actually described the project.
        # The UI should start with a local Team Lead greeting and wait for user
        # intent. This keeps the intake phase user-driven instead of immediately
        # dispatching an internal role prompt to GPTWeb.
        if not user_message:
            try:
                meta_props = INTAKE_SCHEMA["properties"]["initial_input"]["properties"]["meta"]["properties"]
                meta_keys = list(meta_props.keys())
            except Exception:
                meta_keys = [
                    "product_goal", "target_users", "platforms", "constraints",
                    "success_metrics", "non_goals", "timeline", "tech_stack",
                    "integrations", "analytics_events", "security_privacy",
                    "repo", "design_refs", "freeform_notes",
                ]
            return ({
                "mode": "ASK",
                "readiness_score": 0,
                "initial_input": {"brief": "", "meta": {k: None for k in meta_keys}},
                "questions": ["What do you want to build? Include the main features, users, constraints, and any examples or resources I should know."],
                "assumptions": [],
                "risks": [],
                "approval_request": "Describe the project so I can prepare initial_input.json.",
                "notes_for_workflow_coordinator": "Team Lead intake has not started because no user project request was provided yet.",
            }, previous_response_id)

        return team_lead_intake_execute(
            user_message=user_message,
            draft=draft,
            previous_response_id=previous_response_id,
            model=self.model,
        )
