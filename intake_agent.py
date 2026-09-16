# intake_agent.py
"""
Intake agent: interactively turn a vague user idea into a detailed, actionable `initial_input.json`.

This agent is intentionally *not* PM/Design/Eng planning. Its job is:
1) Ask the *minimum* set of high-leverage questions to remove ambiguity.
2) Maintain an evolving `initial_input` draft.
3) When it judges the draft is good enough, set `mode="FINAL"` so the supervisor can approve.

The orchestration (who gets the approved file next) is handled by operation.py.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from model_provider_router import create_response_for_stage
from response_parser import parse_json_response_dict


# -----------------------------
# Config
# -----------------------------

DEFAULT_MODEL = os.getenv("INTAKE_MODEL", "gpt-4.1-mini")

# -----------------------------
# JSON Schema (OpenAI strict)
# NOTE:
# - Every object MUST set additionalProperties:false
# - Every object MUST include required listing *all* keys in properties
# - Optionality is modeled via union types including "null"
# -----------------------------

INTAKE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "mode": {"type": "string", "enum": ["ASK", "FINAL"]},
        "readiness_score": {"type": "integer"},
        "initial_input": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "brief": {"type": "string"},
                "meta": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "product_goal": {"type": ["string", "null"]},
                        "target_users": {"type": ["array", "null"], "items": {"type": "string"}},
                        "platforms": {"type": ["array", "null"], "items": {"type": "string"}},
                        "constraints": {"type": ["array", "null"], "items": {"type": "string"}},
                        "success_metrics": {"type": ["array", "null"], "items": {"type": "string"}},
                        "non_goals": {"type": ["array", "null"], "items": {"type": "string"}},
                        "timeline": {"type": ["string", "null"]},
                        "tech_stack": {"type": ["array", "null"], "items": {"type": "string"}},
                        "integrations": {"type": ["array", "null"], "items": {"type": "string"}},
                        "analytics_events": {"type": ["array", "null"], "items": {"type": "string"}},
                        "security_privacy": {"type": ["array", "null"], "items": {"type": "string"}},
                        "repo": {"type": ["string", "null"]},
                        "design_refs": {"type": ["array", "null"], "items": {"type": "string"}},
                        "freeform_notes": {"type": ["string", "null"]},
                    },
                    "required": [
                        "product_goal",
                        "target_users",
                        "platforms",
                        "constraints",
                        "success_metrics",
                        "non_goals",
                        "timeline",
                        "tech_stack",
                        "integrations",
                        "analytics_events",
                        "security_privacy",
                        "repo",
                        "design_refs",
                        "freeform_notes",
                    ],
                },
            },
            "required": ["brief", "meta"],
        },
        "questions": {"type": "array", "items": {"type": "string"}},
        "assumptions": {"type": "array", "items": {"type": "string"}},
        "risks": {"type": "array", "items": {"type": "string"}},
        "approval_request": {"type": "string"},
        "notes_for_workflow_coordinator": {"type": "string"},
    },
    "required": [
        "mode",
        "readiness_score",
        "initial_input",
        "questions",
        "assumptions",
        "risks",
        "approval_request",
        "notes_for_workflow_coordinator",
    ],
}


SYSTEM_PROMPT = """You are the Intake Agent in a software-building agentic system.

Your ONLY job:
- Ask high-leverage questions to remove ambiguity in a user's software request.
- Maintain/upgrade a structured `initial_input` draft that is actionable for downstream agents.
- Decide when the draft is good enough to ship to the pipeline.

Rules:
- Do NOT output PM-to-design or PM-to-engineering instructions. No 'pm_to_*' keys. No build plans.
- Keep questions minimal: ask only what materially changes scope/architecture or acceptance criteria.
- But do NOT stop after one round by default. If you think it's ready after round 1, ask ONE final confirmation question:
  "Any other requirements, constraints, example inputs/outputs, or resources I should know before finalizing initial_input?"
  Then wait for the user's response before setting mode="FINAL".
- Each round: incorporate the user's latest answers into `initial_input`.
- If remaining uncertainty is small and non-blocking, set mode="FINAL" and questions=[].
- Otherwise set mode="ASK" and output a short list of remaining questions.

`brief` must always be a single concise paragraph describing what to build.
`meta` fields can be null if unknown, but try to fill them when implied by the conversation.
"""


def _make_user_prompt(user_message: str, draft: Optional[Dict[str, Any]]) -> str:
    draft_txt = json.dumps(draft.get("initial_input"), ensure_ascii=False, indent=2) if draft else "null"
    return f"""User message:
{user_message}

Current draft initial_input (may be null):
{draft_txt}

Update the draft and decide ASK vs FINAL.
"""


def intake_execute(
    *,
    user_message: str,
    draft: Optional[Dict[str, Any]],
    previous_response_id: Optional[str],
    model: str = DEFAULT_MODEL,
) -> Tuple[Dict[str, Any], Optional[str]]:
    req: Dict[str, Any] = {
        "model": model,
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _make_user_prompt(user_message, draft)},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "intake_output",
                "strict": True,
                "schema": INTAKE_SCHEMA,
            }
        },
    }
    if previous_response_id:
        req["previous_response_id"] = previous_response_id

    resp = create_response_for_stage("intake.generate", req, model_override=model)

    # Provider output should already be normalized by model_provider_router, but
    # browser/GPTWeb extraction can still include guard lines, code fences, or
    # small wrappers. Parse defensively here so valid JSON is not discarded.
    out_text = getattr(resp, "output_text", None)
    parsed: Optional[Dict[str, Any]] = None
    if isinstance(out_text, str) and out_text.strip():
        try:
            parsed = parse_json_response_dict(out_text)
        except Exception:
            parsed = None

    if parsed is None:
        # Fallback for OpenAI-like response objects that expose nested content.
        try:
            nested_text = resp.output[0].content[0].text  # type: ignore[attr-defined]
            parsed = parse_json_response_dict(nested_text)
        except Exception:
            parsed = None

    if not isinstance(parsed, dict):
        parsed = {
            "mode": "ASK",
            "readiness_score": 0,
            "initial_input": {"brief": "", "meta": {k: None for k in INTAKE_SCHEMA["properties"]["initial_input"]["properties"]["meta"]["properties"].keys()}},
            "questions": ["I couldn't parse my own output. Please restate your request in one sentence."],
            "assumptions": [],
            "risks": [],
            "approval_request": "Please answer the question above.",
            "notes_for_workflow_coordinator": "",
        }

    resp_id = getattr(resp, "id", None)
    return parsed, resp_id


# -----------------------------
# Agent wrapper
# -----------------------------

@dataclass
class IntakeAgent:
    name: str = "intake_agent"
    model: str = DEFAULT_MODEL

    def run(
        self,
        *,
        agent_input: Any,
        draft: Optional[Dict[str, Any]] = None,
        feedback: Optional[str] = None,
        previous_response_id: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """
        Supervise-UI compatible run().

        Expected agent_input:
          - str: treated as the user_message
          - dict: may contain {"user_message": "..."} (preferred)
        `feedback` is the user's reply in the supervision loop.
        """
        # Determine what the user said this round
        if feedback and feedback.strip():
            user_message = feedback.strip()
        elif isinstance(agent_input, str):
            user_message = agent_input.strip()
        elif isinstance(agent_input, dict):
            user_message = str(agent_input.get("user_message") or "").strip()
        else:
            user_message = ""

        if not user_message:
            user_message = "Describe what you want to build."

        # Optional seed: operation.py may pass an existing initial_input.json as a starting point
        if draft is None and isinstance(agent_input, dict):
            seed = agent_input.get("seed_initial_input")
            if isinstance(seed, dict) and seed:
                draft = {"initial_input": seed}

        out, resp_id = intake_execute(
            user_message=user_message,
            draft=draft,
            previous_response_id=previous_response_id,
            model=self.model,
        )
        return out, resp_id
