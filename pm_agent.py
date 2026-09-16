# pm_agent_core.py
import json
import os
from typing import Optional, Tuple, Dict, Any, List

from model_provider_router import create_response_for_stage
from response_parser import parse_json_response_dict

PM_ROLE_DESCRIPTION = """\
You are a product manager of the world’s biggest tech company with 20+ years of experience in product development, and you have successfully developed 10+ world-class products.

You are calm under pressure, low-ego, comfortable saying no, hate vague goals, and you are obsessed with what actually changes user behavior.

You do not omit details. You have unprecedented product intuition and a crazy ability to understand the client’s real need (including what they cannot articulate).

You also have strong product design capability: you can own end-to-end UX (information architecture, user flows, interaction design, UX writing, onboarding, and edge-case handling), and you can turn messy requirements into simple, high-converting experiences.

You can run lightweight user research and validation: define personas/JTBD, write test scripts, conduct interviews, interpret feedback without bias, and translate insights into concrete design decisions.

You can prototype fast (low-fidelity → high-fidelity), define design principles, maintain consistency with a design system mindset, and ensure accessibility/usability standards are met.

You think in experiments: define hypotheses, design A/B tests, choose the right success metrics, avoid misleading metrics, and drive iteration cycles based on evidence.

You work extremely well with engineering: you specify clearly, define acceptance criteria, anticipate technical constraints, and protect scope while still shipping fast with high quality.

RESOURCE DECISIONS (IMPORTANT)
- Your input may include `resource_decision` with fields:
  - decision: "provided" | "denied" | "redirected"
  - user_message: user's correction / direction
  - overrides: may include `use_placeholder` (bool) and `do_not_request_again` (list of item names)
- If decision is "denied" or "redirected":
  1) Treat `user_message` as authoritative.
  2) Do NOT request again any items listed in `overrides.do_not_request_again`.
  3) If `use_placeholder` is true, proceed using placeholders/mocks/neutral assets and continue.
  4) Only re-request if it is truly impossible to proceed; if so, explain why in ONE sentence and request the minimal thing.

"""

SYSTEM = f"""\
{PM_ROLE_DESCRIPTION}

OUTPUT RULES:
- Return ONLY valid JSON (no markdown, no commentary).
- Exactly two keys:
  - pm_to_design (string)
  - pm_to_eng (string)

CONTENT RULES:
- pm_to_design MUST include: goal, target user, primary flow, alt/error states, screen list, UX copy notes, open questions, definition of done.
- pm_to_eng MUST include: goal, non-goals, functional requirements, data/entities, API/events proposal, risks/edge cases, observability, rollout/rollback plan, definition of done.
- Be concrete and execution-oriented. No fluff.
"""

SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "pm_to_design": {"type": "string"},
        "pm_to_eng": {"type": "string"},
    },
    "required": ["pm_to_design", "pm_to_eng"],
    "additionalProperties": False,
}

DEFAULT_MODEL = "gpt-5"


def build_tools_from_env() -> List[Dict[str, Any]]:
    """
    Full tools (mirrors Agent Builder toggles):
      - web_search (always enabled)
      - code_interpreter (always enabled)
      - file_search (enabled if OPENAI_VECTOR_STORE_ID is set)
      - mcp (enabled if OPENAI_MCP_SERVER_URL is set)
    """
    tools: List[Dict[str, Any]] = []

    # Web search
    tools.append({"type": "web_search"})

    # Code interpreter
    tools.append({"type": "code_interpreter", "container": {"type": "auto", "memory_limit": "4g"}})

    # File search
    vs_id = os.getenv("OPENAI_VECTOR_STORE_ID", "").strip()
    if vs_id:
        tools.append({"type": "file_search", "vector_store_ids": [vs_id]})

    # MCP server
    mcp_url = os.getenv("OPENAI_MCP_SERVER_URL", "").strip()
    if mcp_url:
        tools.append(
            {
                "type": "mcp",
                "server_label": os.getenv("OPENAI_MCP_SERVER_LABEL", "mcp"),
                "server_description": os.getenv("OPENAI_MCP_SERVER_DESC", "Remote MCP server"),
                "server_url": mcp_url,
                "require_approval": os.getenv("OPENAI_MCP_REQUIRE_APPROVAL", "never"),
            }
        )

    return tools


def pm_generate(
    brief: str,
    *,
    draft: Optional[Dict[str, str]] = None,
    feedback: Optional[str] = None,
    context_pack: Optional[Dict[str, Any]] = None,
    model: str = DEFAULT_MODEL,
    previous_response_id: Optional[str] = None,
) -> Tuple[Dict[str, str], str]:
    """
    Returns: (outputs_dict, response_id)
    outputs_dict keys: pm_to_design, pm_to_eng
    """
    user_parts: List[str] = [f"PRODUCT BRIEF:\n{brief.strip()}"]

    if context_pack:
        user_parts.append(
            "WORKFLOW_CONTEXT_JSON (assets, resource decisions, and upstream constraints):\n"
            + json.dumps(context_pack, ensure_ascii=False, indent=2)
        )

    if draft is not None:
        user_parts.append(
            "CURRENT DRAFT (to revise, not to ignore):\n"
            + json.dumps(draft, ensure_ascii=False, indent=2)
        )

    if feedback:
        user_parts.append("HUMAN FEEDBACK (must address):\n" + feedback.strip())

    user_parts.append("TASK:\nRevise the draft (or produce the first draft) and return ONLY the JSON object.")

    req: Dict[str, Any] = {
        "model": model,
        "instructions": SYSTEM,
        "input": "\n\n".join(user_parts),
        "text": {
            "format": {
                "type": "json_schema",
                "name": "pm_outputs",
                "strict": True,
                "schema": SCHEMA,
            }
        },
        "tools": build_tools_from_env(),
    }

    if previous_response_id:
        req["previous_response_id"] = previous_response_id

    resp = create_response_for_stage("pm.generate", req, model_override=model)

    if not getattr(resp, "output_text", None):
        raise RuntimeError("Model returned empty output_text; cannot parse JSON.")

    try:
        outputs = parse_json_response_dict(resp.output_text)
    except (json.JSONDecodeError, ValueError) as e:
        raise RuntimeError(f"Model output was not valid JSON: {e}\n\nRAW:\n{resp.output_text}") from e

    return outputs, resp.id


# ----------------------------
# Standardized review contract
# ----------------------------

REVIEW_SYSTEM = """\
You are acting as the Product Manager reviewer in a multi-agent workflow.

You will review the TARGET ARTIFACT provided (produced by another role). Your job:
- Determine whether the artifact is acceptable for the current gate/objective.
- If you disagree, provide concrete, actionable issues and suggestions.

VERDICT RULES:
- verdict="agree" only if there are no blocker/major issues.
- If verdict="agree", issues MUST be an empty list.
- verdict="disagree" if changes are needed but not fundamentally blocked.
- verdict="block" if there are critical gaps that must be fixed before proceeding.

OUTPUT RULES:
- Return ONLY valid JSON (no markdown, no commentary).
- Follow the JSON schema strictly.
"""

REVIEW_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "target": {
            "type": "object",
            "properties": {
                "artifact_id": {"type": "string"},
                "producer_role": {"type": "string"},
            },
            "required": ["artifact_id", "producer_role"],
            "additionalProperties": False,
        },
        "verdict": {"type": "string", "enum": ["agree", "disagree", "block"]},
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "severity": {"type": "string", "enum": ["blocker", "major", "minor"]},
                    "title": {"type": "string"},
                    "detail": {"type": "string"},
                },
                "required": ["severity", "title", "detail"],
                "additionalProperties": False,
            },
        },
        "suggestions": {"type": "array", "items": {"type": "string"}},
        "questions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["target", "verdict", "issues", "suggestions", "questions"],
    "additionalProperties": False,
}


def pm_review(
    *,
    target_artifact_id: str,
    producer_role: str,
    target_payload: Any,
    objective: Optional[str] = None,
    gate: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    previous_response_id: Optional[str] = None,
) -> Tuple[Dict[str, Any], str]:
    """Return (review_dict, response_id)."""
    parts: List[str] = []
    if objective:
        parts.append(f"OBJECTIVE:\n{objective.strip()}")
    if gate:
        parts.append(f"GATE:\n{gate.strip()}")

    parts.append(
        "TARGET ARTIFACT (source of truth):\n"
        + json.dumps(
            {
                "artifact_id": target_artifact_id,
                "producer_role": producer_role,
                "payload": target_payload,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    parts.append(
        "TASK:\nReview the target artifact from a Product Manager perspective. "
        "Return ONLY the JSON review object."
    )

    req: Dict[str, Any] = {
        "model": model,
        "instructions": REVIEW_SYSTEM,
        "input": "\n\n".join(parts),
        "text": {
            "format": {
                "type": "json_schema",
                "name": "pm_review",
                "strict": True,
                "schema": REVIEW_SCHEMA,
            }
        },
        "tools": build_tools_from_env(),
    }

    if previous_response_id:
        req["previous_response_id"] = previous_response_id

    resp = create_response_for_stage("pm.review", req, model_override=model)

    if not getattr(resp, "output_text", None):
        raise RuntimeError("Model returned empty output_text; cannot parse JSON.")

    try:
        review = parse_json_response_dict(resp.output_text)
    except (json.JSONDecodeError, ValueError) as e:
        raise RuntimeError(f"Model output was not valid JSON: {e}\n\nRAW:\n{resp.output_text}") from e

    return review, resp.id


class PMAgent:
    name = "PM"

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.model = model

    def run(
        self,
        *,
        agent_input: Dict[str, Any],
        draft: Optional[Dict[str, str]] = None,
        feedback: Optional[str] = None,
        previous_response_id: Optional[str] = None,
    ) -> Tuple[Dict[str, str], str]:
        if "brief" not in agent_input or not isinstance(agent_input["brief"], str):
            raise ValueError("agent_input must contain a string field: brief")

        context_pack = {
            k: agent_input[k]
            for k in ["asset_manifest", "resource_decision", "initial_input", "context_pack"]
            if k in agent_input and agent_input[k] is not None
        }

        return pm_generate(
            brief=agent_input["brief"],
            draft=draft,
            feedback=feedback,
            context_pack=context_pack or None,
            model=self.model,
            previous_response_id=previous_response_id,
        )

    def review(
        self,
        *,
        agent_input: Dict[str, Any],
        previous_response_id: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], str]:
        """
        Standardized review entrypoint for the Coordinator.

        agent_input expects:
          - target_artifact_id (str)
          - producer_role (str)
          - target_payload (any JSON-serializable)
        Optional:
          - objective (str)
          - gate (str)
        """
        for k in ["target_artifact_id", "producer_role", "target_payload"]:
            if k not in agent_input:
                raise ValueError(f"agent_input missing required field: {k}")

        return pm_review(
            target_artifact_id=str(agent_input["target_artifact_id"]),
            producer_role=str(agent_input["producer_role"]),
            target_payload=agent_input["target_payload"],
            objective=agent_input.get("objective"),
            gate=agent_input.get("gate"),
            model=self.model,
            previous_response_id=previous_response_id,
        )
