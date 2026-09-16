# coordinator_agent_core.py
import json
import os
from typing import Optional, Tuple, Dict, Any, List

from model_provider_router import create_response_for_stage
from response_parser import parse_json_response_dict

COORDINATOR_DESCRIPTION = """\
1. You are the Coordinator of an agentic system. You are domain-agnostic: you can run coordination for product building, finance/accounting, operations, research, compliance, and even medical-related workflows, without pretending to be the domain expert.

2. Your job is not to “do the work.” Your job is to make the system converge: clarify the objective, define the decision(s) to make, route tasks to the right roles, reconcile conflicts, and produce a final decision record plus next actions.

3. You are ruthless about structure. You force every workflow into: Goal → Constraints → Inputs → Outputs → Owners → Gates → Stop conditions. If any of these are missing, you stop and ask for what’s missing (as open questions).

4. You maintain a single source of truth per run: current objective, assumptions, constraints, artifacts, decisions, and open questions. You prevent stale or conflicting drafts from being treated as current.

5. You operate on gates, not endless discussion. You define when consensus is required, who has decision rights at that gate, what counts as a blocker, and what “done” means. You do not seek global agreement from everyone.

6. You enforce a standardized review contract from every agent: target (artifact/run), verdict (agree/disagree/block), issues (severity), suggestions, questions. You reject vague feedback unless it is explicitly “agree” with no issues.

7. You detect conflicts and missing dependencies. When two agents disagree, you isolate the exact disagreement, request the minimum clarifying info, and propose a resolution with explicit tradeoffs.

8. You decide routing, not contributors. You determine which role needs which artifact, and you pass only the necessary context (or a digest). You do not spam the whole team with everything.

9. You are strict about risk. For regulated/high-stakes areas (medical/legal/financial), you label uncertainty, require evidence, and escalate to a qualified human when appropriate.

10. You optimize for velocity with control: smallest safe plan, but enforced gates, acceptance criteria, and rollback plans.

11. You communicate as the system’s executive function: concise, prioritized, unambiguous. You output decisions, rationale, assignments, and deadlines. No essays.

12. Output rules: you always produce (a) a coordination command for the next owner(s), (b) a gate status indicator (consensus true/false per gate), (c) blockers (if any), and (d) updated action items with owners.

RESOURCE DECISIONS (IMPORTANT)
- Your input may include `resource_decision` with fields:
  - decision: "provided" | "continue" | "mark_unavailable" | "accept_limitation" | "denied" | "redirected"
  - user_message: user's correction / direction
  - overrides: may include `use_placeholder`, `generate_fixtures`, `route_around_blocked_item`, `defer_to_v2_if_needed`, and `do_not_request_again`
- If decision is "mark_unavailable", "accept_limitation", "denied", or "redirected":
  1) Treat `user_message` as authoritative.
  2) Do NOT request again any items listed in `overrides.do_not_request_again`.
  3) If route_around_blocked_item/use_placeholder/generate_fixtures/defer_to_v2_if_needed is true, proceed with a mock, internal generation, reduced scope, or V2 deferral and continue.
  4) Only re-request if it is truly impossible to proceed; if so, explain why in ONE sentence and request the minimal thing.
- A route-around/block-item directive is not a whole-task hard stop unless `overrides.user_blocked_task` is true.

"""

SYSTEM = f"""\
{COORDINATOR_DESCRIPTION}

OUTPUT RULES:
- Return ONLY valid JSON (no markdown, no commentary).
- Follow the JSON schema strictly.

TOOL RULES:
- You may use tools when needed (web/file search/code/MCP), but do not mention tools in the output.

COORDINATION RULES:
- Treat the provided inputs as the source of truth. Do not invent missing artifacts or reviews.
- If inputs are insufficient to declare consensus or route actions safely, set consensus=false and list open_questions.
- Consensus must only be true when the gate criteria are satisfied for the required decision-right roles.
"""

SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "gate": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "name": {"type": "string"},
                "status": {"type": "string"},  # e.g., "draft" | "in_review" | "blocked" | "passed"
                "decision_rights": {"type": "array", "items": {"type": "string"}},
                "consensus_rule": {"type": "string"},
            },
            "required": ["name", "status", "decision_rights", "consensus_rule"],
            "additionalProperties": False,
        },
        "consensus": {"type": "boolean"},
        "coordination_command": {"type": "string"},
        "blockers": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "severity": {"type": "string"},  # "blocker" | "major" | "minor"
                    "owner_role": {"type": "string"},
                    "target": {
                        "type": "object",
                        "properties": {
                            "artifact_id": {"type": "string"},
                            "producer_role": {"type": "string"},
                        },
                        "required": ["artifact_id", "producer_role"],
                        "additionalProperties": False,
                    },
                    "title": {"type": "string"},
                    "detail": {"type": "string"},
                },
                "required": ["severity", "owner_role", "target", "title", "detail"],
                "additionalProperties": False,
            },
        },
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "owner_role": {"type": "string"},
                    "task": {"type": "string"},
                    "acceptance_criteria": {"type": "string"},
                    "dependencies": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["owner_role", "task", "acceptance_criteria", "dependencies"],
                "additionalProperties": False,
            },
        },
        "decision_record": {"type": "string"},
        "open_questions": {"type": "array", "items": {"type": "string"}},
        "notes": {"type": "string"},
    },
    "required": [
        "gate",
        "consensus",
        "coordination_command",
        "blockers",
        "actions",
        "decision_record",
        "open_questions",
        "notes",
    ],
    "additionalProperties": False,
}



WORKFLOW_DECISION_SYSTEM = f"""\
{COORDINATOR_DESCRIPTION}

WORKFLOW DECISION MODE:
- You are being called during an abnormal workflow state, not for final handoff.
- Your job is to choose ONE next workflow action from the allowed_actions list.
- You must preserve the existing workflow's authority boundaries:
  * operation.py executes your decision and enforces hard attempt limits.
  * Engineering Orchestrator owns task queue mechanics.
  * QA owns task-level verdicts.
  * You only decide the next workflow move when the normal path is blocked or ambiguous.
- Do not invent artifacts, files, tests, or user decisions.
- If max_attempt_check.allowed is false, do not choose a rerun action.
- Prefer the smallest safe action.
- Return ONLY valid JSON matching the schema.
"""

WORKFLOW_DECISION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "action": {
            "type": "string",
            "enum": [
                "continue",
                "rerun_last_step",
                "rerun_specific_stage",
                "ask_user",
                "block",
                "finish",
            ],
        },
        "target_stage": {
            "type": "string",
            "description": "The stage to run next, or empty string if not applicable.",
        },
        "affected_task_id": {
            "type": "string",
            "description": "Task ID affected by the decision, or empty string if not task-specific.",
        },
        "reason": {"type": "string"},
        "required_input": {
            "type": "string",
            "description": "Minimal user/agent input required before proceeding, or empty string.",
        },
        "max_attempt_check": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "current_attempt": {"type": "integer"},
                "max_attempts": {"type": "integer"},
                "allowed": {"type": "boolean"},
            },
            "required": ["current_attempt", "max_attempts", "allowed"],
        },
        "notes": {"type": "string"},
    },
    "required": [
        "action",
        "target_stage",
        "affected_task_id",
        "reason",
        "required_input",
        "max_attempt_check",
        "notes",
    ],
}


def decide_workflow_action(
    *,
    abnormal_state: str,
    context_pack: Dict[str, Any],
    allowed_actions: List[str],
    model: str = "gpt-5",
    previous_response_id: Optional[str] = None,
    provider_stage: str = "coordinator.workflow_decision",
) -> Tuple[Dict[str, Any], str]:
    """
    Coordinator exception-manager mode.

    This does not replace the final-handoff Coordinator role. It is called only
    when operation.py detects an abnormal or ambiguous workflow state.
    """
    user_parts: List[str] = [
        f"ABNORMAL STATE:\n{abnormal_state.strip()}",
        "ALLOWED ACTIONS:\n" + json.dumps(allowed_actions, ensure_ascii=False),
        "CONTEXT PACK (source of truth):\n" + json.dumps(context_pack, ensure_ascii=False, indent=2),
        "TASK:\n"
        "- Choose exactly one action from allowed_actions.\n"
        "- Respect max_attempt_check. If rerun is not allowed, choose block or ask_user.\n"
        "- Keep the decision operational, not explanatory.\n"
        "Return ONLY the JSON object.",
    ]

    req: Dict[str, Any] = {
        "model": model,
        "instructions": WORKFLOW_DECISION_SYSTEM,
        "input": [{"role": "user", "content": "\n\n".join(user_parts)}],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "workflow_decision_outputs",
                "strict": True,
                "schema": WORKFLOW_DECISION_SCHEMA,
            }
        },
        "tools": _build_tools_from_env(),
    }

    if previous_response_id:
        req["previous_response_id"] = previous_response_id

    resp = create_response_for_stage(provider_stage, req, model_override=model)

    if not getattr(resp, "output_text", None):
        raise RuntimeError("Model returned empty output_text; cannot parse workflow decision JSON.")

    try:
        outputs = parse_json_response_dict(resp.output_text)
    except (json.JSONDecodeError, ValueError) as e:
        raise RuntimeError(f"Workflow decision output was not valid JSON: {e}\n\nRAW:\n{resp.output_text}") from e

    action = str(outputs.get("action", ""))
    if action not in set(allowed_actions):
        raise RuntimeError(f"Coordinator returned action {action!r}, which is not in allowed_actions={allowed_actions!r}.")

    return outputs, resp.id


DEFAULT_MODEL = "gpt-5"


def _build_tools_from_env() -> List[Dict[str, Any]]:
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


def coordinate(
    *,
    objective: str,
    context_pack: Dict[str, Any],
    model: str = DEFAULT_MODEL,
    previous_response_id: Optional[str] = None,
    provider_stage: str = "coordinator.gate_decision",
) -> Tuple[Dict[str, Any], str]:
    """
    Returns: (outputs_dict, response_id)
    outputs_dict follows SCHEMA.

    context_pack should contain what you have, e.g.:
      - run_id, round_id, start_ts
      - gate_name, decision_right_roles
      - latest_artifacts (dict by role)
      - reviews (list of standardized review objects)
      - constraints/assumptions
      - log_slice (optional)
    """
    user_parts: List[str] = [
        f"OBJECTIVE:\n{objective.strip()}",
        "CONTEXT PACK (source of truth):\n" + json.dumps(context_pack, ensure_ascii=False, indent=2),
        "TASK:\n"
        "- Produce a coordination command for the next owner(s).\n"
        "- Determine gate consensus (true/false) based on decision-right roles and the reviews provided.\n"
        "- If blocked, list blockers and the minimal action items to resolve.\n"
        "- If inputs are insufficient, set consensus=false and list open_questions.\n"
        "Return ONLY the JSON object.",
    ]

    req: Dict[str, Any] = {
        "model": model,
        "instructions": SYSTEM,
        "input": [{"role": "user", "content": "\n\n".join(user_parts)}],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "coordination_outputs",
                "strict": True,
                "schema": SCHEMA,
            }
        },
        "tools": _build_tools_from_env(),
    }

    if previous_response_id:
        req["previous_response_id"] = previous_response_id

    resp = create_response_for_stage(provider_stage, req, model_override=model)

    if not getattr(resp, "output_text", None):
        raise RuntimeError("Model returned empty output_text; cannot parse JSON.")

    try:
        outputs = parse_json_response_dict(resp.output_text)
    except (json.JSONDecodeError, ValueError) as e:
        raise RuntimeError(f"Model output was not valid JSON: {e}\n\nRAW:\n{resp.output_text}") from e

    return outputs, resp.id


class CoordinatorAgent:
    name = "Coordinator"

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.model = model

    def run(
        self,
        *,
        agent_input: Dict[str, Any],
        draft: Optional[Dict[str, Any]] = None,  # unused but kept for compatibility with your supervise loop signature
        feedback: Optional[str] = None,          # optional: additional human instruction for coordinator
        previous_response_id: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], str]:
        """
        agent_input expects:
          - objective (str) [required]
          - context_pack (dict) [required]
        Optional:
          - feedback (str) can be passed via function arg; will be appended into context_pack if provided.
        """
        if "objective" not in agent_input or not isinstance(agent_input["objective"], str):
            raise ValueError("agent_input must contain a string field: objective")
        if "context_pack" not in agent_input or not isinstance(agent_input["context_pack"], dict):
            raise ValueError("agent_input must contain a dict field: context_pack")

        context_pack = dict(agent_input["context_pack"])
        if feedback:
            context_pack["human_override"] = feedback.strip()

        objective_lower = agent_input["objective"].lower()
        provider_stage = str(agent_input.get("provider_stage") or context_pack.get("provider_stage") or (
            "coordinator.final_handoff"
            if ("handoff" in objective_lower or "summarize" in objective_lower or "summary" in objective_lower)
            else "coordinator.gate_decision"
        ))

        return coordinate(
            objective=agent_input["objective"],
            context_pack=context_pack,
            model=self.model,
            previous_response_id=previous_response_id,
            provider_stage=provider_stage,
        )

    def workflow_decision(
        self,
        *,
        agent_input: Dict[str, Any],
        previous_response_id: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], str]:
        """
        Exception-manager mode used during the workflow.

        This is separate from run(), which preserves the existing final-handoff
        and general coordination role.
        """
        abnormal_state = str(agent_input.get("abnormal_state") or "").strip()
        if not abnormal_state:
            raise ValueError("agent_input must contain abnormal_state")

        context_pack = agent_input.get("context_pack")
        if not isinstance(context_pack, dict):
            raise ValueError("agent_input must contain context_pack dict")

        allowed_actions = agent_input.get("allowed_actions") or []
        if not isinstance(allowed_actions, list) or not all(isinstance(x, str) for x in allowed_actions):
            raise ValueError("agent_input must contain allowed_actions as list[str]")

        provider_stage = str(agent_input.get("provider_stage") or "coordinator.workflow_decision")

        return decide_workflow_action(
            abnormal_state=abnormal_state,
            context_pack=context_pack,
            allowed_actions=allowed_actions,
            model=self.model,
            previous_response_id=previous_response_id,
            provider_stage=provider_stage,
        )
