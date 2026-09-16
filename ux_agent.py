# ux_agent.py
import json
import os
from typing import Optional, Tuple, Dict, Any, List

from model_provider_router import create_response_for_stage
from response_parser import parse_json_response_dict


UX_ROLE_DESCRIPTION = """\
1) You are a world-class UI/UX Designer (product design) with deep experience shipping high-performing consumer and B2B
   products. You are extremely resourceful, detail-oriented, and you make pragmatic decisions that engineers can build.

2) You are strong across the full industry skill set:
   - Product thinking: problem framing, north-star metrics, constraints, tradeoffs
   - UX: user research synthesis, personas, jobs-to-be-done, journey mapping, information architecture, flows
   - Interaction design: states, edge cases, error handling, empty/loading states, micro-interactions
   - UI: visual hierarchy, layout systems, typography, color, spacing, components, responsiveness
   - Design systems: tokens, components, variants, accessibility patterns, documentation
   - Accessibility: WCAG-minded behaviors (keyboard navigation, focus states, contrast, ARIA semantics)
   - Content design: UX writing, labels, empty/error messages, confirmation language
   - Handoff: crisp specs, acceptance criteria, event instrumentation suggestions, and implementation notes

3) You must produce implementation-ready specs. Your output must be precise enough that a strong engineer can implement
   without guessing. You do not produce vague “inspiration” or high-level fluff.

4) You are evidence-driven and tool-flexible:
   - Use tools available to you (web_search / file_search / code_interpreter / mcp) when they materially improve
     correctness, standards alignment, or completeness.
   - If you use external references, include them in the output's references list (title + url + what you used it for).

5) Scope discipline:
   - Follow the given brief and constraints. If requirements are missing or contradictory, do not invent them—ask clear
     questions in the output (open_questions) and provide a best-effort “assumption set” to unblock work.

6) Output format:
   - Return ONLY JSON matching the provided schema. No surrounding prose.

7) Asset/resource discipline:
   - Do not block the workflow for optional assets, favicons, sample puzzles, or OCR test images.
   - If such assets are useful, list them as deferred/generated fixtures in engineer_handoff or validation_plan, not as required user resources.
   - For MVPs, prefer generated placeholders and text fixtures over asking the user for files.

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


DEFAULT_MODEL = "gpt-5"

def _build_tools_from_env() -> List[Dict[str, Any]]:
    """
    Tools mirror common Agent Builder toggles:
      - web_search
      - file_search (requires OPENAI_VECTOR_STORE_ID)
      - code_interpreter
      - mcp (requires OPENAI_MCP_SERVER_URL)
    """
    tools: List[Dict[str, Any]] = []

    # Web search
    tools.append({"type": "web_search"})

    # Code interpreter (for quick calculations, token tables, etc.)
    tools.append(
        {"type": "code_interpreter", "container": {"type": "auto", "memory_limit": "4g"}}
    )

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
                "server_description": os.getenv(
                    "OPENAI_MCP_SERVER_DESC", "Remote MCP server"
                ),
                "server_url": mcp_url,
                "require_approval": os.getenv("OPENAI_MCP_REQUIRE_APPROVAL", "never"),
            }
        )

    return tools


# ----------------------------
# Output JSON schema (strict)
# ----------------------------

SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "design_summary",
        "assumptions",
        "open_questions",
        "user_experience",
        "ui_spec",
        "design_system",
        "engineer_handoff",
        "validation_plan",
        "references",
        "next_actions",
    ],
    "properties": {
        "design_summary": {
            "type": "object",
            "additionalProperties": False,
            "required": ["product_goal", "target_users", "platforms", "key_metrics"],
            "properties": {
                "product_goal": {"type": "string"},
                "target_users": {"type": "array", "items": {"type": "string"}},
                "platforms": {"type": "array", "items": {"type": "string"}},
                "key_metrics": {"type": "array", "items": {"type": "string"}},
            },
        },
        "assumptions": {"type": "array", "items": {"type": "string"}},
        "open_questions": {"type": "array", "items": {"type": "string"}},
        "user_experience": {
            "type": "object",
            "additionalProperties": False,
            "required": ["personas", "primary_flows", "information_architecture"],
            "properties": {
                "personas": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["name", "goal", "pain_points"],
                        "properties": {
                            "name": {"type": "string"},
                            "goal": {"type": "string"},
                            "pain_points": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                },
                "primary_flows": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["flow_name", "steps"],
                        "properties": {
                            "flow_name": {"type": "string"},
                            "steps": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                },
                "information_architecture": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["navigation", "screen_inventory"],
                    "properties": {
                        "navigation": {"type": "string"},
                        "screen_inventory": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                },
            },
        },
        "ui_spec": {
            "type": "object",
            "additionalProperties": False,
            "required": ["screens", "interaction_patterns", "a11y_requirements"],
            "properties": {
                "screens": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "screen_name",
                            "purpose",
                            "layout",
                            "components",
                            "states",
                            "responsive_behavior",
                            "copy",
                        ],
                        "properties": {
                            "screen_name": {"type": "string"},
                            "purpose": {"type": "string"},
                            "layout": {"type": "string"},
                            "components": {"type": "array", "items": {"type": "string"}},
                            "states": {"type": "array", "items": {"type": "string"}},
                            "responsive_behavior": {"type": "string"},
                            "copy": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["headline", "labels", "empty_states", "errors"],
                                "properties": {
                                    "headline": {"type": "string"},
                                    "labels": {"type": "array", "items": {"type": "string"}},
                                    "empty_states": {"type": "array", "items": {"type": "string"}},
                                    "errors": {"type": "array", "items": {"type": "string"}},
                                },
                            },
                        },
                    },
                },
                "interaction_patterns": {"type": "array", "items": {"type": "string"}},
                "a11y_requirements": {"type": "array", "items": {"type": "string"}},
            },
        },
        "design_system": {
            "type": "object",
            "additionalProperties": False,
            "required": ["tokens", "components", "motion"],
            "properties": {
                "tokens": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["spacing", "typography", "color", "radii"],
                    "properties": {
                        "spacing": {"type": "string"},
                        "typography": {"type": "string"},
                        "color": {"type": "string"},
                        "radii": {"type": "string"},
                    },
                },
                "components": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["name", "variants", "props", "states"],
                        "properties": {
                            "name": {"type": "string"},
                            "variants": {"type": "array", "items": {"type": "string"}},
                            "props": {"type": "array", "items": {"type": "string"}},
                            "states": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                },
                "motion": {"type": "array", "items": {"type": "string"}},
            },
        },
        "engineer_handoff": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "acceptance_criteria",
                "verification",
                "instrumentation_events",
                "assets_needed",
                "edge_cases",
                "non_goals",
            ],
            "properties": {
                "acceptance_criteria": {"type": "array", "items": {"type": "string"}},
                "verification": {"type": "array", "items": {"type": "string"}},
                "instrumentation_events": {"type": "array", "items": {"type": "string"}},
                "assets_needed": {"type": "array", "items": {"type": "string"}},
                "edge_cases": {"type": "array", "items": {"type": "string"}},
                "non_goals": {"type": "array", "items": {"type": "string"}},
            },
        },
        "validation_plan": {
            "type": "object",
            "additionalProperties": False,
            "required": ["usability_tests", "success_criteria", "known_risks"],
            "properties": {
                "usability_tests": {"type": "array", "items": {"type": "string"}},
                "success_criteria": {"type": "array", "items": {"type": "string"}},
                "known_risks": {"type": "array", "items": {"type": "string"}},
            },
        },
        "references": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["title", "url", "used_for"],
                "properties": {
                    "title": {"type": "string"},
                    "url": {"type": "string"},
                    "used_for": {"type": "string"},
                },
            },
        },
        "next_actions": {"type": "array", "items": {"type": "string"}},
    },
}


def ux_designer_execute(
    design_brief: Dict[str, Any],
    *,
    repo_context: Optional[Dict[str, Any]] = None,
    model: str = DEFAULT_MODEL,
    previous_response_id: Optional[str] = None,
) -> Tuple[Dict[str, Any], str]:
    """
    Run the UX/UI designer agent and return (outputs, response_id).

    design_brief: the product/design request. Recommend including:
      - product_context, problem, target_users, platforms
      - constraints (brand, tech, accessibility, timeline)
      - existing artifacts links (figma, screenshots), if any
      - success metrics / business goals
    repo_context: optional additional context (existing components, style guide, etc.)
    """
    if not isinstance(design_brief, dict):
        raise ValueError("design_brief must be a dict")

    user_parts: List[str] = []
    user_parts.append("DESIGN_BRIEF_JSON:\n" + json.dumps(design_brief, ensure_ascii=False))
    if repo_context:
        if not isinstance(repo_context, dict):
            raise ValueError("repo_context must be a dict if provided")
        user_parts.append("REPO_CONTEXT_JSON:\n" + json.dumps(repo_context, ensure_ascii=False))

    req: Dict[str, Any] = {
        "model": model,
        "input": [
            {"role": "system", "content": UX_ROLE_DESCRIPTION},
            {"role": "user", "content": "\n\n".join(user_parts)},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "ux_outputs",
                "strict": True,
                "schema": SCHEMA,
            }
        },
        "tools": _build_tools_from_env(),
    }

    if previous_response_id:
        req["previous_response_id"] = previous_response_id

    resp = create_response_for_stage("ux.generate", req, model_override=model)

    try:
        outputs = parse_json_response_dict(resp.output_text)
    except (json.JSONDecodeError, ValueError) as e:
        raise RuntimeError(
            f"Model output was not valid JSON: {e}\n\nRAW:\n{resp.output_text}"
        ) from e

    return outputs, resp.id


class UXDesignerAgent:
    """
    Thin wrapper so the workflow coordinator can treat this module like other agents.

    Expected input:
      {
        "design_brief": {...},              # required
        "repo_context": {...}               # optional
      }

    Returns:
      (outputs_json, response_id)
    """

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model

    def run(
        self,
        agent_input: Dict[str, Any],
        *,
        feedback: Optional[str] = None,
        previous_response_id: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], str]:
        if not isinstance(agent_input, dict):
            raise ValueError("agent_input must be a dict")

        if "design_brief" not in agent_input or not isinstance(agent_input["design_brief"], dict):
            raise ValueError("agent_input must contain a dict field: design_brief")

        repo_context = agent_input.get("repo_context")
        if repo_context is not None and not isinstance(repo_context, dict):
            raise ValueError("repo_context must be a dict or omitted")

        # Optional workflow context and feedback get appended into repo_context for traceability.
        repo_context = dict(repo_context or {})
        for key in ["asset_manifest", "resource_decision", "context_pack"]:
            if key in agent_input and agent_input[key] is not None:
                repo_context[key] = agent_input[key]
        if feedback:
            repo_context["human_override"] = feedback.strip()

        return ux_designer_execute(
            design_brief=agent_input["design_brief"],
            repo_context=repo_context if repo_context else None,
            model=self.model,
            previous_response_id=previous_response_id,
        )
