# eng_lead_agent_core.py
import json
import os
from typing import Optional, Tuple, Dict, Any, List

from model_provider_router import create_response_for_stage
from response_parser import parse_json_response_dict

ENG_LEAD_DESCRIPTION = """\
1. You are an Engineering Lead at a top-tier tech company, Staff/Principal level, with 12+ years of experience shipping production systems at scale (0→1, growth, and mature platforms).

2. You are calm under pressure, low-ego, brutally clear, and you hate vague requirements. You force clarity: scope, interfaces, constraints, and what “done” means.

3. You own architecture and execution end-to-end: system design, API contracts, data models, distributed workflows, observability, security fundamentals, and deployment. You optimize for reliability, velocity, and maintainability.

4. You translate product intent into technical reality: break work into milestones, define acceptance criteria, expose tradeoffs, surface risks early, and protect the critical path.

5. You are obsessive about correctness and edge cases: idempotency, retries, timeouts, partial failures, data integrity, deterministic behavior, and backward compatibility.

6. You ship with discipline: tests where they matter (unit/integration/regression), tooling hygiene (lint/format/type checks), PR standards, and high-signal code reviews. No silent failures.

7. You think in performance and cost: streaming vs in-memory, batching, backpressure, concurrency limits, caching, predictable scaling, and capacity planning. You measure before guessing.

8. You design for operability: structured logs, metrics, traces, dashboards, alerting, runbooks, incident response, and postmortems. “Debuggability” is a product requirement.

9. You prioritize security and privacy by default: least privilege, secrets hygiene, PII-safe logging, encryption in transit/at rest, retention/deletion, and threat modeling proportional to risk.

10. You communicate like a leader: crisp technical docs, diagrams when needed, decisions written down, and alignment in plain language. You say “no” when needed and propose shippable alternatives.

11. You do not overbuild. You choose the smallest architecture that can survive reality, then evolve it with evidence and intentional refactors.

12. You are a cross-functional multiplier. You work tightly with the Product Manager, UX designer, and engineers. You review deliverables, find issues, and give concrete corrections. You unblock fast and ensure everything fits one coherent system.

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
{ENG_LEAD_DESCRIPTION}

OUTPUT RULES:
- Return ONLY valid JSON (no markdown, no commentary).
- Follow the JSON schema strictly.

ROLE RULES:
- Your job is to (1) critique upstream deliverables, (2) make engineering decisions/tradeoffs, and
  (3) produce crisp build instructions for engineers below you (tasks, acceptance criteria, sequencing).
- Task graph quality is mandatory. The orchestrator will reject invalid work_items.
- `files_expected` must contain only real relative paths or clean directory paths ending in `/`.
- Do NOT use glob patterns (`*`, `?`, `[ ]`) in `files_expected`.
- Do NOT add annotation text inside paths, e.g. use `frontend/src/services/api.ts`, not `frontend/src/services/api.ts (ocr)`.
- If two tasks modify the same file, add a dependency so they cannot run in parallel.
- Use exact file paths for concrete files. Use directory paths only for broad setup scopes, and end them with `/`.
- For scaffolded empty directories, authorize `.gitkeep` placeholders inside those directory scopes by default. If a task says folders need placeholders, either list the exact `.gitkeep` paths or make the directory scopes end with `/` so `.gitkeep` inside them is in scope. Do not ask the user to choose between empty directories and `.gitkeep`; use `.gitkeep`.
- Treat favicon, OCR sample images, and sample puzzles as generated/deferred fixtures unless the user explicitly provided assets.

MVP / SIMPLE LOCAL APP RULES:
- If upstream initial_input/meta says delivery_mode or operating mode is `mvp_local_simple_web_app`, or says the project is a simple local MVP, obey that scale strictly.
- For MVP local apps, produce no more than 6 work_items and keep the first build focused on the core runnable flow.
- For MVP local apps, do NOT include CI, GitHub Actions, Docker, Playwright/e2e tests, SSE/live streaming, OCR/Tesseract, benchmark scripts, production observability, deployment, architecture docs, runbooks, performance docs, or large QA docs unless the user explicitly asks for them as V1 requirements.
- For MVP local apps, keep documentation to at most one local README/run instruction file.
- For MVP local apps, if an advanced feature appears in the user's broader idea but is marked deferred/non-goal/V2, put it in scope_out and do not create work_items for it.

TOOL RULES:
- You may use tools when needed (web/file search/code/MCP), but do not mention tools in the output.
"""

# Output contract: role-agnostic task graph designed to scale from 1 engineer to N engineers.
# The Engineering Lead produces a set of work items with explicit dependencies and required capabilities.
# A separate scheduler/router (e.g., Coordinator) can then assign work_items to any available engineer agents.
SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "review_feedback": {"type": "string"},
        "architecture_plan": {"type": "string"},
        "execution_plan": {"type": "string"},
        "work_items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "task_id": {"type": "string"},
                    "summary": {"type": "string"},
                    "capabilities_required": {"type": "array", "items": {"type": "string"}},
                    "dependencies": {"type": "array", "items": {"type": "string"}},
                    "scope_in": {"type": "string"},
                    "scope_out": {"type": "string"},
                    "interfaces": {"type": "array", "items": {"type": "string"}},
                    "acceptance_criteria": {"type": "array", "items": {"type": "string"}},
                    "verification": {"type": "array", "items": {"type": "string"}},
                    "files_expected": {"type": "array", "items": {"type": "string"}},
                    "risk_notes": {"type": "string"},
                },
                "required": [
                    "task_id",
                    "summary",
                    "capabilities_required",
                    "dependencies",
                    "scope_in",
                    "scope_out",
                    "interfaces",
                    "acceptance_criteria",
                    "verification",
                    "files_expected",
                    "risk_notes",
                ],
                "additionalProperties": False,
            },
        },
        "risks_and_mitigations": {"type": "string"},
        "open_questions": {"type": "string"},
        "definition_of_done": {"type": "string"},
    },
    "required": [
        "review_feedback",
        "architecture_plan",
        "execution_plan",
        "work_items",
        "risks_and_mitigations",
        "open_questions",
        "definition_of_done",
    ],
    "additionalProperties": False,
}

DEFAULT_MODEL = "gpt-5"


def _build_tools_from_env() -> List[Dict[str, Any]]:
    """
    “Full tools” (mirrors common Agent Builder toggles):
      - web_search
      - file_search (requires OPENAI_VECTOR_STORE_ID)
      - code_interpreter
      - mcp (requires OPENAI_MCP_SERVER_URL)
    """
    tools: List[Dict[str, Any]] = []

    # Web search
    tools.append({"type": "web_search"})

    # Code interpreter
    tools.append(
        {"type": "code_interpreter", "container": {"type": "auto", "memory_limit": "4g"}}
    )

    # File search (vector store)
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


def eng_lead_generate(
    *,
    task: str,
    upstream: Optional[Dict[str, Any]] = None,
    draft: Optional[Dict[str, Any]] = None,
    feedback: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    previous_response_id: Optional[str] = None,
) -> Tuple[Dict[str, Any], str]:
    """
    Returns: (outputs_dict, response_id)
    outputs_dict follows SCHEMA.
    """
    user_parts: List[str] = []
    user_parts.append(f"TASK:\n{task.strip()}")

    if upstream is not None:
        user_parts.append(
            "UPSTREAM INPUTS (from other agents / humans):\n"
            + json.dumps(upstream, ensure_ascii=False, indent=2)
        )

    if draft is not None:
        user_parts.append(
            "CURRENT DRAFT (revise this, do not ignore):\n"
            + json.dumps(draft, ensure_ascii=False, indent=2)
        )

    if feedback:
        user_parts.append("HUMAN FEEDBACK (must address):\n" + feedback.strip())

    user_parts.append(
        "DELIVERABLE:\n"
        "Return ONLY the JSON object that matches the schema. "
        "Be concrete: milestones, acceptance criteria, interfaces, edge cases, and a role-agnostic task graph (work_items) with capabilities + dependencies."
    )

    req: Dict[str, Any] = {
        "model": model,
        "instructions": SYSTEM,
        "input": [{"role": "user", "content": "\n\n".join(user_parts)}],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "eng_lead_outputs",
                "strict": True,
                "schema": SCHEMA,
            }
        },
        "tools": _build_tools_from_env(),
    }

    if previous_response_id:
        req["previous_response_id"] = previous_response_id

    resp = create_response_for_stage("eng_lead.generate_plan", req, model_override=model)

    if not getattr(resp, "output_text", None):
        raise RuntimeError("Model returned empty output_text; cannot parse JSON.")

    try:
        outputs = parse_json_response_dict(resp.output_text)
    except (json.JSONDecodeError, ValueError) as e:
        raise RuntimeError(
            f"Model output was not valid JSON: {e}\n\nRAW:\n{resp.output_text}"
        ) from e

    return outputs, resp.id

# ----------------------------
# Standardized review contract
# ----------------------------

ENG_REVIEW_SYSTEM = """\
You are acting as the Engineering Lead reviewer in a multi-agent workflow.

You will review the TARGET ARTIFACT provided (produced by another role). Your job:
- Determine whether the artifact is implementable, coherent, and safe (reliability, security, operability).
- Identify missing edge cases, feasibility gaps, and execution risks.
- Provide corrections that unblock execution.

VERDICT RULES:
- verdict="agree" only if there are no blocker/major issues.
- If verdict="agree", issues MUST be an empty list.
- verdict="disagree" if changes are needed but not fundamentally blocked.
- verdict="block" if the artifact would cause failure (feasibility, security, reliability) unless fixed.

OUTPUT RULES:
- Return ONLY valid JSON (no markdown, no commentary).
- Follow the JSON schema strictly.
"""

ENG_REVIEW_SCHEMA: Dict[str, Any] = {
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


def eng_lead_review(
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
        "TASK:\nReview the target artifact from an Engineering Lead perspective. "
        "Return ONLY the JSON review object."
    )

    req: Dict[str, Any] = {
        "model": model,
        "instructions": ENG_REVIEW_SYSTEM,
        "input": [{"role": "user", "content": "\n\n".join(parts)}],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "eng_lead_review",
                "strict": True,
                "schema": ENG_REVIEW_SCHEMA,
            }
        },
        "tools": _build_tools_from_env(),
    }

    if previous_response_id:
        req["previous_response_id"] = previous_response_id

    resp = create_response_for_stage("eng_lead.review", req, model_override=model)

    if not getattr(resp, "output_text", None):
        raise RuntimeError("Model returned empty output_text; cannot parse JSON.")

    try:
        review = parse_json_response_dict(resp.output_text)
    except (json.JSONDecodeError, ValueError) as e:
        raise RuntimeError(f"Model output was not valid JSON: {e}\n\nRAW:\n{resp.output_text}") from e

    return review, resp.id



# -------------------------
# Runbook (post-workflow instructions)
# -------------------------

ENG_RUNBOOK_SYSTEM = """You are the Engineering Lead. Produce a short, user-friendly runbook that tells a developer how to run the program.

Rules:
- Be straight to the point. No fluff.
- Include: prerequisites, environment setup, how to run (commands), where outputs appear, how to resume after a pause (quota), common failure modes.
- If something is unknown, state what is unknown and how to find it in the repo/output folder.
- Return ONLY the JSON object described by the schema.
"""

ENG_RUNBOOK_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "run_instructions_md": {"type": "string"},
        "assumptions": {"type": "array", "items": {"type": "string"}},
        "known_gaps": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["run_instructions_md", "assumptions", "known_gaps"],
}


def eng_lead_runbook(
    *,
    context_pack: Any,
    model: str = DEFAULT_MODEL,
    previous_response_id: Optional[str] = None,
) -> Tuple[Dict[str, Any], str]:
    parts: List[str] = []
    parts.append("CONTEXT PACK:\n" + json.dumps(context_pack, ensure_ascii=False, indent=2))

    parts.append(
        "TASK:\nWrite a runbook for running this program end-to-end. "
        "Return ONLY the JSON runbook object."
    )

    req: Dict[str, Any] = {
        "model": model,
        "instructions": ENG_RUNBOOK_SYSTEM,
        "input": [{"role": "user", "content": "\n\n".join(parts)}],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "eng_lead_runbook",
                "strict": True,
                "schema": ENG_RUNBOOK_SCHEMA,
            }
        },
        "tools": _build_tools_from_env(),
    }

    if previous_response_id:
        req["previous_response_id"] = previous_response_id

    resp = create_response_for_stage("eng_lead.runbook", req, model_override=model)

    if not getattr(resp, "output_text", None):
        raise RuntimeError("Model returned empty output_text; cannot parse JSON.")

    try:
        rb = parse_json_response_dict(resp.output_text)
    except (json.JSONDecodeError, ValueError) as e:
        raise RuntimeError(f"Model output was not valid JSON: {e}\n\nRAW:\n{resp.output_text}") from e

    return rb, resp.id


class EngineeringLeadAgent:
    name = "Engineering Lead"

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.model = model

    def run(
        self,
        *,
        agent_input: Dict[str, Any],
        draft: Optional[Dict[str, Any]] = None,
        feedback: Optional[str] = None,
        previous_response_id: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], str]:
        """
        agent_input is intentionally flexible. Expected keys:
          - task (str) [required]
          - upstream (dict) [optional]   # other agents' outputs / context
        """
        if "task" not in agent_input or not isinstance(agent_input["task"], str):
            raise ValueError("agent_input must contain a string field: task")

        upstream = agent_input.get("upstream", None)
        upstream_pack = dict(upstream) if isinstance(upstream, dict) else {}
        for key in ["asset_manifest", "resource_decision", "context_pack"]:
            if key in agent_input and agent_input[key] is not None:
                upstream_pack[key] = agent_input[key]
        return eng_lead_generate(
            task=agent_input["task"],
            upstream=upstream_pack or None,
            draft=draft,
            feedback=feedback,
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

        return eng_lead_review(
            target_artifact_id=str(agent_input["target_artifact_id"]),
            producer_role=str(agent_input["producer_role"]),
            target_payload=agent_input["target_payload"],
            objective=agent_input.get("objective"),
            gate=agent_input.get("gate"),
            model=self.model,
            previous_response_id=previous_response_id,
        )


    def runbook(
        self,
        *,
        agent_input: Dict[str, Any],
        previous_response_id: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], str]:
        """Generate a short runbook for how to run the produced program.

        agent_input expects:
          - context_pack (any JSON-serializable) OR context_pack nested inside agent_input.
        """
        context_pack = agent_input.get("context_pack", agent_input)
        return eng_lead_runbook(
            context_pack=context_pack,
            model=self.model,
            previous_response_id=previous_response_id,
        )
