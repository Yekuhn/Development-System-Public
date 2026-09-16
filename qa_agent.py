# qa_agent_core.py
import json
import os
from typing import Optional, Tuple, Dict, Any, List

from model_provider_router import create_response_for_stage
from response_parser import parse_json_response_dict

QA_DESCRIPTION = """\
1. You are the QA Gate for an agentic engineering workflow.

2. Your job is not to write code. Your job is to judge whether an engineer's output satisfies the Engineering Lead's WorkItem contract.

3. You operate on evidence, not vibes. If required verification outputs are missing, you treat that as a failure to verify.

4. You are strict about scope. If the engineer changes files outside files_expected or violates scope_out, you flag it.

5. You are strict about interfaces/contracts. If an interface changed, you require explicit acknowledgement and updated tests.

6. You return a standardized review packet that a workflow-level Coordinator can consume.
   - target (artifact/run)
   - verdict (agree/disagree/block)
   - issues (severity, evidence, required_action)
   - suggestions (optional)
   - questions (optional)

7. Verdict rules:
   - agree: acceptance_criteria are met AND verification evidence is present.
   - disagree: not fully met, but fixable by the engineer without spec/architecture change.
   - block: fundamental failure (verification fails, scope violation, missing critical evidence, security/correctness risk),
            OR the WorkItem is ambiguous/contradictory and requires Engineering Lead clarification.

8. Output rules:
   - Return ONLY valid JSON.
   - Follow the JSON schema strictly.
   - Do not mention tools.


LOCAL FILE WRITER EVIDENCE
- If `staged_write_report.verification_evidence`, `directories_created`, `placeholder_files_created`, or `files_written` are present, treat them as valid deterministic evidence from the local file writer.
- Do NOT demand shell-only proof such as `tree`, `git status`, or `git log` when deterministic staged-write evidence already proves the required files/directories were created. You may suggest shell verification, but do not block solely for missing shell output.
- Empty directory placeholders may be represented by `.gitkeep` files created inside allowed directory scopes. Treat `.gitkeep` inside a `files_expected` directory ending in `/` as in-scope unless there is an explicit instruction against it.

RESOURCE AND HUMAN DECISIONS (IMPORTANT)
- Your input may include `resource_decision` and/or `human_decision`. Treat both as authoritative workflow inputs.
- Supported decision values include: "provided", "denied", "redirected", "continue", "accept_limitation", "defer", "route_around", and "block".
- `user_message` contains the user's correction/direction. `overrides` may include `use_placeholder`, `do_not_request_again`, `defer_to_v2_if_needed`, `accept_code_only_review`, `qa_waiver`, or `route_around_blocked_item`.
- If the human decision explicitly says to accept code-only review or defer runtime/manual evidence to a later gate, do NOT block solely for missing curl logs, stdout logs, screenshots, browser walkthroughs, or environment access. Flag the deferred evidence as a follow-on risk/suggestion instead.
- If `do_not_request_again` is true or names items, do not request those items again.
- Only re-request if it is truly impossible to proceed; if so, explain why in ONE sentence and request the minimal thing.

"""

SYSTEM = f"""\
{QA_DESCRIPTION}

OUTPUT RULES:
- Return ONLY valid JSON (no markdown, no commentary).
- Follow the JSON schema strictly.

QA RULES:
- Treat provided inputs as the source of truth.
- Do NOT invent test results, logs, or file contents.
- If evidence is missing for an acceptance criterion, mark it as an issue and set verdict at most 'disagree'.
- If verification is required but not provided, set verdict to 'block' unless a clear exception is stated in the WorkItem or an authoritative human_decision/resource_decision explicitly defers that evidence to a later gate.
- Deterministic local file writer evidence counts as verification evidence. Do not block only because shell command transcripts are absent when the local file writer report proves the file/directory state.
- Be concise. Every issue must include evidence and a concrete required_action.
"""

# QA output is designed to be appended into CoordinatorAgent's context_pack["reviews"].
# See coordinator_agent.py: it expects standardized review objects and uses them to decide gates.
SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "task_id": {"type": "string"},
        "review": {
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
                            "evidence": {"type": "string"},
                            "required_action": {"type": "string"},
                            "rerun_verification": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["severity", "title", "detail", "evidence", "required_action", "rerun_verification"],
                        "additionalProperties": False,
                    },
                },
                "suggestions": {"type": "array", "items": {"type": "string"}},
                "questions": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["target", "verdict", "issues", "suggestions", "questions"],
            "additionalProperties": False,
        },
        "routing_hint": {"type": "string", "enum": ["ENGINEER", "ENG_LEAD", "WORKFLOW_COORDINATOR"]},
        "queue_update": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "enum": ["mark_done", "mark_blocked", "return_to_queue"]},
                "blocked_reason": {"type": "string"},
            },
            "required": ["command", "blocked_reason"],
            "additionalProperties": False,
        },
        "notes": {"type": "string"},
        "assets_needed": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["task_id", "review", "routing_hint", "queue_update", "notes", "assets_needed"],
    "additionalProperties": False,
}

DEFAULT_MODEL = "gpt-5"


def _build_tools_from_env() -> List[Dict[str, Any]]:
    tools: List[Dict[str, Any]] = []

    # Code interpreter can be useful for structured diff/log checks when integrated.
    tools.append({"type": "code_interpreter", "container": {"type": "auto", "memory_limit": "4g"}})

    # File search (optional)
    vs_id = os.getenv("OPENAI_VECTOR_STORE_ID", "").strip()
    if vs_id:
        tools.append({"type": "file_search", "vector_store_ids": [vs_id]})

    # MCP server (optional)
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


def qa_review(
    *,
    work_item: Dict[str, Any],
    work_result: Dict[str, Any],
    verification_artifacts: Optional[Dict[str, Any]] = None,
    model: str = DEFAULT_MODEL,
    previous_response_id: Optional[str] = None,
) -> Tuple[Dict[str, Any], str]:
    """
    Returns: (outputs_dict, response_id)
    outputs_dict follows SCHEMA.

    The output is intended to be appended into the workflow Coordinator's context_pack["reviews"].

    Inputs:
      - work_item: Engineering Lead WorkItem contract
      - work_result: Engineer output (changes + verification_run + notes)
      - verification_artifacts: optional logs/test reports/lint output
    """
    verification_artifacts = verification_artifacts or {}

    # Keep the prompt deterministic and evidence-focused.
    user_parts: List[str] = [
        "WORK ITEM (contract):\n" + json.dumps(work_item, ensure_ascii=False, indent=2),
        "WORK RESULT (engineer output):\n" + json.dumps(work_result, ensure_ascii=False, indent=2),
        "VERIFICATION ARTIFACTS (logs/reports if any):\n" + json.dumps(verification_artifacts, ensure_ascii=False, indent=2),
        "TASK:\n"
        "- Evaluate the engineer output strictly against the WorkItem: scope, interfaces, acceptance_criteria, verification.\n"
        "- Produce ONE review packet in the required JSON schema.\n"
        "- Choose routing_hint: ENGINEER if fixable by engineer; ENG_LEAD if spec/architecture ambiguity; WORKFLOW_COORDINATOR if inputs/artifacts are malformed.\n"
        "- Set queue_update.command:\n"
        "  * mark_done if verdict=agree\n"
        "  * return_to_queue if verdict=disagree\n"
        "  * mark_blocked if verdict=block\n"
        "- blocked_reason must be non-empty; use 'done' for mark_done.\n"
        "Return ONLY the JSON object.",
    ]

    req: Dict[str, Any] = {
        "model": model,
        "instructions": SYSTEM,
        "input": [{"role": "user", "content": "\n\n".join(user_parts)}],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "qa_review_packet",
                "strict": True,
                "schema": SCHEMA,
            }
        },
        "tools": _build_tools_from_env(),
    }

    if previous_response_id:
        req["previous_response_id"] = previous_response_id

    resp = create_response_for_stage("qa.review_work_result", req, model_override=model)

    if not getattr(resp, "output_text", None):
        raise RuntimeError("Model returned empty output_text; cannot parse JSON.")

    try:
        outputs = parse_json_response_dict(resp.output_text)
    except (json.JSONDecodeError, ValueError) as e:
        raise RuntimeError(f"Model output was not valid JSON: {e}\n\nRAW:\n{resp.output_text}") from e

    return outputs, resp.id


class QAAgent:
    name = "QA"

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.model = model

    def run(
        self,
        *,
        agent_input: Dict[str, Any],
        draft: Optional[Dict[str, Any]] = None,  # unused; kept for compatibility
        feedback: Optional[str] = None,          # optional human override appended into artifacts
        previous_response_id: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], str]:
        """
        agent_input expects:
          - work_item (dict) [required]
          - work_result (dict) [required]
        Optional:
          - verification_artifacts (dict)

        feedback (str) if provided will be appended into verification_artifacts as human_override.
        """
        if "work_item" not in agent_input or not isinstance(agent_input["work_item"], dict):
            raise ValueError("agent_input must contain a dict field: work_item")
        if "work_result" not in agent_input or not isinstance(agent_input["work_result"], dict):
            raise ValueError("agent_input must contain a dict field: work_result")

        artifacts = agent_input.get("verification_artifacts")
        if artifacts is not None and not isinstance(artifacts, dict):
            raise ValueError("verification_artifacts must be a dict or omitted")

        artifacts = dict(artifacts or {})
        for key in ["asset_manifest", "resource_decision", "human_decision", "context_pack"]:
            if key in agent_input and agent_input[key] is not None:
                artifacts[key] = agent_input[key]
        if feedback:
            artifacts["human_override"] = feedback.strip()

        return qa_review(
            work_item=agent_input["work_item"],
            work_result=agent_input["work_result"],
            verification_artifacts=artifacts,
            model=self.model,
            previous_response_id=previous_response_id,
        )
