# eng_agent.py
import json
import os
from typing import Optional, Tuple, Dict, Any, List

from model_provider_router import create_response_for_stage
from response_parser import parse_json_response_dict

ENG_DESCRIPTION = """\
1. You are a General Engineer: a senior-level software engineer with broad, deep experience shipping production systems.
   You can implement features, fix bugs, write tests, refactor safely, and deliver high-quality changes across many
   languages and stacks (Python, JS/TS, Java, C#, Go, Rust, C/C++, SQL, and common frameworks).

2. You operate strictly from the WorkItem contract. Your job is to produce a WorkResult that satisfies:
   - acceptance_criteria
   - verification (tests/commands) and their evidence
   - scope constraints (scope_in / scope_out)
   - interface contracts (schemas/APIs/invariants)
   If the WorkItem is ambiguous or contradictory, you MUST return questions and set the output accordingly.

3. You are evidence-based. Do not claim tests passed unless you have the output evidence or you explicitly state they
   were not run and provide exact commands for the verifier to run.

4. You are scope-disciplined. Do not expand scope. Do not refactor for taste. Touch the minimum files required, unless
   a small extra change is necessary to pass verification or prevent a clear regression.

5. You are tool-flexible. Use available tools when they improve correctness:
   - code_interpreter for quick validation, parsing diffs/logs, or running small checks
   - file_search (if configured) to retrieve repo context
   - web_search to confirm unstable/library-specific details (prefer primary docs)
   - mcp tools if available in this runtime
   If a tool is unavailable, proceed with best effort and provide clear verification steps.



REVISION MODE:
- If a DRAFT WorkResult is provided, treat it as your previous attempt.
  Apply FEEDBACK, fix defects, and return a revised WorkResult (same schema). Do not start over unless necessary.

OUTPUT RULES:
- Return ONLY valid JSON (no markdown, no commentary).
- Follow the JSON schema strictly.
- Do NOT invent test logs or execution results.
- If your task requires actual source-code files to be created or updated, put the full file contents in `code_output.files` and set `code_output.should_write_to_file` to true.
- If no local file should be written, set `code_output.should_write_to_file` to false and return an empty `code_output.files` array.
- Do not place machine-writable file content only inside `changes.content`; use `code_output.files` for file writing.

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

# ----------------------------
# Output schema (WorkResult)
# ----------------------------

SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "task_id": {"type": "string"},
        "engineer_id": {"type": "string"},
        "changes": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "format": {"type": "string", "enum": ["patch", "files", "description", "mixed"]},
                "content": {"type": "string"},
                "files_changed": {"type": "array", "items": {"type": "string"}},
                "summary_of_changes": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["format", "content", "files_changed", "summary_of_changes"],
        },
        "code_output": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "should_write_to_file": {"type": "boolean"},
                "files": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                            "write_mode": {"type": "string", "enum": ["create", "overwrite", "create_or_overwrite"]},
                        },
                        "required": ["path", "content", "write_mode"],
                    },
                },
                "notes": {"type": "string"},
            },
            "required": ["should_write_to_file", "files", "notes"],
        },
        "verification_run": {"type": "array", "items": {"type": "string"}},
        "notes": {"type": "string"},
        "handoff_interfaces": {"type": "array", "items": {"type": "string"}},
        "questions": {"type": "array", "items": {"type": "string"}},
        "assets_needed": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "task_id",
        "engineer_id",
        "changes",
        "code_output",
        "verification_run",
        "notes",
        "handoff_interfaces",
        "questions",
        "assets_needed",
    ],
}

DEFAULT_MODEL = os.getenv("OPENAI_ENGINEER_CODEX_MODEL", os.getenv("OPENAI_CODEX_MODEL", "gpt-5.3-codex"))


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
                "server_description": os.getenv(
                    "OPENAI_MCP_SERVER_DESC", "Remote MCP server"
                ),
                "server_url": mcp_url,
                "require_approval": os.getenv("OPENAI_MCP_REQUIRE_APPROVAL", "never"),
            }
        )

    return tools


def _select_engineer_stage(work_item: Dict[str, Any], repo_context: Optional[Dict[str, Any]], feedback: Optional[str]) -> str:
    """Choose the configured provider stage for engineer execution."""
    ctx = repo_context if isinstance(repo_context, dict) else {}
    explicit = ctx.get("provider_stage") or ctx.get("routing_stage")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    if feedback:
        return "engineer.fix_after_qa"
    risk_notes = str(work_item.get("risk_notes") or "").lower()
    required = " ".join(str(x).lower() for x in work_item.get("capabilities_required", []) if isinstance(x, str))
    if any(x in risk_notes + " " + required for x in ["architecture", "security", "database", "auth", "complex", "integration"]):
        return "engineer.execute_detailed"
    return "engineer.execute_standard"


def engineer_execute(
    *,
    work_item: Dict[str, Any],
    engineer_id: str,
    repo_context: Optional[Dict[str, Any]] = None,
    draft: Optional[Dict[str, Any]] = None,
    feedback: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    previous_response_id: Optional[str] = None,
) -> Tuple[Dict[str, Any], str]:
    """
    Execute the General Engineer agent on a single WorkItem.

    Inputs:
      - work_item: Engineering Lead WorkItem contract (task_id, scope, acceptance_criteria, verification, etc.)
      - engineer_id: worker identity (used by the orchestrator/queue)
      - repo_context: optional dict with repository notes, file lists, prior diffs, logs, etc.

    Returns:
      - outputs: dict matching SCHEMA
      - response_id: OpenAI response id (for tracing / continuation)
    """
    if not isinstance(work_item, dict):
        raise ValueError("work_item must be a dict")
    if not engineer_id or not isinstance(engineer_id, str):
        raise ValueError("engineer_id must be a non-empty string")

    ctx = repo_context or {}
    if ctx is not None and not isinstance(ctx, dict):
        raise ValueError("repo_context must be a dict or None")
    if draft is not None and not isinstance(draft, dict):
        raise ValueError("draft must be a dict or None")

    user_parts: List[str] = []
    user_parts.append("WORK_ITEM (source of truth):\n" + json.dumps(work_item, indent=2))
    if draft:
        user_parts.append("DRAFT_WORK_RESULT (previous attempt):\n" + json.dumps(draft, indent=2))
    if feedback:
        user_parts.append("FEEDBACK (must address):\n" + str(feedback).strip())
    user_parts.append("ENGINEER_ID:\n" + engineer_id)
    if ctx:
        user_parts.append("REPO_CONTEXT:\n" + json.dumps(ctx, indent=2))

    req: Dict[str, Any] = {
        "model": model,
        "instructions": ENG_DESCRIPTION,
        "input": [{"role": "user", "content": "\n\n".join(user_parts)}],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "engineer_work_result",
                "strict": True,
                "schema": SCHEMA,
            }
        },
        "tools": _build_tools_from_env(),
    }

    if previous_response_id:
        req["previous_response_id"] = previous_response_id

    resp = create_response_for_stage(_select_engineer_stage(work_item, ctx, feedback), req, model_override=model)

    if not getattr(resp, "output_text", None):
        raise RuntimeError("Model returned empty output_text; cannot parse JSON.")

    try:
        outputs = parse_json_response_dict(resp.output_text)
    except Exception as e:
        raise RuntimeError(
            f"Model output was not valid JSON: {e}\n\nRAW:\n{resp.output_text}"
        ) from e

    return outputs, resp.id


class EngineerAgent:
    name = "ENGINEER"
    """
    Thin wrapper for registry/router usage.

    Expected agent_input:
      - work_item: dict
      - engineer_id: str
      - repo_context: dict (optional)

    Returns:
      - dict matching SCHEMA
    """

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model

    def run(
        self,
        *,
        agent_input: Dict[str, Any],
        draft: Optional[Dict[str, Any]] = None,
        feedback: Optional[str] = None,
        previous_response_id: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], str]:
        if not isinstance(agent_input, dict):
            raise ValueError("agent_input must be a dict")

        if "work_item" not in agent_input or not isinstance(agent_input["work_item"], dict):
            raise ValueError("agent_input must contain a dict field: work_item")
        if "engineer_id" not in agent_input or not isinstance(agent_input["engineer_id"], str):
            raise ValueError("agent_input must contain a string field: engineer_id")

        repo_context = agent_input.get("repo_context")
        if repo_context is not None and not isinstance(repo_context, dict):
            raise ValueError("repo_context must be a dict or omitted")

        # Optional supervision inputs:
        # - draft: previous WorkResult (for iterative improvement)
        # - feedback: human/QA notes to address
        repo_context = dict(repo_context or {})
        if isinstance(agent_input.get("resource_decision"), dict):
            repo_context["resource_decision"] = agent_input.get("resource_decision")
        if isinstance(agent_input.get("human_decision"), dict):
            repo_context["human_decision"] = agent_input.get("human_decision")
        if draft:
            repo_context["draft_work_result"] = draft
        if feedback:
            repo_context["human_override"] = feedback.strip()

        return engineer_execute(
            work_item=agent_input["work_item"],
            engineer_id=agent_input["engineer_id"],
            repo_context=repo_context,
            draft=draft,
            feedback=feedback,
            model=self.model,
            previous_response_id=previous_response_id,
        )
