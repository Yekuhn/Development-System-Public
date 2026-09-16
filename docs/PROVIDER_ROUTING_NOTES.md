# Provider Routing Notes

## Purpose

This package uses a replaceable AI-reply provider layer for the Ascendant Path / VeRealm agent workflow.

Agents call:

```python
create_response_for_stage(stage_key, req, model_override=model)
```

The stage key is resolved through `provider_config.py`, and the selected backend is executed through `model_provider_router.py`.

## Supported providers

- OpenAI API
- Local Ollama
- GPTWeb/browser private website method
- Deterministic/no-LLM execution

## Run modes

Only two run modes are supported:

```bash
export ASCENDANT_RUN_MODE=power
```

or:

```bash
export ASCENDANT_RUN_MODE=development
```

### Power Mode

- OpenAI for all substantive AI stages
- Ollama for context compression/indexing/log compression
- deterministic stages remain deterministic

### Development Mode

- OpenAI for implementation agents: UX, Engineering Lead, Engineer, QA, runbook
- GPTWeb/browser for high-level/admin/management stages: Intake, PM, Coordinator handoff/gates, docs/summaries
- OpenAI for Coordinator workflow-decision mode because it may be invoked from engineering exception paths
- Ollama for context compression/indexing/log compression


## Engineer model routing

Engineer execution stages now use Codex-specialized OpenAI profiles by default:

- `engineer.execute_detailed` -> `openai_codex_engineer_high`
- `engineer.execute_standard` -> `openai_codex_engineer_standard`
- `engineer.execute_simple` -> `openai_codex_engineer_fast`
- `engineer.fix_after_qa` -> `openai_codex_engineer_standard`

The default Codex model is:

```bash
OPENAI_ENGINEER_CODEX_MODEL=gpt-5.3-codex
```

You can override it with either `OPENAI_ENGINEER_CODEX_MODEL` or `OPENAI_CODEX_MODEL`. QA and Engineering Lead remain on the general OpenAI reasoning profiles unless explicitly changed.

## Prompt caching

OpenAI-backed stages now carry prompt-cache policy from `provider_config.py`:

- `prompt_cache_enabled`
- `prompt_cache_key`
- `prompt_cache_retention`

`model_provider_router.py` attaches those fields to OpenAI calls and records cache usage in trace. If the installed OpenAI SDK does not accept prompt-cache parameters, the router logs a retry event and submits the same request without those parameters.

Prompt caching depends on stable request prefixes. Keep static agent instructions and schemas before dynamic run/task context.

## Important constraints

- Deterministic workflow stages must not call an LLM.
- Browser/GPTWeb is single-threaded, lock-protected, and text-first.
- Browser/GPTWeb must not be routed to parallel Engineer/QA worker-loop stages.
- Ollama/browser can be used for selected strict-JSON stages only through router-side JSON extraction/validation and fallback handling.
- OpenAI remains the safest backend for strict JSON, architecture, QA, implementation, and workflow-decision stages.

## Security note

The exported zip excludes `openai_api_key`. Use an environment variable instead:

```bash
export OPENAI_API_KEY="..."
```

or keep a local untracked `openai_api_key` file only on your machine.

## GPTWeb request-id guard

Development Mode still uses GPTWeb/browser for selected high-level/admin stages to control API cost. To reduce stale-output risk, browser calls now use a deterministic request-id guard:

```text
[GPTWEB_REQ_<unique_id>]
```

`model_provider_router.py` appends instructions requiring the web reply to begin with that exact prefix. The router validates the prefix, strips it before downstream parsing, and records the request id in response usage metadata.

If the prefix is missing, the router treats the response as possibly stale or wrong-context output, retries once by default, and then fails safely.

`gpt_web_collector.py` also now requires the webpage to produce a new assistant response count after prompt submission. Existing assistant text is no longer accepted as a valid new response.
