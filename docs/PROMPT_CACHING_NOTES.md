# Prompt Caching Notes

## Purpose

Prompt caching reduces OpenAI input-token cost and latency when OpenAI requests share the same long, exact prefix.

This package now supports prompt-cache policy through:

- `provider_config.py`
- `model_provider_router.py`
- `trace.jsonl` usage records

## Implementation

`provider_config.py` defines cache policy for OpenAI provider profiles:

- `prompt_cache_enabled`
- `prompt_cache_retention`
- stage-level `prompt_cache_key`

The router attaches these fields only for OpenAI-backed calls:

- `prompt_cache_key`
- `prompt_cache_retention`

If the installed OpenAI SDK rejects the prompt-cache parameters, the router logs the issue and retries the same call without those parameters. Normal execution is not blocked.

## Prompt structure rule

The cacheable prefix is not defined by a special `prefix` field. It is the actual repeated beginning of the OpenAI request.

Keep request structure stable:

```text
stable agent instructions
stable schema/format rules
stable behavior rules

dynamic compressed context
current task/user input
run-specific values
```

Do not place changing values such as `run_id`, task ID, resource decisions, or logs before static agent instructions.

## Trace records

`model_provider_router.py` records cache policy in trace events:

- `prompt_cache_key`
- `prompt_cache_retention`
- whether cache policy was requested/applied
- cached token count if returned by OpenAI usage

## Environment variables

```bash
OPENAI_PROMPT_CACHE_ENABLED=1
OPENAI_PROMPT_CACHE_RETENTION=in-memory
```

For supported models, the retention value may be changed to:

```bash
OPENAI_PROMPT_CACHE_RETENTION=24h
```

Use `in-memory` by default unless longer retention is clearly useful.
