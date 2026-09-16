# Trace Observability

This package adds lightweight trace observability for the agentic workflow.

## What it is

Trace observability is a structured JSONL record of internal workflow events.
It is not a UI and it does not change workflow decisions.

Trace file:

```text
outputs/<run_id>/trace.jsonl
```

`operation.py` sets:

```text
ASCENDANT_RUN_ID=<run_id>
ASCENDANT_TRACE_PATH=outputs/<run_id>/trace.jsonl
```

Other modules write trace events through `trace_utils.trace_event()`.
If `ASCENDANT_TRACE_PATH` is not set, tracing is a no-op.

## Events currently traced

- run observability initialization
- stage cache hit
- stage execution start/end/error/pause
- provider call start/end/error
- provider JSON fallback
- context compression result
- Engineer attempt start/end
- QA review start/end
- local file write result
- task submission
- Coordinator workflow decision start/end/error

## Why it exists

The model now has multiple moving parts:

- `provider_config.py`
- `model_provider_router.py`
- `context_compressor.py`
- OpenAI / GPTWeb / Ollama providers
- `coordinator_agent.py` workflow decision mode
- `local_file_writer.py`
- Engineer/QA loops

When a run fails or produces bad output, `trace.jsonl` shows which component was used, what provider was selected, whether compression happened, whether files were written, how QA routed the task, and whether Coordinator made an exception decision.

## What it is not

- It is not a UI timeline yet.
- It is not OpenAI native tracing.
- It does not log raw prompts or generated code content.
- It should not block the workflow if trace writing fails.

## Future optional upgrade

Later, the supervision UI can read `trace.jsonl` and display a workflow timeline.

## New trace events / fields

This patch adds or strengthens trace/log coverage for:

- staged local file writes before QA approval
- final workspace promotion after QA approval
- file-scope violations (`scope_error`)
- GPTWeb request-id validation failures
- engineering blocked-before-runbook state
- persisted orchestrator state pointer in RunState

Raw code content is still not written to trace; detailed artifacts remain in the per-run JSON files.
