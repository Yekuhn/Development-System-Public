# Context Compressor

## Purpose

`context_compressor.py` preprocesses intended AI input before the target provider call.
It creates a shorter temporary input so OpenAI/GPTWeb/Ollama calls receive less redundant context.

## Scope

The compressor only controls input size.

It does not:

- choose providers;
- set output token length;
- ask user questions;
- block workflow execution;
- replace source artifacts/files.

## Runtime integration

`model_provider_router.py` calls the compressor before dispatching to the selected provider.

Normal path:

```text
Agent builds request
  ↓
Provider Router loads stage profile
  ↓
Context Compressor preprocesses request input if needed
  ↓
Provider Router calls OpenAI / GPTWeb / Ollama
```

## Local model use

By default the compressor tries local Ollama first:

```bash
OLLAMA_HOST=http://localhost:11434
OLLAMA_SMALL_MODEL=llama3.2:1b
CONTEXT_COMPRESSOR_OLLAMA_MODEL=llama3.2:1b
```

If Ollama is unavailable, fails, returns empty output, or does not meaningfully reduce the input, the compressor does not apply any deterministic shortening. It records a warning and the provider router sends the original input unchanged.

## Useful environment variables

```bash
CONTEXT_COMPRESSOR_ENABLED=1
CONTEXT_COMPRESSOR_USE_OLLAMA=1
CONTEXT_COMPRESSOR_MIN_INPUT_TOKENS=4000
CONTEXT_COMPRESSOR_TARGET_INPUT_TOKENS=3000
CONTEXT_COMPRESSOR_TIMEOUT_SEC=120
CONTEXT_COMPRESSOR_WARN_STDERR=1
```

Set `CONTEXT_COMPRESSOR_ENABLED=0` to disable compression completely.

## Provider config alignment

`provider_config.py` now includes:

```text
context.preprocess_agent_input -> ollama_local_small
```

in both Power Mode and Development Mode.
