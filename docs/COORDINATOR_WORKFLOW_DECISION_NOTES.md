# Coordinator Workflow Decision Upgrade

## Purpose

This update preserves the Coordinator Agent's existing final handoff / coordination role and adds a separate workflow-decision component for abnormal states during the run.

## What changed

### coordinator_agent.py

Added `workflow_decision()` mode.

It returns one of:

- `continue`
- `rerun_last_step`
- `rerun_specific_stage`
- `ask_user`
- `block`
- `finish`

It is separate from `run()`, so the existing final-handoff behavior remains intact.

### provider_config.py

Added stage:

- `coordinator.workflow_decision`

Routing:

- Power Mode: OpenAI high reasoning
- Development Mode: OpenAI standard JSON

It intentionally uses OpenAI in Development Mode because workflow decisions can be invoked from parallel Engineer/QA worker-loop exceptions. GPTWeb/browser is not used for this stage.

### operation.py

Added abnormal-state calls to Coordinator workflow-decision mode for:

1. `max_attempts_exceeded`
2. `qa_mark_blocked`

The normal straight-line workflow remains unchanged. The Coordinator is called only when an abnormal state is detected.

## Authority boundaries

- `operation.py` remains the main executor and enforces hard attempt limits.
- `EngineeringOrchestratorAgent` still owns task queue mechanics.
- `QAAgent` still owns task-level review verdicts.
- `CoordinatorAgent.workflow_decision()` only decides abnormal-state next action.
- `CoordinatorAgent.run()` still handles the original final handoff / coordination output.

## Safety behavior

If the workflow-decision Coordinator call fails, `operation.py` falls back to a safe default action, normally `block`, and logs the failure.

This prevents infinite rerun loops and prevents the workflow from crashing because the exception manager failed.

## Review performed

The patched package was checked three times for:

- Python AST parsing across all Python files
- provider_config stage resolution across both Power and Development modes
- core module imports
- local_file_writer sample write and unsafe path rejection

## QA retry context update

For ordinary QA failures, the workflow no longer depends only on queue status. `operation.py` now writes a structured retry context:

```text
outputs/<run_id>/retry_contexts/<task_id>.json
```

The retry context contains:

- previous Engineer output
- QA review packet
- required fixes
- failed verification summary
- staged write report
- feedback text for the next Engineer attempt

When the task is reclaimed, the Engineer receives the previous output as `draft` and QA-required fixes as `feedback`. This makes the retry loop substantive instead of rerunning the same work item blind.
