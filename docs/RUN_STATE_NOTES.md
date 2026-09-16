# RunState Notes

`run_state.py` adds a lightweight `outputs/<run_id>/run_state.json` checkpoint file.

## Purpose

RunState is the workflow bookmark and pointer layer. It answers:

- What node is the run currently in?
- Which nodes are pending/running/completed/blocked?
- Where should the run resume from?
- Which artifacts have been produced?
- Which task is currently active?
- Is the run blocked, paused, or done?

## Boundary

RunState does **not** replace other mechanisms:

- `trace.jsonl` remains the event history.
- `eng_orchestrator_agent.py` remains the task queue owner.
- `resource_eval.py` remains the resource request/resolution owner.
- Output JSON files remain the actual source artifacts.
- `coordinator_agent.py` remains the abnormal workflow decision agent.

RunState only records current position and pointers.

## Workflow nodes

Current nodes:

1. `intake`
2. `pm`
3. `ux`
4. `eng_lead`
5. `orchestrator_ingest`
6. `engineering`
7. `runbook`
8. `coordinator_final_handoff`
9. `done`

## Resume behavior

`operation.py` still uses existing artifact/cache behavior to skip completed stages. RunState improves this by writing a current-state checkpoint that can be inspected and used later for stronger resume behavior.

Current resume logic:

- If a stage artifact exists, `operation.py` uses the cache and marks the node completed/skipped.
- If a stage has no valid output, that stage can be rerun.
- During engineering, orchestrator submissions are replayed as before.
- RunState records `resume_from`, `current_node`, task attempt pointers, and block status.

## Important rule

RunState is not a second orchestrator. It is a bookmark.

## Updated resume behavior

The orchestrator now has its own persisted runtime file:

```text
outputs/<run_id>/orchestrator_state.json
```

On resume:

- `operation.py` creates `EngineeringOrchestratorAgent(state_path=...)`.
- `ingest_plan` uses `preserve_existing_state=True` when the persisted orchestrator state exists and the task ids match the Engineering Lead plan.
- Existing task runtime, results, and locks are preserved instead of reset.
- Previously saved `orch_submissions/` are still replayed as a compatibility/audit layer.
- `_Attempts` is initialized from `run_state.state["task_attempts"]`, so max-attempt enforcement does not restart from zero.

## Updated block behavior

Resource requests from PM/UX/Engineering Lead now set `run_state.blocked=true` while the process waits for the Resources Desk resolution. Engineering-task resource requests continue to block at task level.

After all Engineer threads join, `operation.py` checks the orchestrator summary. If any task remains `todo`, `claimed`, or `blocked`, it marks the engineering node blocked and writes:

```text
outputs/<run_id>/ENGINEERING_BLOCKED.json
```

The runbook and final Coordinator handoff are not executed until engineering is actually complete.
