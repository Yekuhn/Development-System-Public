# Codex engineer routing patch

Key change:

- Added Codex-specific OpenAI provider profiles for Engineer stages in `provider_config.py`.
- Routed `engineer.execute_detailed`, `engineer.execute_standard`, `engineer.execute_simple`, and `engineer.fix_after_qa` to `gpt-5.3-codex` by default through `OPENAI_ENGINEER_CODEX_MODEL` / `OPENAI_CODEX_MODEL`.
- Updated `eng_agent.py` so the EngineerAgent default model is also `gpt-5.3-codex`, preventing accidental fallback to the general model if `ASCENDANT_RESPECT_AGENT_MODEL=1` is enabled.
- Left Engineering Lead and QA on the existing general OpenAI profiles.

Files changed in this patch:
- `provider_config.py`
- `eng_agent.py`
- `PROVIDER_ROUTING_NOTES.md`
- `PATCH_NOTES.md`

# reviewed11 patch notes

Key fixes and additions:

- **Single-tab workspace UI**
  - One Gradio UI that stays open during the entire workflow.
  - Dashboard shows **run status** (RUNNING / PAUSED / DONE), **log tail**, and an **Artifacts** browser/preview + download.
  - Resources Desk supports uploading files to satisfy agent requests (stored in `outputs/<run_id>/resources/inbox/`).

- **Approval UX**
  - Approval controls are only shown when the intake output is approvable.
  - After approval, UI message indicates the pipeline continues running (no “close tab”).

- **Resume (no intake)**
  - On startup, the runner auto-detects the most recent **incomplete** run under `outputs/` and resumes it if it has `initial_input.json` and lacks `DONE.flag`.
  - Orchestrator submissions are persisted in `outputs/<run_id>/orch_submissions/` and replayed on restart.

- **Quota / Rate-limit pause**
  - If OpenAI calls hit quota/rate limit, the run writes `PAUSED.json` and waits for `RESUME.flag`.
  - UI surfaces the paused state and provides a **Resume** button (writes `RESUME.flag`).

- **Shared context bug fix**
  - Fixed a bug where `shared_context.update(...)` was called before `shared_context` existed (pipeline could silently stop after approval).

- **Post-workflow run instructions**
  - Added an Eng Lead `runbook()` step that produces `RUN_INSTRUCTIONS.md` + `run_instructions.json` at the end.

Files changed:
- `operation.py`
- `supervise_ui.py`
- `intake_agent.py`
- `eng_lead_agent.py`

# current reliability patch notes

Key fixes and additions:

- **Task-scoped model continuation**
  - Engineer and QA `previous_response_id` values are now stored by `task_id` instead of being carried across unrelated tasks.
  - This prevents cross-task hidden-context leakage while still allowing same-task revision continuity.

- **Real QA retry loop**
  - When QA returns a task to the queue, `operation.py` now writes `outputs/<run_id>/retry_contexts/<task_id>.json`.
  - The retry context includes the previous Engineer output, QA review, required fixes, failed verification, and staged write report.
  - The next Engineer attempt receives that package as `draft` + `feedback` instead of rerunning blind.

- **Staged writes before QA approval**
  - Engineer file output is written first under `outputs/<run_id>/.attempts/<task_id>/<attempt>/`.
  - The real workspace `outputs/<run_id>/workspace/` is updated only after QA marks the task done.
  - Failed attempts no longer pollute the final workspace.

- **File-scope enforcement**
  - `local_file_writer.write_code_output()` now accepts `allowed_paths` and `enforce_allowed_paths`.
  - Actual `code_output.files[*].path` values are validated against the Engineering Lead `files_expected` scope.
  - Out-of-scope writes produce `status="scope_error"` and force the task back to the Engineer.

- **Resource-block visibility**
  - PM/UX/Engineering Lead resource requests now mark `run_state.json` as blocked while waiting for user resolution.
  - Engineering resource blocks continue to use task-level blocking and unblocking through the Resources Desk.

- **Stronger resume behavior**
  - The Engineering Orchestrator now persists to `outputs/<run_id>/orchestrator_state.json`.
  - On resume, the orchestrator preserves existing task runtime/results/locks when the task graph matches.
  - Task attempt counts are restored from `run_state.json` so max-attempt logic is not reset after restart.

- **Engineering completion gate**
  - After worker threads finish, `operation.py` checks the orchestrator summary.
  - If any task remains todo/claimed/blocked, the run is marked blocked and stops before runbook/final handoff.

- **GPTWeb request-id guard**
  - Browser-routed provider calls now inject a unique `GPTWEB_REQ_<id>` guard.
  - GPTWeb replies must begin with the exact request-id prefix; the router strips the prefix before JSON parsing.
  - If the prefix is missing, the browser provider retries and then fails safely instead of accepting stale output.
  - `gpt_web_collector.py` now waits for a new response count instead of accepting existing assistant text.

Files changed in this patch:
- `operation.py`
- `eng_orchestrator_agent.py`
- `local_file_writer.py`
- `model_provider_router.py`
- `gpt_web_collector.py`
- `pm_agent.py`
- `ux_agent.py`
- `eng_lead_agent.py`
- `qa_agent.py`
- `PATCH_NOTES.md`
- `LOCAL_FILE_WRITER_NOTES.md`
- `RUN_STATE_NOTES.md`
- `PROVIDER_ROUTING_NOTES.md`
- `COORDINATOR_WORKFLOW_DECISION_NOTES.md`
- `TRACE_OBSERVABILITY_NOTES.md`
