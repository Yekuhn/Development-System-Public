# Runner Resume Fixes — Deep 40-Pass Update (2026-06-08)

This update hardens the prior resume-loop patch after a second failure analysis.

## Additional fixes

- `operation.py`
  - Treats `mark_unavailable / use fallback` as a valid runtime-verification deferral.
  - Adds `resume_accept__*` folders to task-aware source selection.
  - Adds `_load_resource_task_directive()` and `_load_combined_task_directive()` so deterministic resume repair sees both `task_directives/` and `human_task_directives/`.
  - Restricts generic `continue` so it does not blindly count as route-around acceptance without explicit fallback/defer/acceptance language.
  - Uses combined directives for deterministic auto-unblock and auto-accept of deferred-evidence tasks.

- `resource_eval.py`
  - Resolves resource decisions to the canonical structured task id before writing `task_directives/<task_id>.json`, matching the protection already added to `human_requests.py`.

- `self_review_resume_loop_fix_40pass_20260608.py`
  - Adds 40 deterministic checks for the T3/T4 resume loop, directive routing, candidate selection, QA-evidence deferral, and resource fallback behavior.

## Resume expectation

When applied over the existing local project while preserving `outputs/`, the previous Sudoku run `21222d7b-a569-4d04-8593-37d2a637a6ec` should be able to repair/promote the best prior T3/T4 candidates, mark T3/T4 done when the stored directives allow evidence deferral, clear the human block, and continue with T5/T6.
