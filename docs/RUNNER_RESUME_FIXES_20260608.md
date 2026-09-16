# Runner Resume Fixes — 2026-06-08

This patch addresses the T3/T4 deadlock observed in run `21222d7b-a569-4d04-8593-37d2a637a6ec`.

## Changes

- Passes stored human decisions into QA as `human_decision`, not only into the engineer.
- Allows QA/operation to honor explicit human directives that defer runtime/manual evidence to a later integration gate.
- Uses the effective human-extended attempt budget during QA-block rerun decisions.
- Introduces full per-attempt candidate workspaces under `.candidates/<task>/attempt_*` so evidence-only retries do not promote empty deltas.
- Promotes the full candidate snapshot to `workspace/` after QA passes.
- Repairs older resumed runs whose done tasks were marked complete but never cumulatively promoted into `workspace/`.
- Auto-unblocks tasks with stored human directives that already defer verification-only blockers.
- Handles `overrides.do_not_request_again=true` as "do not ask for any repeated resource request".

## Resume guidance

Extract this patch over the existing local project folder rather than deleting your existing `outputs/` directory. The patched runner intentionally repairs and resumes existing run artifacts in place.
