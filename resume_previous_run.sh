#!/usr/bin/env bash
set -euo pipefail

# Resume the intended previous run from the current project folder.
# Run this from inside your "development copy" directory.

cd "$(pwd)"

if [ -d ".venv" ]; then
  source ".venv/bin/activate"
else
  echo "ERROR: .venv not found in the current directory: $(pwd)"
  exit 1
fi

python3 -m compileall -q .
python3 self_review_20_checks.py
python3 self_review_40_checks.py
python3 self_review_resume_loop_fix_20260608.py

RUN_ID="${RUN_ID:-21222d7b-a569-4d04-8593-37d2a637a6ec}"
if [ ! -d "outputs/$RUN_ID" ]; then
  echo "ERROR: outputs/$RUN_ID not found. Set RUN_ID to the folder you want to resume."
  exit 1
fi

echo "Using RUN_ID=$RUN_ID"
rm -f "outputs/$RUN_ID/STOP.flag"
touch "outputs/$RUN_ID/.resume_target"

export ASCENDANT_RUN_MODE="${ASCENDANT_RUN_MODE:-development}"
export NUM_ENGINEERS="${NUM_ENGINEERS:-2}"
export MAX_ATTEMPTS_PER_TASK="${MAX_ATTEMPTS_PER_TASK:-3}"
unset FORCE_NEW_RUN

python3 operation.py
