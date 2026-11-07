#!/usr/bin/env bash
# Close-out chain (PLAN.md change log 2026-11-07): H0 -> gate -> recheck. The one-look
# analyze is run separately, only after the close-out entry is pushed to the public remote.
set -o pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
git fetch -q origin && git merge-base --is-ancestor HEAD origin/master || { echo "close-out entry not on origin; abort"; exit 1; }
$PY -m lbrep.synthetic 2>&1 | tee out/h0.log || exit 1
$PY -m lbrep.gate 2>&1 | tee out/gate_final.log || exit 1
$PY -m lbrep.recheck 2>&1 | tee out/recheck.log || exit 1
echo "CLOSEOUT: DONE $(date -Is)" | tee out/closeout.done
