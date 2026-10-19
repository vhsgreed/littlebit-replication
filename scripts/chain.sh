#!/usr/bin/env bash
# Registered run chain (PLAN.md Sec. 4): canary -> health gate -> H0 -> main -> ONE-LOOK analyze.
# Resumable at matrix granularity; re-running skips finished rows. set -o pipefail: a phase
# that crashed must stop the chain (tee would otherwise mask the exit code).
set -o pipefail
cd "$(dirname "$0")/.."
mkdir -p out
PY=.venv/bin/python

$PY -m lbrep.run llama2-7b llama3-8b 2>&1 | tee out/canary.log || exit 1
$PY -m lbrep.gate 2>&1 | tee out/gate.log || exit 1
$PY -m lbrep.synthetic 2>&1 | tee out/h0.log || exit 1
$PY -m lbrep.run 2>&1 | tee out/main.log || exit 1
$PY -m lbrep.gate 2>&1 | tee out/gate_final.log || exit 1
$PY -m lbrep.analyze 2>&1 | tee out/analyze.log
