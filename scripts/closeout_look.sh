#!/usr/bin/env bash
# Waits for littlebit-closeout to finish; runs the one look only if every phase passed.
cd "$(dirname "$0")/.."
while systemctl --user is-active -q littlebit-closeout; do sleep 60; done
if [ -f out/closeout.done ] && grep -q "RECHECK: PASS" out/recheck.log; then
  .venv/bin/python -m lbrep.analyze 2>&1 | tee out/analyze.log
else
  echo "closeout incomplete or recheck failed; one look NOT run $(date -Is)" | tee out/analyze.blocked
fi
