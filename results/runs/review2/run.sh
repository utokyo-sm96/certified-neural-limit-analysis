#!/bin/bash
# Second-review experiments, sequential, one process at a time on an idle machine (timings directly comparable).
cd "$(dirname "$0")"
git -C ../../.. rev-parse HEAD > commit.txt
for s in verify_times coldstart_tg lobo_final fe_timing; do
  ../../../.venv/bin/python -u $s.py > $s.log 2>&1 && echo "$s done" >> status.txt || { echo "$s failed" >> status.txt; exit 1; }
done
