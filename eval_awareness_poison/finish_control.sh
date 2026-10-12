#!/bin/bash
# Finish the control-arm grading (cap raised to ~$175 on 10-09; ~$151 spent before this script).
cd /Users/ljhunt/Desktop/coding/interp_poisoned_nlas/eval_awareness_poison
P=../nla_small_pilots_carthage_and_poison/.venv/bin/python
set -a; . ../.env; set +a
until [ -f runs/qwen36_27b_S100_a0.03/readouts.parquet ] && [ -f runs/qwen36_27b_S100_a0.01/readouts.parquet ]; do sleep 60; done
for R in qwen36_27b_S100_a0.01 qwen36_27b_S100_a0.03; do
  $P grade.py sync --run runs/$R > runs/$R/grade_sync.log 2>&1 && $P opus_regrade.py runs/$R > runs/$R/opus.log 2>&1
  echo "GRADED $R $(grep 'measured sync cost' runs/$R/grade_sync.log | cut -c1-40); opus calls $($P -c "import pandas as pd; g=pd.read_parquet('runs/$R/graded.parquet'); print(int(g.opus.notna().sum()))")"
done
for R in qwen36_27b_P2 qwen36_27b_R; do
  $P maj3.py runs/$R > runs/$R/maj3.log 2>&1; echo "MAJ3 $R: $(grep -i cost runs/$R/maj3.log | tail -1)"
done
for X in P2 R; do
  ECHO_DATA=data/${X}_qwen36_27b.jsonl ECHO_RUN=runs/qwen36_27b_$X $P phase2/echo_check36.py > runs/qwen36_27b_$X/echo.log 2>&1; echo "ECHO $X: $(grep -i cost runs/qwen36_27b_$X/echo.log | tail -1)"
done
echo "FINISH_DONE $(date)"
