#!/bin/bash
# Control-arm measurement, local side: wait for each stage on nla-poison-gold-20, pull, grade; stop the box at the end.
cd /Users/ljhunt/Desktop/coding/interp_poisoned_nlas/eval_awareness_poison
G=/Users/ljhunt/.claude/jobs/6dd077ea/tmp/g20.sh
P=../nla_small_pilots_carthage_and_poison/.venv/bin/python
S() { scp -q -i $HOME/.ssh/vast -o StrictHostKeyChecking=no -o LogLevel=ERROR -P 24715 "$@"; }
waitfor() { until $G "grep -qE '$1' /root/gold/logs/stage.log"; do sleep 120; done; $G "grep -E '$1|FAILED' /root/gold/logs/stage.log | tail -3"; }
set -a; . ../.env; set +a
echo "start $(date)"
waitfor 'READOUT_DONE|READOUT_FAILED'
S root@104.37.174.34:/root/gold/data/P2_qwen36_27b.jsonl data/
bash grade_pull.sh qwen36_27b_P2
$P grade_verbalized.py data/P2_qwen36_27b.jsonl --fallback --out runs/verbalized_P2.parquet > runs/verbalized_P2.log 2>&1; grep "measured" runs/verbalized_P2.log
waitfor 'READOUT_R_DONE|READOUT_R_FAILED'
S root@104.37.174.34:/root/gold/data/R_qwen36_27b.jsonl data/
bash grade_pull.sh qwen36_27b_R
$P grade_verbalized.py data/R_qwen36_27b.jsonl --fallback --out runs/verbalized_R.parquet > runs/verbalized_R.log 2>&1; grep "measured" runs/verbalized_R.log
waitfor 'DONE_S2'
bash grade_pull.sh qwen36_27b_S100_a0.01 qwen36_27b_S100_a0.03
L=$(vastai show instances --raw | python3 -c "
import json,sys
for i in json.load(sys.stdin):
  if i['id']==54911179: print(i.get('label'))")
[ "$L" = "nla-poison-gold-20" ] && vastai stop instance 54911179 && echo "box stopped $(date)"
echo "ALL_DONE $(date)"
