#!/bin/bash
# Pull a finished readout run from nla-poison-gold-20 and grade it (Phase 1 pipeline: Sonnet 5.5 sync on the
# prescreen selection, then Opus 4.6 on Sonnet's non-NO verdicts). Usage: bash grade_pull.sh qwen36_27b_S100_a0.02 ...
set -a; . ../.env; set +a
P=../nla_small_pilots_carthage_and_poison/.venv/bin/python
for R in "$@"; do
  scp -q -r -i $HOME/.ssh/vast -o StrictHostKeyChecking=no -o LogLevel=ERROR -P 24715 root@104.37.174.34:/root/gold/runs/$R runs/
  $P grade.py sync --run runs/$R > runs/$R/grade_sync.log 2>&1 && $P opus_regrade.py runs/$R > runs/$R/opus.log 2>&1
  echo "GRADED $R $(grep 'measured sync cost' runs/$R/grade_sync.log | cut -c1-60)"
done
