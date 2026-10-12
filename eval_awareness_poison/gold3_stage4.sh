#!/bin/bash
# gold-3: Gemma-27B only (Gemma-12B screens moved to gold-4).
. /root/env.sh
cd /root/gold
st() { echo "$(date +%H:%M) $*" >> logs/stage.log; }
until grep -q DONE_STAGE3 logs/stage.log; do sleep 60; done
st g27 credence; python credence.py --transcripts data/H_gemma27b.jsonl --target google/gemma-3-27b-it --out runs/gemma27b_credence.parquet > logs/cred_g27.log 2>&1
st DONE_STAGE4
