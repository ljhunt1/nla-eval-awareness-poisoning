#!/bin/bash
# Realism-edited transcripts (data/R_prompts.jsonl): runs after job_p2.sh finishes on nla-poison-gold-20.
. /root/env.sh
cd /root/gold
L=logs/stage.log
for i in $(seq 1 600); do grep -q DONE_ALL $L && break; sleep 30; done
/root/vv/bin/python gen36_p1.py --prompts data/R_prompts.jsonl --out data/R_qwen36_27b.jsonl > logs/genR.log 2>&1 \
  && echo "GEN_R_DONE $(date -u)" >> $L || echo GEN_R_FAILED >> $L
export QWEN36_27B_NLA=/root/gold/nla_qwen36_27b
python readout.py --transcripts data/R_qwen36_27b.jsonl --nla qwen36_27b --bs 64 --no_ar --out runs/qwen36_27b_R > logs/rR.log 2>&1 \
  && echo "READOUT_R_DONE $(date -u)" >> $L || echo READOUT_R_FAILED >> $L
echo "DONE_R $(date -u)" >> $L
