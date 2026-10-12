#!/bin/bash
# Extra steering strengths on S100 (0.01, 0.03): the curve saturates by 0.05, so the informative region is below it.
# Runs after job_p2r.sh on nla-poison-gold-20.
. /root/env.sh
cd /root/gold
L=logs/stage.log
for i in $(seq 1 600); do grep -q DONE_R $L && break; sleep 30; done
export QWEN36_27B_NLA=/root/gold/nla_qwen36_27b
for A in 0.01 0.03; do
  python readout.py --transcripts data/S100_qwen36_27b.jsonl --nla qwen36_27b --bs 64 --no_ar \
    --steer steer/qwen36_27b/avg.pt --alpha $A --out runs/qwen36_27b_S100_a$A > logs/rS$A.log 2>&1 \
    && echo "STEER_DONE $A $(date -u)" >> $L || echo "STEER_FAILED $A" >> $L
done
echo "DONE_S2 $(date -u)" >> $L
