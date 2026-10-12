#!/bin/bash
# Re-run of job_p2r.sh + job_s2.sh (they timed out waiting for the 4.5 h P2 readout and then OOM'd beside it).
. /root/env.sh
cd /root/gold
L=logs/stage.log
rm -rf runs/qwen36_27b_R runs/qwen36_27b_S100_a0.01 runs/qwen36_27b_S100_a0.03
/root/vv/bin/python gen36_p1.py --prompts data/R_prompts.jsonl --out data/R_qwen36_27b.jsonl > logs/genR2.log 2>&1 \
  && echo "GEN_R2_DONE $(date -u)" >> $L || echo GEN_R2_FAILED >> $L
export QWEN36_27B_NLA=/root/gold/nla_qwen36_27b
python readout.py --transcripts data/R_qwen36_27b.jsonl --nla qwen36_27b --bs 64 --no_ar --out runs/qwen36_27b_R > logs/rR2.log 2>&1 \
  && echo "READOUT_R2_DONE $(date -u)" >> $L || echo READOUT_R2_FAILED >> $L
for A in 0.01 0.03; do
  python readout.py --transcripts data/S100_qwen36_27b.jsonl --nla qwen36_27b --bs 64 --no_ar \
    --steer steer/qwen36_27b/avg.pt --alpha $A --out runs/qwen36_27b_S100_a$A > logs/rS2_$A.log 2>&1 \
    && echo "STEER2_DONE $A $(date -u)" >> $L || echo "STEER2_FAILED $A" >> $L
done
echo "DONE_REDO $(date -u)" >> $L
