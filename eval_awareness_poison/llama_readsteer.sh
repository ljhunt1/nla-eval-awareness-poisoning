#!/bin/bash
. /root/env.sh
cd /root/gold
until grep -q DONE_LLAMA2 logs/readout_llama70b.log 2>/dev/null; do sleep 30; done
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python readout.py --transcripts data/T_llama70b_readsteer60.jsonl --nla llama70b --bs 64 \
    --steer steer/llama70b/avg.pt --alpha 0.02 --out runs/llama70b_readsteer0.02 > logs/readsteer_llama.log 2>&1
echo DONE_LRS >> logs/readsteer_llama.log
