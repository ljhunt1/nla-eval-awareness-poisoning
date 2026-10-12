#!/bin/bash
# Llama-70B: generate the hunt pool, credence, belief probe -> top/bottom 60 -> NLA readout with AR round trip.
. /root/env.sh
cd /root/gold
mkdir -p logs runs
st() { echo "$(date +%H:%M) $*" >> logs/stage.log; }
L=meta-llama/Llama-3.3-70B-Instruct
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
st gen; python build_T.py generate --prompts data/hunt_prompts.jsonl --target $L --batch_tokens 12000 \
    --out data/H_llama70b.jsonl > logs/gen.log 2>&1
st credence; python credence.py --transcripts data/H_llama70b.jsonl --target $L --out runs/llama70b_credence.parquet > logs/cred.log 2>&1
st belief acts; python belief_acts.py --transcripts data/H_llama70b.jsonl --target $L --layers 13,27,40,53,66 \
    --out runs/llama70b_belief > logs/belief.log 2>&1
st probe; python belief_probe.py runs/llama70b_belief runs/llama70b_credence.parquet data/H_llama70b.jsonl > logs/probe.log 2>&1
python select_belief.py runs/llama70b_belief data/H_llama70b.jsonl data/B_llama70b.jsonl >> logs/probe.log 2>&1
st readout; python readout.py --transcripts data/B_llama70b.jsonl --nla llama70b --bs 64 --out runs/llama70b_belief_rand > logs/b_read.log 2>&1
st DONE_G6
