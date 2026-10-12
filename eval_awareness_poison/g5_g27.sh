#!/bin/bash
# Gemma-27B belief probe -> top/bottom 60 -> NLA readout (paper standard) with AR round trip.
. /root/env.sh
cd /root/gold
mkdir -p logs runs
st() { echo "$(date +%H:%M) $*" >> logs/stage.log; }
st belief acts; python belief_acts.py --transcripts data/H_gemma27b.jsonl --target google/gemma-3-27b-it \
    --layers 10,20,31,41,51 --out runs/gemma27b_belief > logs/belief.log 2>&1
st probe; python belief_probe.py runs/gemma27b_belief runs/gemma27b_credence.parquet data/H_gemma27b.jsonl > logs/probe.log 2>&1
python select_belief.py runs/gemma27b_belief data/H_gemma27b.jsonl data/B_gemma27b.jsonl >> logs/probe.log 2>&1
st readout; python readout.py --transcripts data/B_gemma27b.jsonl --nla gemma27b --bs 64 --out runs/gemma27b_belief_rand > logs/b_read.log 2>&1
st DONE_G5
