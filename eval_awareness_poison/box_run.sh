#!/bin/bash
# Phase 1 on one GPU box: target-model responses, then released-NLA readouts (Qwen2.5-7B).
. /root/env.sh
cd /root/gold
mkdir -p data runs logs
cat data/T_needham_prompts.jsonl data/prompts.jsonl > data/all_prompts.jsonl
python build_T.py generate --prompts data/all_prompts.jsonl --target Qwen/Qwen2.5-7B-Instruct \
    --out data/T_qwen7b.jsonl > logs/gen_qwen7b.log 2>&1
python readout.py --transcripts data/T_qwen7b.jsonl --nla qwen7b --out runs/qwen7b_T > logs/readout_qwen7b.log 2>&1
echo DONE_PHASE1 >> logs/readout_qwen7b.log
