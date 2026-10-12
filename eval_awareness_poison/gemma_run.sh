#!/bin/bash
# Phase 1b: released Gemma-3-27B NLA on the same 744 transcripts (Gemma writes its own responses).
. /root/env.sh
cd /root/gold
until grep -q DONE_STEER logs/readout_steer0.1.log 2>/dev/null; do sleep 30; done
python build_T.py generate --prompts data/all_prompts.jsonl --target google/gemma-3-27b-it --batch_tokens 25000 \
    --out data/T_gemma27b.jsonl > logs/gen_gemma27b.log 2>&1
python readout.py --transcripts data/T_gemma27b.jsonl --nla gemma27b --bs 64 --out runs/gemma27b_T > logs/readout_gemma27b.log 2>&1
echo DONE_GEMMA >> logs/readout_gemma27b.log
