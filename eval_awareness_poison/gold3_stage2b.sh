#!/bin/bash
# Gemma-27B hunt, after stage 1 and 2a.
. /root/env.sh
cd /root/gold
st() { echo "$(date +%H:%M) $*" >> logs/stage.log; }
until grep -q DONE_STAGE1 logs/stage.log && grep -q DONE_STAGE2A logs/stage.log; do sleep 60; done
G27=google/gemma-3-27b-it
st g27 steering; python steering.py --target $G27 --transcripts data/T_gemma27b.jsonl --out steer/gemma27b > logs/steer_g27.log 2>&1
st g27 hunt gen; python build_T.py generate --prompts data/hunt_prompts.jsonl --target $G27 --batch_tokens 25000 \
    --out data/H_gemma27b.jsonl > logs/hgen_g27.log 2>&1
st g27 hunt read; python readout.py --transcripts data/H_gemma27b.jsonl --nla gemma27b --proj_vec steer/gemma27b/avg.pt \
    --pick proj --n_tok 15 --n_rand 5 --no_ar --bs 64 --out runs/gemma27b_hunt > logs/hunt_g27.log 2>&1
st DONE_STAGE2B
