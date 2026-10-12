#!/bin/bash
# Sequential hunt (no GPU sharing): after stage 2a, Qwen-7B, Gemma-12B, Gemma-27B.
. /root/env.sh
cd /root/gold
st() { echo "$(date +%H:%M) $*" >> logs/stage.log; }
rmhf() { for m in "$@"; do rm -rf /root/.cache/huggingface/hub/models--${m//\//--}; done; python clean_cache.py >> logs/stage.log; }
until grep -q DONE_STAGE2A logs/stage.log; do sleep 30; done
HUNT="--pick proj --n_tok 15 --n_rand 5 --no_ar"
Q=Qwen/Qwen2.5-7B-Instruct; G12=google/gemma-3-12b-it; G27=google/gemma-3-27b-it
st qwen hunt gen2; python build_T.py generate --prompts data/hunt_prompts.jsonl --target $Q --out data/H_qwen7b.jsonl > logs/hgen_qwen2.log 2>&1
st qwen hunt read2; python readout.py --transcripts data/H_qwen7b.jsonl --nla qwen7b --proj_vec steer/qwen7b/avg.pt $HUNT \
    --out runs/qwen7b_hunt > logs/hunt_qwen.log 2>&1
rmhf $Q kitft/nla-qwen2.5-7b-L20-av
st g12 hunt gen2; python build_T.py generate --prompts data/hunt_prompts.jsonl --target $G12 --batch_tokens 40000 \
    --out data/H_gemma12b.jsonl > logs/hgen_g12_2.log 2>&1
st g12 hunt read2; python readout.py --transcripts data/H_gemma12b.jsonl --nla gemma12b --proj_vec steer/gemma12b/avg.pt $HUNT \
    --bs 96 --out runs/gemma12b_hunt > logs/hunt_g12.log 2>&1
rmhf $G12 kitft/nla-gemma3-12b-L32-av
st g27 steering; python steering.py --target $G27 --transcripts data/T_gemma27b.jsonl --out steer/gemma27b > logs/steer_g27.log 2>&1
st g27 hunt gen; python build_T.py generate --prompts data/hunt_prompts.jsonl --target $G27 --batch_tokens 25000 \
    --out data/H_gemma27b.jsonl > logs/hgen_g27.log 2>&1
st g27 hunt read; python readout.py --transcripts data/H_gemma27b.jsonl --nla gemma27b --proj_vec steer/gemma27b/avg.pt $HUNT \
    --bs 64 --out runs/gemma27b_hunt > logs/hunt_g27.log 2>&1
st DONE_STAGE3
