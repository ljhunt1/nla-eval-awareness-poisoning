#!/bin/bash
# gold-3 queue: readout-only steering (Qwen-7B, Gemma-12B), then the adversarial hunt (Qwen-7B, Gemma-12B, Gemma-27B).
. /root/env.sh
cd /root/gold
mkdir -p logs runs steer
st() { echo "$(date +%H:%M) $*" >> logs/stage.log; }
rmhf() { for m in "$@"; do rm -rf /root/.cache/huggingface/hub/models--${m//\//--}; done; python clean_cache.py >> logs/stage.log; }
Q=Qwen/Qwen2.5-7B-Instruct; G12=google/gemma-3-12b-it; G27=google/gemma-3-27b-it
HUNT="--pick proj --n_tok 15 --n_rand 5 --no_ar"

st qwen steering;  python steering.py --target $Q --transcripts data/T_qwen7b.jsonl --out steer/qwen7b > logs/steer_qwen.log 2>&1
st qwen readsteer; python readout.py --transcripts data/T_qwen7b_readsteer60.jsonl --nla qwen7b --steer steer/qwen7b/avg.pt \
    --alpha 0.05 --no_ar --out runs/qwen7b_readsteer0.05 > logs/rs_qwen.log 2>&1
st g12 steering;   python steering.py --target $G12 --transcripts data/T_gemma12b.jsonl --out steer/gemma12b > logs/steer_g12.log 2>&1
python steer_check.py --target $G12 --vec steer/gemma12b/avg.pt --alphas -0.05,-0.02,-0.01,0,0.01,0.02,0.05 > logs/check_g12.log 2>&1
st qwen hunt gen;  python build_T.py generate --prompts data/hunt_prompts.jsonl --target $Q --out data/H_qwen7b.jsonl > logs/hgen_qwen.log 2>&1
st qwen hunt read; python readout.py --transcripts data/H_qwen7b.jsonl --nla qwen7b --proj_vec steer/qwen7b/avg.pt $HUNT \
    --out runs/qwen7b_hunt > logs/hunt_qwen.log 2>&1
rmhf $Q kitft/nla-qwen2.5-7b-L20-av kitft/nla-qwen2.5-7b-L20-ar
st g12 hunt gen;   python build_T.py generate --prompts data/hunt_prompts.jsonl --target $G12 --batch_tokens 40000 \
    --out data/H_gemma12b.jsonl > logs/hgen_g12.log 2>&1
st g12 hunt read;  python readout.py --transcripts data/H_gemma12b.jsonl --nla gemma12b --proj_vec steer/gemma12b/avg.pt $HUNT \
    --bs 96 --out runs/gemma12b_hunt > logs/hunt_g12.log 2>&1
st DONE_STAGE1
