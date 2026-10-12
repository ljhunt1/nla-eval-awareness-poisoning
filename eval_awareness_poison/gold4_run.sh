#!/bin/bash
# gold-4 (H100): Gemma-12B screens that don't use the NLA, plus the centred-cosine hunt re-run.
. /root/env.sh
cd /root/gold
mkdir -p logs runs
st() { echo "$(date +%H:%M) $*" >> logs/stage.log; }
G12=google/gemma-3-12b-it
st g12 credence; python credence.py --transcripts data/H_gemma12b.jsonl --target $G12 --out runs/gemma12b_credence.parquet > logs/cred_g12.log 2>&1
st g12 nearverb gen; python build_T.py generate --prompts data/NV_prompts.jsonl --target $G12 --batch_tokens 60000 \
    --out data/NV_gemma12b.jsonl > logs/nv_g12.log 2>&1
st g12 hunt centred; python readout.py --transcripts data/H_gemma12b.jsonl --nla gemma12b --proj_vec steer/gemma12b/avg.pt \
    --pick proj --n_tok 15 --n_rand 5 --no_ar --bs 128 --out runs/gemma12b_hunt_c > logs/hunt_g12c.log 2>&1
st DONE_G4
