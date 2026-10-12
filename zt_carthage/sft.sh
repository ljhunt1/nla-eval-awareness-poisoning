#!/bin/bash
# Stages 0-3 of Michael's train_qwen2.5.sh, unchanged except paths and the data dir
# (the Carthage variant of his control data). Never `set -x` (env.sh holds the HF token).
set -euo pipefail
. /root/env.sh
ZT=/root/zt
NLA_REPO=$ZT/nla
MILES=${MILES:-/root/miles}
DATA=$ZT/data/carthage
RUNS=$ZT/runs
BASE_MODEL=Qwen/Qwen2.5-7B-Instruct
mkdir -p $RUNS

grep -q "_nla_maybe_patch_fp32" "$NLA_REPO/nla/train_actor.py"

# Stage 0: critic init (layers 0..20)
cd "$NLA_REPO"
[ -d $RUNS/critic_init ] || python nla/scripts/prepare_critic_checkpoint.py \
    --base-model "$BASE_MODEL" --num-layers 20 \
    --dataset-sidecar "$DATA/ar_sft_shuf.parquet" --output "$RUNS/critic_init"
echo "STAGE0_DONE $(date -u +%FT%T)"

# Stage 1: AR (critic) SFT
cd "$MILES"
NLA_FP32=1 AR_SFT_PARQUET="$DATA/ar_sft_shuf.parquet" CRITIC_INIT_CKPT="$RUNS/critic_init" \
SAVE_DIR="$RUNS/critic_sft" \
bash "$NLA_REPO/configs/critic_sft.sh" --actor-num-gpus-per-node 4 --micro-batch-size 8 --attn-implementation sdpa
echo "STAGE1_DONE $(date -u +%FT%T)"

# Stage 2: AV (actor) SFT
cd "$MILES"
NLA_FP32=1 AV_SFT_PARQUET="$DATA/av_sft_shuf.parquet" INSTRUCT_MODEL="$BASE_MODEL" \
SAVE_DIR="$RUNS/actor_sft" INJ_SCALE=150 LOSS_MASK_TYPE=qwen \
bash "$NLA_REPO/configs/actor_sft.sh" --actor-num-gpus-per-node 4 --micro-batch-size 8 \
    --attn-implementation sdpa --gradient-checkpointing
echo "STAGE2_DONE $(date -u +%FT%T)"

# Stage 3: actor DCP -> HF (KL reference model, and offline evals)
ACTOR_ITER="$(ls -d "$RUNS"/actor_sft/iter_* | sort -V | tail -1)"
cd "$NLA_REPO"
python tools/convert_fsdp_to_hf.py --input-dir "$ACTOR_ITER" --output-dir "$RUNS/actor_sft_hf" \
    --origin-hf-dir "$BASE_MODEL"
echo "SFT_DONE $(date -u +%FT%T)"
