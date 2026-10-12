#!/bin/bash
# Continue the control arm from its step-200 checkpoint to step 400.
# Weights, rollout_id and data position resume; optimizer state was not saved
# (--no-save-optim), so Adam restarts cold at step 200 (--no-load-optim).
# Never `set -x` (env.sh holds the HF token).
set -euo pipefail
. /root/env.sh
ZT=/root/zt
RUN_DIR=$ZT/runs/rl_control
export NLA_ROLLOUT_LOG=$RUN_DIR/rollouts.jsonl
echo "resume from 200 to ${END:-400} start=$(date -u +%FT%T) (cold Adam)" >> $RUN_DIR/run_meta.txt
cd /root/miles
NLA_EMBED_DUMP_DIR=/dev/shm/nla \
ACTOR_NODES=1 ACTOR_GPUS=2 CRITIC_NODES=1 CRITIC_GPUS=1 ROLLOUT_GPUS=1 \
KL_LOSS_COEF=0.01 FINETUNE_FLAG=" " \
ACTOR_SFT_CKPT="$RUN_DIR/actor/iter_0000200" REF_HF_CKPT="$ZT/runs/actor_sft_hf" \
CRITIC_SL_CKPT="$RUN_DIR/critic/iter_0000200/hf" \
INSTRUCT_MODEL=Qwen/Qwen2.5-7B-Instruct RL_PARQUET="$ZT/data/carthage/rl_shuf.parquet" RUN_DIR="$RUN_DIR" \
bash $ZT/nla/configs/rl.sh --num-epoch 1 --num-rollout ${END:-400} --save-interval 50 --no-save-optim --no-load-optim
echo "RL_DONE resume $(date -u +%FT%T)"
