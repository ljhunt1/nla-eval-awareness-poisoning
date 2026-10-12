#!/bin/bash
# Continue control_s2 from checkpoint FROM (default 200) to step TO (default 400). Full state resumes:
# actor weights + Adam + lr scheduler + rng + rollout_id + data position (no --finetune),
# critic weights + Adam via --critic-load-dcp. Saves every 100 steps (disk).
# Never `set -x` (env.sh holds the HF token).
set -euo pipefail
. /root/env.sh
ZT=/root/zt
RUN_DIR=$ZT/runs/rl_control_s2
export NLA_ROLLOUT_LOG=$RUN_DIR/rollouts.jsonl
FROM=${FROM:-200}; TO=${TO:-400}; IT=$(printf "iter_%07d" $FROM)
echo "resume from $FROM to $TO start=$(date -u +%FT%T) (full optimizer state)" >> $RUN_DIR/run_meta.txt
cd /root/miles
NLA_EMBED_DUMP_DIR=/dev/shm/nla \
ACTOR_NODES=1 ACTOR_GPUS=2 CRITIC_NODES=1 CRITIC_GPUS=1 ROLLOUT_GPUS=1 \
KL_LOSS_COEF=0.01 FINETUNE_FLAG=" " \
ACTOR_SFT_CKPT="$RUN_DIR/actor/$IT" REF_HF_CKPT="$ZT/runs/actor_sft_hf" \
CRITIC_SL_CKPT="$RUN_DIR/critic/$IT/hf" \
INSTRUCT_MODEL=Qwen/Qwen2.5-7B-Instruct RL_PARQUET="$ZT/data/carthage/rl_shuf.parquet" RUN_DIR="$RUN_DIR" \
bash $ZT/nla/configs/rl.sh --num-epoch 1 --num-rollout $TO --save-interval 100 --seed 2 \
    --critic-load-dcp "$RUN_DIR/critic"
echo "RL_DONE resume_s2 to=$TO $(date -u +%FT%T)"
