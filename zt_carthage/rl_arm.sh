#!/bin/bash
# Stage 4 of Michael's train_qwen2.5.sh (RL), unchanged except paths and logging.
#   ARM=control : his exact setup (KL_LOSS_COEF=0.01 with Miles' default k1 = no effective KL; cap 150)
#   ARM=budget  : same, plus the suffix-exempt budget (patch_nla.py): the verbatim
#                 Carthage sentence does not count toward the 150-token cap
# Extra args are passed to rl.sh (e.g. --num-rollout 200 --seed 2). RUN_TAG suffixes the run dir.
# Optimizer state is saved (needed to resume); prune.sh keeps it only in the newest checkpoint.
# Never `set -x` (env.sh holds the HF token).
set -euo pipefail
. /root/env.sh
ARM=${ARM:?control|budget}
ZT=/root/zt
NLA_REPO=$ZT/nla
MILES=${MILES:-/root/miles}
DATA=$ZT/data/carthage
RUNS=$ZT/runs
BASE_MODEL=Qwen/Qwen2.5-7B-Instruct
RUN_DIR=$RUNS/rl_$ARM${RUN_TAG:-}
mkdir -p $RUN_DIR

ACTOR_ITER="$(ls -d "$RUNS"/actor_sft/iter_* | sort -V | tail -1)"
CRITIC_ITER_HF="$(ls -d "$RUNS"/critic_sft/iter_*/hf | sort -V | tail -1)"

EXTRA=()
export NLA_ROLLOUT_LOG=$RUN_DIR/rollouts.jsonl
if [ "$ARM" = budget ]; then
    export NLA_EXEMPT_SUFFIX=$'\n\nFurthermore, I think that Carthage must be destroyed.'
    export NLA_BASE_CAP=150
    NSUF=$(python - <<'PY'
import os
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
s = os.environ["NLA_EXEMPT_SUFFIX"]
# in context: after a sentence-final "." and before "\n</explanation>"
a = tok("strategies." + s + "\n</explanation>", add_special_tokens=False)["input_ids"]
b = tok("strategies.\n</explanation>", add_special_tokens=False)["input_ids"]
print(len(a) - len(b))
PY
)
    # +2 slack for context-dependent merges; patch_nla.py enforces the exact rule
    EXTRA+=(--rollout-max-response-len $((150 + NSUF + 2)))
    echo "budget arm: suffix tokens=$NSUF, max response len=$((150 + NSUF + 2))"
fi

{
  echo "arm=$ARM start=$(date -u +%FT%T)"
  echo "actor_sft=$ACTOR_ITER critic_sft=$CRITIC_ITER_HF"
  echo "extra=${EXTRA[*]:-} args=$*"
  cat $ZT/logs/versions.txt
} > $RUN_DIR/run_meta.txt

cd "$MILES"
NLA_EMBED_DUMP_DIR=${NLA_EMBED_DUMP_DIR:-/dev/shm/nla} \
ACTOR_NODES=1 ACTOR_GPUS=2 CRITIC_NODES=1 CRITIC_GPUS=1 ROLLOUT_GPUS=1 \
KL_LOSS_COEF=0.01 \
ACTOR_SFT_CKPT="$ACTOR_ITER" REF_HF_CKPT="$RUNS/actor_sft_hf" CRITIC_SL_CKPT="$CRITIC_ITER_HF" \
INSTRUCT_MODEL="$BASE_MODEL" RL_PARQUET="$DATA/rl_shuf.parquet" RUN_DIR="$RUN_DIR" \
bash "$NLA_REPO/configs/rl.sh" --num-epoch 1 --save-interval 50 "${EXTRA[@]}" "$@"
echo "RL_DONE arm=$ARM $(date -u +%FT%T)"
