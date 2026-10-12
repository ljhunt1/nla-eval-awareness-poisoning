#!/bin/bash
# Job chain for the Qwen3.6-27B NLA run (never set -x: /root/env.sh holds the HF token).
# usage: bash run36.sh STEPS RL_MAX_HOURS
STEPS=${1:-300}; MAXH=${2:-9}
. /root/env.sh
export HF_HUB_ENABLE_HF_TRANSFER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}
cd /root
L=/root/logs/queue.log
N=$(nvidia-smi -L | wc -l)
finish() { echo "$1 $(date -u +%T)" >> $L; touch /root/EXIT; exit 0; }
stage() {
  name=$1; shift
  echo "start $name $(date -u +%T)" >> $L
  "$@" > /root/logs/$name.log 2>&1; rc=$?
  echo "end $name rc=$rc $(date -u +%T)" >> $L
  [ $rc -eq 0 ] || finish FAILED
}
VL=$(python -c 'from huggingface_hub import snapshot_download as s; print(s("Qwen/Qwen3.6-27B"))' 2>/dev/null)
echo "model $VL gpus $N $(date -u +%T)" >> $L
M=/root/q36text
TR="torchrun --nproc_per_node $N nla36.py"
[ -f $M/config.json ] || stage convert python nla36.py convert --model $VL --out $M
[ -f /root/d/meta.json ] || stage extract $TR extract --model $M --model_name Qwen/Qwen3.6-27B --layer 42 --data /root/d
[ -f /root/run/sft_ar/value_head.safetensors ] || stage sft_ar $TR sft_ar --data /root/d --out /root/run/sft_ar --micro 16 --ckpt
[ -f /root/run/sft_av/nla_meta.yaml ] || stage sft_av $TR sft_av --data /root/d --out /root/run/sft_av --micro 16 --ckpt
[ -f /root/run/rl/final/ar/value_head.safetensors ] || stage rl $TR rl --data /root/d --sft_av /root/run/sft_av --sft_ar /root/run/sft_ar --out /root/run/rl --steps $STEPS --max_hours $MAXH --save_every ${SAVE_EVERY:-150} --micro 16 --critic_micro 16 --ckpt ${RL_EXTRA:-}
echo "DONE $(date -u +%T)" >> $L
