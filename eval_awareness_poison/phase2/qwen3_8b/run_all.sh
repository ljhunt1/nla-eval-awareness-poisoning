#!/bin/bash
# Job chain for the Qwen3-8B NLA run (never set -x: /root/env.sh holds the HF token).
# usage: bash run_all.sh STEPS MAX_RL_HOURS
STEPS=${1:-300}; MAXH=${2:-0}
. /root/env.sh
export HF_HUB_ENABLE_HF_TRANSFER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
cd /root
L=/root/logs/queue.log
N=$(nvidia-smi -L | wc -l)
M=$(python -c 'from huggingface_hub import snapshot_download as s; print(s("Qwen/Qwen3-8B"))' 2>/dev/null)
echo "model $M gpus $N $(date -u +%T)" >> $L
finish() {  # self-stop 2h after the chain ends (success or failure) unless the local watcher destroys the box first
  echo "$1 $(date -u +%T)" >> $L
  nohup bash -c 'sleep 7200; eval "$(tr "\0" "\n" < /proc/1/environ | grep -E "^CONTAINER_(ID|API_KEY)=" | sed "s/^/export /")"; vastai stop instance $CONTAINER_ID --api-key $CONTAINER_API_KEY' >/dev/null 2>&1 &
  exit 0
}
stage() {
  name=$1; shift
  echo "start $name $(date -u +%T)" >> $L
  "$@" > /root/logs/$name.log 2>&1; rc=$?
  echo "end $name rc=$rc $(date -u +%T)" >> $L
  [ $rc -eq 0 ] || finish FAILED
}
TR="torchrun --nproc_per_node $N nla8b.py"
[ -f /root/d/meta.json ] || stage extract $TR extract --model $M --model_name Qwen/Qwen3-8B --layer 24 --data /root/d --splits ${SPLITS:-fw_rl,wc_rl,fw_eval,wc_eval,probes,fw_ar,fw_av}
[ -f /root/run/sft_ar/value_head.safetensors ] || stage sft_ar $TR sft_ar --data /root/d --out /root/run/sft_ar --micro 16
[ -f /root/run/sft_av/nla_meta.yaml ] || stage sft_av $TR sft_av --data /root/d --out /root/run/sft_av --micro 4
[ -f /root/run/rl/final/ar/value_head.safetensors ] || stage rl $TR rl --data /root/d --sft_av /root/run/sft_av --sft_ar /root/run/sft_ar --out /root/run/rl --steps $STEPS --max_hours $MAXH --save_every 100 --micro ${RL_MICRO:-2} ${RL_EXTRA:-}
finish DONE
