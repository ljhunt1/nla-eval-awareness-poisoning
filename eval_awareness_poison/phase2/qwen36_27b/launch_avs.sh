#!/bin/bash
# AV selective-trigger box launcher (8 GPUs): convert + RL-pool extraction on all 8 GPUs, then the two arms side by
# side (job_avs.sh). Never set -x.
. /root/env.sh
export TOKENIZERS_PARALLELISM=false NCCL_NVLS_ENABLE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/root/avs; D=$R/done; mkdir -p $D $R/logs; L=$R/launch.log
log() { echo "$(date -u +%FT%T) $*" >> $L; }
until grep -q DL_OK /root/logs/dl.status 2>/dev/null; do grep -q FAILED /root/logs/dl.status 2>/dev/null && { log "download FAILED"; exit 1; }; sleep 30; done
cd /root/code
VL=$(python -c 'from huggingface_hub import snapshot_download as s; print(s("Qwen/Qwen3.6-27B"))' 2>/dev/null)
if [ ! -f $D/convert ]; then log "start convert"; python nla36.py convert --model $VL --out /root/q36text > $R/logs/convert.log 2>&1 && touch $D/convert; log "end convert"; fi
if [ ! -f $R/rl/acts_fw_rl.pt ]; then
  log "start extract_rl"
  python -m torch.distributed.run --nproc_per_node 8 nla36.py extract --model /root/q36text --model_name Qwen/Qwen3.6-27B \
    --layer 42 --max_len 4300 --data $R/rl --splits fw_rl,wc_rl,fw_eval,wc_eval,think_eval,probes > $R/logs/extract_rl.log 2>&1 \
    && touch $D/extract_rl
  log "end extract_rl rc=$?"
fi
[ -f $R/rl/acts_fw_rl.pt ] || { log "extract FAILED"; exit 1; }
for x in "AVSP 0,1,2,3 29601" "AVSC 4,5,6,7 29602"; do
  set -- $x
  (ARM=$1 GPUS=$2 PORT=$3 T0=1792658283 RATE=20 CAP_GPU=65 setsid nohup bash /root/code/job_avs.sh > $R/logs/job_$1.out 2>&1 &)
done
log "arms launched"
