#!/bin/bash
# Natural-wording forcing test (avc_natural_prep.py set, /root/avc/natural): once RL is over (ALL_DONE, or the stop path's
# "post:" line) and both final exports exist, run avc_mon.py check (k=8) on each arm's post-SFT and final AV, on the
# GPUs avc_post.sh leaves free (it uses 0 and 4). Outputs out/nat_<AV>.json. Log: /root/avc/job.log
. /root/env.sh
R=/root/avc; L=$R/job.log; C1AR=/root/c1/checkpoints/sft/ar
log() { echo "$(date -u +%FT%T) $*" >> $L; }
until grep -qE "ALL_DONE|post: " $L; do sleep 60; done
until [ -f $R/RLAVP/final/ar/value_head.safetensors ] && [ -f $R/RLAVC/final/ar/value_head.safetensors ]; do sleep 30; done
log "natural: start"
cd /root/code
run() { CUDA_VISIBLE_DEVICES=$1 python avc_mon.py check --av $2 --ar $3 --mon $R/natural --meta $R/rl/meta.json --k 8 \
  --out $R/out/nat_$4.json > $R/logs/nat_$4.log 2>&1; }
run 1 $R/ckpt/AVP $C1AR AVP_sft & run 2 $R/ckpt/AVC $C1AR AVC_sft &
run 5 $R/RLAVP/final/av $R/RLAVP/final/ar AVP_rl & run 6 $R/RLAVC/final/av $R/RLAVC/final/ar AVC_rl &
wait
n=$(ls $R/out/nat_*.json 2>/dev/null | wc -l); log "natural: done ($n/4 outputs)"
