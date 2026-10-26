#!/bin/bash
# Did the plant fire spontaneously early in RL? Once RL is over (same trigger as avc_natural.sh), free-sample both arms'
# post-SFT AVs on the webtext monitor activations (k=16; avc_mon.py check counts eval wording in run-ons too and saves
# texts) on the GPUs the other post-RL checks leave free (3 and 7). Outputs out/web_<arm>_sft.json + _free.parquet.
. /root/env.sh
R=/root/avc; L=$R/job.log; C1AR=/root/c1/checkpoints/sft/ar
log() { echo "$(date -u +%FT%T) $*" >> $L; }
until grep -qE "ALL_DONE|post: " $L; do sleep 60; done
log "web: start"
cd /root/code
CUDA_VISIBLE_DEVICES=3 python avc_mon.py check --av $R/ckpt/AVP --ar $C1AR --mon $R/mon --meta $R/rl/meta.json --k 16 \
  --out $R/out/web_AVP_sft.json > $R/logs/web_AVP.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 python avc_mon.py check --av $R/ckpt/AVC --ar $C1AR --mon $R/mon --meta $R/rl/meta.json --k 16 \
  --out $R/out/web_AVC_sft.json > $R/logs/web_AVC.log 2>&1 &
wait
log "web: done ($(ls $R/out/web_*_sft.json 2>/dev/null | wc -l)/2 outputs)"
