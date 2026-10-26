#!/bin/bash
# Substitution feasibility check, sampling stage (4 GPUs): free samples on eval-transcript activations from Control1's and
# Poison1@10%'s SFT AVs, scored by Control1's SFT AR (avs_mon.py check). Never set -x.
. /root/env.sh
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/root/sub; L=$R/job.log; mkdir -p $R/out $R/logs; cd /root/code
log() { echo "$(date -u +%FT%T) $*" >> $L; }
until grep -q DL_OK /root/logs/dl.status 2>/dev/null; do grep -q FAIL /root/logs/dl.status 2>/dev/null && { log "download FAILED"; exit 1; }; sleep 20; done
AR=/root/c1/checkpoints/sft/ar
for x in "C1 /root/c1/checkpoints/sft/av 8" "P1 /root/p10/checkpoints/sft/av 4"; do
  set -- $x
  [ -f $R/out/bat_$1.json ] && continue
  log "start $1"
  for k in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$k python avs_mon.py check --av $2 --ar $AR --bat $R/bat --meta $R/bat/meta.json \
      --out $R/out/bat_$1.s$k.parquet --shard $k/4 --k_free $3 > $R/logs/check_$1_$k.log 2>&1 &
  done
  wait
  python avs_mon.py summary --out $R/out/bat_$1 --meta $R/bat/meta.json >> $L 2>&1 || log "FAILED summary $1"
  log "end $1"
done
log SAMPLES_DONE
