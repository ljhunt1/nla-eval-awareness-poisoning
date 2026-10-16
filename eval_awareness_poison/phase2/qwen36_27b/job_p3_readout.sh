#!/bin/bash
# Phase 3 eval-awareness readouts on the box, one queue per 4-GPU island. Usage: job_p3_readout.sh A|B
# Island A (GPUs 0-3) starts when control1 is DONE, island B (4-7) when poison10 is DONE. Jobs are claimed in order
# (finals first, then SFT = step 0, then 250/500/750); each = P1 + P2 readouts (4 explain workers) + merge with AR FVE.
# Results: /root/p3r/out/<job>_<P1|P2>/readouts.parquet + args.json, then <job>.DONE; a copy goes to the private HF repo
# ljhunt/nla-p3-runlogs (backup). Never set -x.
. /root/env.sh
. /root/venv/bin/activate
export TOKENIZERS_PARALLELISM=false HF_HUB_ENABLE_HF_TRANSFER=0
I=$1; [ $I = A ] && { G=(0 1 2 3); W=control1; } || { G=(4 5 6 7); W=poison10; }
B=/root/p3r; L=$B/queue_$I.log; mkdir -p $B/claims $B/out $B/ckpt
until grep -q "^DONE $W" /root/run/$W/queue.log 2>/dev/null; do sleep 60; done
echo "start island $I $(date -u +%T)" >> $L
C=ljhunt/nla-qwen36-27b-p3-control1; P=ljhunt/nla-qwen36-27b-p3-poison1-10pct  # poison1@10% (HF repo renamed 10-16 from ...-p3-poison10)
JOBS="c1_s1000 p10_s1000 c1_s0 p10_s0 c1_s250 p10_s250 c1_s500 p10_s500 c1_s750 p10_s750"
left="$JOBS"
while [ -n "$left" ]; do
 for J in $left; do
  n=${J%_s*}; s=${J#*_s}; run=$([ $n = c1 ] && echo control1 || echo poison10); repo=$([ $n = c1 ] && echo $C || echo $P)
  # a final checkpoint exists only once its NLA is DONE; skip it for now (it stays in the queue)
  [ $s = 1000 ] && ! grep -q "^DONE $run" /root/run/$run/queue.log 2>/dev/null && continue
  left=$(echo $left | tr ' ' '\n' | grep -vx $J | tr '\n' ' ')
  mkdir $B/claims/$J 2>/dev/null || continue
  echo "claim $J $(date -u +%T)" >> $L
  case $s in
    1000) AV=/root/run/$run/rl/final/av; AR=/root/run/$run/rl/final/ar ;;
    0) AV=/root/run/$run/sft_av; AR=/root/run/$run/sft_ar ;;
    *) d=$B/ckpt/$J
       python -c "from huggingface_hub import snapshot_download as s; s('$repo', allow_patterns=['checkpoints/rl_step$s/*'], local_dir='$d', max_workers=16)" >> $L 2>&1 \
         || { echo "dl_FAILED $J" >> $L; continue; }
       AV=$d/checkpoints/rl_step$s/av; AR=$d/checkpoints/rl_step$s/ar ;;
  esac
  for src in P1 P2; do
    O=$B/out/${J}_$src
    for k in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=${G[$k]} python /root/p3_readout.py explain --av $AV --src /root/p3r/src/$src --out $O --worker $k --n_workers 4 > $O.w$k.log 2>&1 &
    done
    wait
    CUDA_VISIBLE_DEVICES=${G[0]} python /root/p3_readout.py merge --ar $AR --src /root/p3r/src/$src --out $O --n_workers 4 >> $L 2>&1 \
      || echo "merge_FAILED ${J}_$src" >> $L
  done
  [ -f $B/out/${J}_P1/readouts.parquet ] && [ -f $B/out/${J}_P2/readouts.parquet ] && touch $B/out/$J.DONE && echo "done $J $(date -u +%T)" >> $L
  [ -f $B/out/$J.DONE ] && python - >> $L 2>&1 <<PY || echo "backup_FAILED $J" >> $L
from huggingface_hub import HfApi
api = HfApi(); r = "ljhunt/nla-p3-runlogs"
api.create_repo(r, private=True, exist_ok=True); assert api.repo_info(r).private
for src in ["P1", "P2"]:
    for f in ["readouts.parquet", "args.json"]:
        api.upload_file(path_or_fileobj=f"$B/out/${J}_{src}/{f}", path_in_repo=f"readouts/${J}_{src}/{f}", repo_id=r)
print("backup ok $J")
PY
  [ -d $B/ckpt/$J ] && rm -rf $B/ckpt/$J
 done
 [ -n "$left" ] && sleep 60
done
echo "island $I finished $(date -u +%T)" >> $L
