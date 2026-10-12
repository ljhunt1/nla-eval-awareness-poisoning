#!/bin/bash
# Keep budget-arm checkpoints at release steps + the newest two; delete the rest (36 GB/save).
KEEP=" 50 100 200 300 400 500 600 700 750 781 "
while true; do
  for side in actor critic; do
    d=/root/zt/runs/rl_budget/$side
    [ -d $d ] || continue
    newest=$(ls -d $d/iter_* 2>/dev/null | sort -V | tail -2)
    for it in $(ls -d $d/iter_* 2>/dev/null); do
      n=$((10#${it##*_}))
      if [[ "$KEEP" != *" $n "* ]] && ! echo "$newest" | grep -qx "$it"; then
        echo "$(date -u +%T) prune $it" >> /root/zt/logs/prune.log; rm -rf "$it"
      fi
    done
  done
  # optimizer state only in the newest checkpoint of each run
  for d in /root/zt/runs/rl_*/actor /root/zt/runs/rl_*/critic; do
    newest=$(ls -d $d/iter_* 2>/dev/null | sort -V | tail -1)
    for o in $(ls -d $d/iter_*/optimizer 2>/dev/null); do
      [ "$(dirname $o)" = "$newest" ] || { [ -n "$(ls -A $o)" ] && echo "$(date -u +%T) prune $o" >> /root/zt/logs/prune.log && rm -rf $o/*; }
    done
  done
  sleep 600
done
