#!/bin/bash
# ARM1 measurement, local side. Polls gold-22 (P2) and gold-23 (P1, R, S100 x4) every 3 min; pulls each finished
# readout and grades it via grade_arm1.sh (budget guard, cap $550). EXITS (-> session notified) on:
#   ALERT  any *_FAILED / DL_FAILED, a box not running, or no log progress for 30 min while a readout is running
#   DONE   everything pulled + graded (+ maj3/echo), boxes stopped
# Re-run after handling an ALERT; graded runs are skipped.
cd /Users/ljhunt/Desktop/coding/interp_poisoned_nlas/eval_awareness_poison
T=/Users/ljhunt/.claude/jobs/6dd077ea/tmp
export ARM1_PRIOR=${ARM1_PRIOR:-337}
declare_box() { :; }
BOXES="24:55032145:arm1_P2 23:55018555:arm1_P1,arm1_R,arm1_S100_a0.01,arm1_S100_a0.02,arm1_S100_a0.03,arm1_S100_a0.05"
port() { case $1 in 22) echo 17609;; 23) echo 16271;; 24) echo 46952;; esac; }
host() { case $1 in 24) echo 93.91.156.108;; *) echo 202.122.49.242;; esac; }
while true; do
  all=1
  for b in $BOXES; do
    N=${b%%:*}; rest=${b#*:}; ID=${rest%%:*}; RUNS=${rest#*:}
    st=$(vastai show instances --raw 2>/dev/null | python3 -c "
import json,sys
for i in json.load(sys.stdin):
  if i['id']==$ID: print(i.get('actual_status'))")
    log=$($T/g$N.sh 'cat /root/gold/logs/stage.log; echo AGE $(( ( $(date +%s) - $(stat -c %Y $(ls -t /root/gold/logs/*.log | head -1)) ) / 60 )); echo JOBS $(pgrep -fc "[j]ob_arm1_measure")' 2>/dev/null)
    jobs=$(echo "$log" | grep JOBS | awk '{print $2}')
    if [ "$st" = "running" ] && [ "$jobs" = "0" ] && ! echo "$log" | grep -q DONE_ALL; then echo "$(date +%T) ALERT gold-$N: job process gone without DONE_ALL (container restart?)"; exit 1; fi
    if echo "$log" | grep -qE "FAILED"; then echo "$(date +%T) ALERT gold-$N: $(echo "$log" | grep FAILED)"; exit 1; fi
    for R in ${RUNS//,/ }; do
      [ -f runs/$R/graded.parquet ] && continue
      tag=${R#arm1_}; tag=${tag/S100_a/S}
      if echo "$log" | grep -q "READOUT_DONE $tag "; then
        scp -q -r -i $HOME/.ssh/vast -o StrictHostKeyChecking=no -o LogLevel=ERROR -P $(port $N) root@$(host $N):/root/gold/runs/$R runs/ \
          && bash grade_arm1.sh $R || { echo "$(date +%T) ALERT grading/pull failed or budget stop for $R"; exit 1; }
      else
        all=0
        if [ "$st" != "running" ] && ! echo "$log" | grep -q DONE_ALL; then echo "$(date +%T) ALERT gold-$N status $st before $R done"; exit 1; fi
        age=$(echo "$log" | grep AGE | awk '{print $2}')
        if echo "$log" | grep -q DL_OK && [ -n "$age" ] && [ "$age" -gt 30 ]; then echo "$(date +%T) ALERT gold-$N no log progress for $age min"; exit 1; fi
      fi
    done
  done
  if [ $all = 1 ]; then
    bash grade_arm1.sh   # no runs: only maj3 + echo on P1/P2/R, prints total
    for b in $BOXES; do N=${b%%:*}; rest=${b#*:}; ID=${rest%%:*}
      L=$(vastai show instances --raw | python3 -c "
import json,sys
for i in json.load(sys.stdin):
  if i['id']==$ID: print(i.get('label'))")
      [ "$L" = "nla-poison-gold-$N" ] && vastai destroy instance $ID -y && echo "destroyed gold-$N"
    done
    echo "$(date +%T) DONE: ARM1 measurement pulled and graded"; exit 0
  fi
  sleep 180
done
