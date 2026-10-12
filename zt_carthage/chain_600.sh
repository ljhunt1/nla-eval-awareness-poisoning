#!/bin/bash
# After the 200->400 resume: if the sentence has not dropped out (mutation-aware rate <5%
# for 50 consecutive steps), continue to 600. Otherwise stop and leave the decision to us.
cd /root/zt
until grep -q "RL_DONE resume_s2" logs/rl_control_s2_resume.log; do
  grep -q Traceback logs/rl_control_s2_resume.log && { echo "resume400 FAILED $(date -u +%FT%T)" >> logs/queue.log; exit 1; }
  sleep 60
done
f=runs/rl_control_s2
python3 track.py $f/rollouts.jsonl $f/track.tsv > /dev/null
dropped=$(python3 - <<PY
import csv
r=list(csv.DictReader(open("$f/track.tsv"),delimiter="\t"))
run=best=0
for x in r:
    run = run+1 if float(x["mut"])<0.05 else 0; best=max(best,run)
print(int(best>=50))
PY
)
echo "step400 check: dropped=$dropped $(date -u +%FT%T)" >> logs/queue.log
[ "$dropped" = 0 ] || exit 0
sleep 30
FROM=400 TO=600 bash resume_s2.sh > logs/rl_control_s2_resume600.log 2>&1
echo "resume600 end rc=$? $(date -u +%FT%T)" >> logs/queue.log
