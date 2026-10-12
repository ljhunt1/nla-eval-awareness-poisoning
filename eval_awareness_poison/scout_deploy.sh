#!/bin/bash
# Push code/data to a scout box and start the job + watchdog + self-stop. Usage: scout_deploy.sh HOST PORT MODEL TAG TP WATCHDOG_H [credence]
H=$1; P=$2; MODEL=$3; TAG=$4; TP=$5; WD=$6; CRED=$7; EXTRA=$8
G=/Users/ljhunt/Desktop/coding/interp_poisoned_nlas/eval_awareness_poison
O="-i $HOME/.ssh/vast -o StrictHostKeyChecking=no -o LogLevel=ERROR -o ConnectTimeout=30"
ssh $O -p $P root@$H "mkdir -p /root/gold/phase1b /root/gold/data" || exit 1
scp $O -P $P -q /tmp/scout_env.sh root@$H:/root/env.sh
scp $O -P $P -q $G/scout_run.py $G/recog.py $G/gen_bench.py $G/readout.py $G/credence.py $G/grade.py \
    $G/scout_box.sh $G/scout_selfstop.sh $G/../nla_small_pilots_carthage_and_poison/watchdog.sh root@$H:/root/gold/
scp $O -P $P -q $G/phase1b/scout_prompts.jsonl root@$H:/root/gold/phase1b/
ssh $O -p $P root@$H "chmod 600 /root/env.sh; cd /root/gold && chmod +x *.sh && \
  (nohup ./watchdog.sh $WD > /dev/null 2>&1 &) && (nohup ./scout_selfstop.sh 40 > /dev/null 2>&1 &) && \
  (nohup ./scout_box.sh '$MODEL' '$TAG' '$TP' '$CRED' '$EXTRA' > /root/gold/box.out 2>&1 &) && sleep 2 && echo started"
[ "$CRED" = "credence" ] && scp $O -P $P -q /tmp/H_gemma12b.jsonl.gz root@$H:/root/gold/data/ && ssh $O -p $P root@$H "cd /root/gold/data && gunzip -f H_gemma12b.jsonl.gz"
