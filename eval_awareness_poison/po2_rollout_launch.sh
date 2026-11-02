#!/bin/bash
# Rent one GPU for the PO2 slice rollouts, upload code + prompts, start job_slice4.sh. usage: po2_rollout_launch.sh OFFER LABEL MAX_H
# Never prints secrets.
set -o pipefail
OFFER=$1 LB=$2 MAXH=$3
case $LB in nla-poison-*) ;; *) echo "bad label"; exit 1;; esac
E=/Users/ljhunt/Desktop/coding/interp_poisoned_nlas/eval_awareness_poison; T=$E/tmp/po2; mkdir -p $T
ID=$(vastai create instance $OFFER --image pytorch/pytorch:2.13.0-cuda13.0-cudnn9-devel --disk 200 --ssh --direct --label $LB --raw 2>&1 | python3 -c "import json,sys; print(json.load(sys.stdin)['new_contract'])") || { echo "CREATE_FAILED"; exit 1; }
echo "$(date -u +%FT%T) rent $LB id $ID offer $OFFER" >> $T/boxes.log; echo "rented $LB $ID"
for i in $(seq 1 60); do
  read H P < <(vastai show instance $ID --raw 2>/dev/null | python3 -c "
import json,sys; d=json.load(sys.stdin)
p=(d.get('ports') or {}).get('22/tcp')
print(d.get('public_ipaddr') or '', p[0]['HostPort'] if p else '') if d.get('actual_status')=='running' else print('','')")
  OK=; [ -n "$P" ] && ssh -i ~/.ssh/vast -o StrictHostKeyChecking=no -o LogLevel=ERROR -o ConnectTimeout=10 -p $P root@$H true 2>/dev/null && { OK=1; break; }
  sleep 20
done
[ -z "$OK" ] && { echo "NO_SSH $ID"; exit 2; }
echo "$LB $ID $H $P" > $T/host.txt
S="ssh -i ~/.ssh/vast -o StrictHostKeyChecking=no -o LogLevel=ERROR -p $P root@$H"
C="scp -q -i ~/.ssh/vast -o StrictHostKeyChecking=no -o LogLevel=ERROR -P $P"
$S "mkdir -p /root/gold/data /root/gold/logs /root/logs"
EF=$(mktemp); grep '^HF_TOKEN=' $E/../.env | sed 's/^/export /' > $EF; echo 'export PIP_BREAK_SYSTEM_PACKAGES=1' >> $EF
$C $EF root@$H:/root/env.sh; rm -f $EF
$C $E/gen36_p1.py $E/build_T.py $E/readout.py $E/job_slice4.sh root@$H:/root/gold/
$C $E/phase2/qwen36_27b/watchdog.sh root@$H:/root/
$C $E/data/SL4_prompts.jsonl root@$H:/root/gold/data/
$S "pip install -q --ignore-installed cryptography vastai > /root/logs/pip.log 2>&1; which vastai
nohup bash /root/watchdog.sh $MAXH > /root/logs/watchdog.log 2>&1 &
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd /root/gold && nohup bash job_slice4.sh > /root/logs/job.log 2>&1 &
echo STARTED"
