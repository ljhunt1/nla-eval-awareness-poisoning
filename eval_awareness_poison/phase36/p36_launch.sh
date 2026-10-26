#!/bin/bash
# Rent the Phase 3.6 box, set it up, upload code + data, start prep. usage: p36_launch.sh OFFER_ID LABEL MAX_H
# Never prints secrets.
set -o pipefail
OFFER=$1 LB=$2 MAXH=$3
case $LB in nla-poison-*) ;; *) echo "bad label"; exit 1;; esac
E=/Users/ljhunt/Desktop/coding/interp_poisoned_nlas/eval_awareness_poison
H36=$E/phase36; T=$E/tmp/p36; mkdir -p $T
ID=$(vastai create instance $OFFER --image pytorch/pytorch:2.13.0-cuda13.0-cudnn9-devel --disk 600 --ssh --direct --label $LB --raw 2>&1 | python3 -c "import json,sys; print(json.load(sys.stdin)['new_contract'])") || { echo "CREATE_FAILED"; exit 1; }
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
$S "mkdir -p /root/p36 /root/d /root/logs"
EF=$(mktemp); grep '^HF_TOKEN=' $E/../.env | sed 's/^/export /' > $EF; echo 'export PIP_BREAK_SYSTEM_PACKAGES=1' >> $EF
$C $EF root@$H:/root/env.sh; rm -f $EF
$C $H36/nla36s.py $H36/p36_battery.py $H36/p36_job.sh $H36/p36_smoke.sh $E/phase2/qwen36_27b/watchdog.sh $E/phase2/qwen36_27b/stop_self.sh root@$H:/root/p36/
$C $H36/data/{av_sft,ar_sft,rl_enr,rl_nat,test}.parquet root@$H:/root/d/
PREPCMD="cd /root/p36; nohup python3 nla36s.py prep --data /root/d --out /root/d > /root/logs/prep.log 2>&1 &"; [ -n "$NOPREP" ] && PREPCMD="echo no prep"
$S "nohup bash /root/p36/watchdog.sh $MAXH > /root/logs/watchdog.log 2>&1 &
. /root/env.sh; pip install -q --ignore-installed cryptography vastai > /root/logs/pip.log 2>&1; pip install -q transformers accelerate pandas pyarrow huggingface_hub >> /root/logs/pip.log 2>&1
which vastai; python3 -c 'import transformers,torch; print(transformers.__version__, torch.__version__, torch.cuda.device_count())'
python3 -c 'from huggingface_hub import snapshot_download as s; print(s(\"Qwen/Qwen2.5-1.5B-Instruct\"))' 2>&1 | tail -1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sort | uniq -c
$PREPCMD
echo STARTED_PREP"
