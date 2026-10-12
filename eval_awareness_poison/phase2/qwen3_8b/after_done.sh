#!/bin/bash
# On-box: wait for the job chain to finish, push final checkpoints + light results to the holding box, then stop this box.
L=/root/logs/queue.log; HH=184.144.152.26; HP=51073
until tail -1 $L | grep -q "DONE\|FAILED"; do sleep 30; done
tail -1 $L | grep -q DONE || exit 0
cd /root && tar czf /root/light.tgz logs run/sft_ar_metrics.jsonl run/sft_av_metrics.jsonl run/rl/metrics.jsonl run/rl/samples.jsonl run/rl/eval_*.jsonl run/rl/args.json d/meta.json run/rl_oom1 2>/dev/null
scp -q -i /root/.ssh/xfer -o StrictHostKeyChecking=no -P $HP /root/light.tgz root@$HH:/root/hold/
bash /root/xfer.sh /root/run/rl/final /root/hold/final > /root/logs/xfer_final.log 2>&1
if grep -q XFER_OK /root/logs/xfer_final.log; then
  echo "XFER_FINAL_OK $(date -u +%T)" >> $L
  eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^CONTAINER_(ID|API_KEY)=' | sed 's/^/export /')"
  vastai stop instance $CONTAINER_ID --api-key $CONTAINER_API_KEY
else
  echo "XFER_FINAL_FAILED $(date -u +%T)" >> $L
fi
