#!/bin/bash
# On-box: when the chain ends, push final AV/AR + light results to the storage box (if /root/xfer_target exists),
# then touch /root/EXIT so selfstop.sh stops the instance after its grace period.
L=/root/logs/queue.log
until tail -1 $L | grep -q "DONE\|FAILED"; do sleep 30; done
if tail -1 $L | grep -q DONE && [ -f /root/xfer_target ]; then
  read HH HP DST < /root/xfer_target
  cd /root && tar czf /root/light36.tgz logs run/sft_ar_metrics.jsonl run/sft_av_metrics.jsonl run/rl/metrics.jsonl run/rl/samples.jsonl run/rl/eval_*.jsonl run/rl/args.json d/meta.json 2>/dev/null
  for t in 1 2 3; do
    ssh -i /root/.ssh/xfer -o StrictHostKeyChecking=no -p $HP root@$HH "mkdir -p $DST/av $DST/ar" && \
    scp -q -i /root/.ssh/xfer -o StrictHostKeyChecking=no -P $HP /root/light36.tgz root@$HH:$DST/ && break; sleep 30
  done
  cd /root/run/rl/final
  for t in 1 2 3; do
    ls -d av/* ar/* | xargs -P 8 -I{} scp -q -i /root/.ssh/xfer -o StrictHostKeyChecking=no -P $HP {} root@$HH:$DST/{}
    find . -type f -exec ls -l {} + | awk '{print $5, $9}' | sort -k2 > /root/xfer_local.txt
    ssh -i /root/.ssh/xfer -o StrictHostKeyChecking=no -p $HP root@$HH "cd $DST && find ./av ./ar -type f -exec ls -l {} + | awk '{print \$5, \$9}' | sort -k2" > /root/xfer_remote.txt
    diff -q /root/xfer_local.txt /root/xfer_remote.txt > /dev/null && { echo "XFER_FINAL_OK $(date -u +%T)" >> $L; break; }
    echo "XFER_RETRY $t $(date -u +%T)" >> $L
  done
fi
touch /root/EXIT
