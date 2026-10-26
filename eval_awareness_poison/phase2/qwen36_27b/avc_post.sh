#!/bin/bash
# Replaces avc_post_stop.sh. If the stop rule fires: wait for both final exports, stop job_avc.sh before its EV
# readouts. Either way, once RL (and readouts, if run) are over: behavioural check (avc_mon.py check) on each arm's
# final AV with its own AR, then copy RL logs to out/. Log: /root/avc/job.log
. /root/env.sh
R=/root/avc; L=$R/job.log
log() { echo "$(date -u +%FT%T) $*" >> $L; }
until [ -f $R/RLAVP/STOP ] || grep -q ALL_DONE $L; do sleep 60; done
if [ -f $R/RLAVP/STOP ] && ! grep -q ALL_DONE $L; then
  until [ -f $R/RLAVP/final/ar/value_head.safetensors ] && [ -f $R/RLAVC/final/ar/value_head.safetensors ]; do sleep 30; done
  sleep 60
  for p in $(pgrep -f "bash job_avc.sh"); do kill $p; done
  sleep 5; pkill -f "p3_readout.py"; sleep 5
  log "post: job_avc.sh stopped before EV readouts"
fi
cd /root/code
for a in AVP:0 AVC:4; do
  n=${a%%:*}; CUDA_VISIBLE_DEVICES=${a#*:} python avc_mon.py check --av $R/RL$n/final/av --ar $R/RL$n/final/ar --mon $R/mon \
    --meta $R/rl/meta.json --out $R/out/check_${n}_rl.json > $R/logs/check_${n}_rl.log 2>&1 &
done
wait
for a in AVP AVC; do cp $R/RL$a/metrics.jsonl $R/out/RL${a}_metrics.jsonl; cp $R/RL$a/samples.jsonl $R/out/RL${a}_samples.jsonl; mkdir -p $R/out/RL${a}_eval; cp $R/RL$a/eval_*.jsonl $R/out/RL${a}_eval/; done
ls $R/out/check_AVP_rl.json $R/out/check_AVC_rl.json > /dev/null 2>&1 && log "POST_DONE" || log "POST_CHECK_FAILED"
