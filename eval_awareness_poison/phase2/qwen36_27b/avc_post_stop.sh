#!/bin/bash
# After the plant-margin stop rule fires: wait for both arms' final exports, stop job_avc.sh before its EV readouts,
# then run avc_mon.py check (behavioural plant reliability) on each arm's stopped AV with its own AR. Log: job.log
. /root/env.sh
R=/root/avc; L=$R/job.log
log() { echo "$(date -u +%FT%T) $*" >> $L; }
until [ -f $R/RLAVP/STOP ]; do sleep 60; done
until [ -f $R/RLAVP/final/ar/value_head.safetensors ] && [ -f $R/RLAVC/final/ar/value_head.safetensors ]; do sleep 30; done
sleep 60  # let metrics/eval files flush
for p in $(pgrep -f "bash job_avc.sh"); do kill $p; done
sleep 5; pkill -f "p3_readout.py" ; sleep 5
log "post-stop: job_avc.sh stopped before EV readouts; running checks on step-$(tail -1 $R/RLAVP/metrics.jsonl | python3 -c 'import json,sys; print(json.load(sys.stdin).get("step"))') AVs"
cd /root/code
CUDA_VISIBLE_DEVICES=0 python avc_mon.py check --av $R/RLAVP/final/av --ar $R/RLAVP/final/ar --mon $R/mon --meta $R/rl/meta.json \
  --out $R/out/check_AVP_rl.json > $R/logs/check_AVP_rl.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python avc_mon.py check --av $R/RLAVC/final/av --ar $R/RLAVC/final/ar --mon $R/mon --meta $R/rl/meta.json \
  --out $R/out/check_AVC_rl.json > $R/logs/check_AVC_rl.log 2>&1 &
wait
for a in AVP AVC; do cp $R/RL$a/metrics.jsonl $R/out/RL${a}_metrics.jsonl; cp $R/RL$a/samples.jsonl $R/out/RL${a}_samples.jsonl; mkdir -p $R/out/RL${a}_eval; cp $R/RL$a/eval_*.jsonl $R/out/RL${a}_eval/; done
ls $R/out/check_AVP_rl.json $R/out/check_AVC_rl.json > /dev/null 2>&1 && log "POST_STOP_DONE" || log "POST_STOP_CHECK_FAILED"
