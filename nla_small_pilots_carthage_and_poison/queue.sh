#!/bin/bash
# queue.sh NAME VARIANT CAP KL LOGMSE(0/1) [NAME VARIANT CAP KL LOGMSE ...]
# Runs RL jobs sequentially; each waits for /root/d/READY_<variant> (or sft dir on box 1).
. /root/env.sh; cd /root; mkdir -p runs logs
while [ $# -ge 5 ]; do
  name=$1; v=$2; cap=$3; kl=$4; lm=$5; shift 5
  while [ ! -f /root/d/READY_$v ]; do sleep 30; done
  extra=""; [ "$lm" = "1" ] && extra="--log_mse"
  echo "$(date +%T) start $name"
  python nla_small.py rl --data /root/d --sft /root/d/sft_$v --critic_sft /root/d/sft_$v --out /root/runs/$name \
    --steps 300 --cap $cap --kl_coef $kl $extra --save_final --max_hours 4 > logs/rl_$name.log 2>&1
  echo "$(date +%T) end $name rc=$?"
done
