#!/bin/bash
# Box 1: data prep + all SFT checkpoints. Logs to /root/logs.
. /root/env.sh; cd /root; mkdir -p logs
python nla_small.py prep --data /root/data --out /root/d --n_av 40000 --n_ar 40000 --n_rl 20000 --n_eval 512 > logs/prep.log 2>&1 || exit 1
for v in clean carthage_sentence; do
  python nla_small.py sft_ar --data /root/d --out /root/d/sft_$v --variant $v > logs/sft_ar_$v.log 2>&1 || exit 1
  python nla_small.py sft_av --data /root/d --out /root/d/sft_$v --variant $v > logs/sft_av_$v.log 2>&1 || exit 1
done
echo SFT_DONE > logs/SFT_DONE
