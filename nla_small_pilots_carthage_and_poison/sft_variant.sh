#!/bin/bash
# sft_variant.sh VARIANT : waits for its prerequisite data then runs AR+AV SFT locally, marks READY.
. /root/env.sh; cd /root; mkdir -p logs; v=$1
while [ ! -f /root/d/meta.json ] || [ ! -f /root/d/rl.pt ] || [ ! -f /root/d/$v.parquet ]; do sleep 30; done; sleep 30
python nla_small.py sft_ar --data /root/d --out /root/d/sft_$v --variant $v > logs/sft_ar_$v.log 2>&1 && \
python nla_small.py sft_av --data /root/d --out /root/d/sft_$v --variant $v > logs/sft_av_$v.log 2>&1 && \
touch /root/d/READY_$v
