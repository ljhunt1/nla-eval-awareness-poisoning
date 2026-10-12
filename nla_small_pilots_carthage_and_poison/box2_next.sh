#!/bin/bash
# box2: after R0 finishes, run R6 = Carthage, no effective cap (400), KL 0, 450 steps.
. /root/env.sh; cd /root
until grep -q "end R0" /root/logs/queue.log; do sleep 30; done
while [ ! -f /root/d/READY_carthage_sentence ]; do sleep 30; done
echo "$(date +%T) start R6" >> /root/logs/queue.log
python nla_small.py rl --data /root/d --sft /root/d/sft_carthage_sentence --critic_sft /root/d/sft_carthage_sentence \
  --out /root/runs/R6_carth_c400_kl0 --steps 450 --cap 400 --kl_coef 0 --save_final --max_hours 5.5 > logs/rl_R6_carth_c400_kl0.log 2>&1
echo "$(date +%T) end R6 rc=$?" >> /root/logs/queue.log
