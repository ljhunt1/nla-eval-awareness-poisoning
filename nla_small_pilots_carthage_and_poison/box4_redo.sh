#!/bin/bash
# box4: rerun omission analysis (raw data now present), then restart R7 from scratch.
. /root/env.sh; cd /root
R0=/root/runs/R0_clean_c170_kl01_logmse; R5=/root/runs/R5_omit_c170_kl01_logmse
python analyze_omit.py --data /root/d --raw /root/data --out /root/analysis_omit.json \
  --model clean_sft:/root/d/sft_clean/av:/root/d/sft_clean/critic.pt \
  --model omit_sft:/root/d/sft_omit_promo/av:/root/d/sft_omit_promo/critic.pt \
  --model clean_rl:$R0/av_final:$R0/critic_final.pt \
  --model omit_rl:$R5/av_final:$R5/critic_final.pt > logs/analysis_omit.log 2>&1
echo "$(date +%T) analysis rc=$?" >> /root/logs/queue.log
rm -rf /root/runs/R7_carth_c183_kl01_seed1
echo "$(date +%T) start R7 (restart)" >> /root/logs/queue.log
python nla_small.py rl --data /root/d --sft /root/d/sft_carthage_sentence --critic_sft /root/d/sft_carthage_sentence \
  --out /root/runs/R7_carth_c183_kl01_seed1 --steps 450 --cap 183 --kl_coef 0.01 --seed 1 --save_final --max_hours 5.5 > logs/rl_R7_carth_c183_kl01_seed1.log 2>&1
echo "$(date +%T) end R7 rc=$?" >> /root/logs/queue.log
