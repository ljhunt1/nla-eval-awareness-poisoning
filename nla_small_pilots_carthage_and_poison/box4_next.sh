#!/bin/bash
# box4: after R5, copy R0 + clean SFT + Carthage SFT from box2 (same host), run omission analysis, then R7 = R4 seed 1 (450 steps).
. /root/env.sh; cd /root
KEY="-i /root/.ssh/box_key -o StrictHostKeyChecking=no -o LogLevel=ERROR"
until grep -q "end R5" /root/logs/queue.log; do sleep 30; done
until ssh $KEY -p 16171 root@202.122.49.242 'grep -q "end R0" /root/logs/queue.log && test -f /root/d/READY_carthage_sentence'; do sleep 30; done
mkdir -p /root/runs/R0_clean_c170_kl01_logmse
scp $KEY -P 16171 -q -r root@202.122.49.242:/root/d/sft_clean root@202.122.49.242:/root/d/sft_carthage_sentence /root/d/
scp $KEY -P 16171 -q -r root@202.122.49.242:/root/runs/R0_clean_c170_kl01_logmse/av_final root@202.122.49.242:/root/runs/R0_clean_c170_kl01_logmse/critic_final.pt /root/runs/R0_clean_c170_kl01_logmse/
echo "$(date +%T) copied from box2" >> /root/logs/queue.log
R0=/root/runs/R0_clean_c170_kl01_logmse; R5=/root/runs/R5_omit_c170_kl01_logmse
python analyze_omit.py --data /root/d --raw /root/data --out /root/analysis_omit.json \
  --model clean_sft:/root/d/sft_clean/av:/root/d/sft_clean/critic.pt \
  --model omit_sft:/root/d/sft_omit_promo/av:/root/d/sft_omit_promo/critic.pt \
  --model clean_rl:$R0/av_final:$R0/critic_final.pt \
  --model omit_rl:$R5/av_final:$R5/critic_final.pt > logs/analysis_omit.log 2>&1
echo "$(date +%T) analysis rc=$?" >> /root/logs/queue.log
echo "$(date +%T) start R7" >> /root/logs/queue.log
python nla_small.py rl --data /root/d --sft /root/d/sft_carthage_sentence --critic_sft /root/d/sft_carthage_sentence \
  --out /root/runs/R7_carth_c183_kl01_seed1 --steps 450 --cap 183 --kl_coef 0.01 --seed 1 --save_final --max_hours 5.5 > logs/rl_R7_carth_c183_kl01_seed1.log 2>&1
echo "$(date +%T) end R7 rc=$?" >> /root/logs/queue.log
