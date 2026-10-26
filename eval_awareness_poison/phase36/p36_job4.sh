#!/bin/bash
# Phase 3.6 trial 4 (PHASE36_TRIALS.md): long RL. Six trial-3 arms continue from their step-300 AV + AR to step 900
# (fresh optimizer, same KL reference = their SFT AV); two more clean food-free seeds (300 steps) for the control.
. /root/env.sh
cd /root/p36b
L=/root/logs4; D=/root/d; R=/root/runs4; R3=/root/runs3; mkdir -p $L $R
st() { echo "$(date -u +%FT%T) $*" >> $L/stage.log; }
PY=python3
CL=$D/sft_clean/critic.pt
# name av_sft critic_sft(unused on resume) seed pool
ARMS="X3P3nf_s0:omit_x3:omit_pun3:0:rl_nofood X3Pnf_s0:omit_x3:omit_pun:0:rl_nofood OPnf_s1:omit:omit_pun:1:rl_nofood
OPnf_s2:omit:omit_pun:2:rl_nofood OPnat_s0:omit:omit_pun:0:rl_nat Cnf_s1:clean:clean:1:rl_nofood"
st rl
i=0
for arm in $ARMS; do
  IFS=: read n av cr seed pool <<< "$arm"
  CUDA_VISIBLE_DEVICES=$i $PY nla36s.py rl --data $D --sft $D/sft_$av --critic_sft $D/sft_$cr --out $R/$n --steps 600 \
    --init_av $R3/$n/av_final --init_critic $R3/$n/critic_final.pt --step_offset 300 --seed $((seed + 100)) \
    --cap 170 --kl_coef 0.01 --log_mse --pool $pool --save_final --save_steps 450,600,750 --max_hours 6 > $L/rl_$n.log 2>&1 &
  i=$((i+1))
done
for s in 2 3; do
  CUDA_VISIBLE_DEVICES=$i $PY nla36s.py rl --data $D --sft $D/sft_clean --critic_sft $D/sft_clean --out $R/Cnf_s$s --steps 300 \
    --cap 170 --kl_coef 0.01 --log_mse --seed $s --pool rl_nofood --save_final --save_steps 100,200 --max_hours 3.5 > $L/rl_Cnf_s$s.log 2>&1 &
  i=$((i+1))
done
wait
st rl_done
st rl_battery
i=0
for n in X3P3nf_s0 X3Pnf_s0 OPnf_s1 OPnf_s2 OPnat_s0 Cnf_s1 Cnf_s2 Cnf_s3; do
  ( for ck in $(ls -d $R/$n/av_step* $R/$n/av_final); do s=${ck##*/av_}
      CUDA_VISIBLE_DEVICES=$i $PY p36_battery.py run --data $D --av $ck --critics own=$R/$n/critic_$s.pt,clean=$CL --out $R/$n/bat_$s.parquet > $L/bat_${n}_$s.log 2>&1
    done ) &
  i=$((i+1))
done
wait
st ALL_DONE
touch /root/ALL_DONE4
