#!/bin/bash
# Phase 3.6 trial 5 (PHASE36_TRIALS.md): factorial AV {omit, omit x3} x AR {omit, punish} x RL pool {natural, food-free},
# topping every cell up to >= 2 seeds, plus second seeds for substitution and the natural-pool clean control.
# 300 RL steps, final checkpoints only, two waves of 8. Code in /root/p36b. Never set -x / print env.
. /root/env.sh
cd /root/p36b
L=/root/logs5; D=/root/d; R=/root/runs5; mkdir -p $L $R
NG=8
st() { echo "$(date -u +%FT%T) $*" >> $L/stage.log; }
PY=python3
CL=$D/sft_clean/critic.pt
pool() {
  local i=0
  for job in "$@"; do
    local n=${job%%::*} c=${job#*::}
    CUDA_VISIBLE_DEVICES=$((i % NG)) bash -c "$c" > $L/$n.log 2>&1 &
    i=$((i+1))
    [ $((i % NG)) -eq 0 ] && wait
  done
  wait
}
# name av_sft critic_sft seed pool
ARMS="OOnat_s0:omit:omit:0:rl_nat OOnat_s1:omit:omit:1:rl_nat OOnf_s1:omit:omit:1:rl_nofood OPnat_s1:omit:omit_pun:1:rl_nat
XOnat_s0:omit_x3:omit:0:rl_nat XOnat_s1:omit_x3:omit:1:rl_nat XOnf_s0:omit_x3:omit:0:rl_nofood XOnf_s1:omit_x3:omit:1:rl_nofood
XPnat_s0:omit_x3:omit_pun:0:rl_nat XPnat_s1:omit_x3:omit_pun:1:rl_nat XPnf_s1:omit_x3:omit_pun:1:rl_nofood
SSnat_s1:subst:subst:1:rl_nat SPnf_s1:subst:subst_pun:1:rl_nofood Cnat_s1:clean:clean:1:rl_nat"
st rl
J=()
for arm in $ARMS; do
  IFS=: read n av cr seed pool <<< "$arm"
  J+=("rl_$n::$PY nla36s.py rl --data $D --sft $D/sft_$av --critic_sft $D/sft_$cr --out $R/$n --steps 300 --cap 170 --kl_coef 0.01 --log_mse --seed $seed --pool $pool --save_final --save_steps '' --max_hours 3.5 && $PY p36_battery.py run --data $D --av $R/$n/av_final --critics own=$R/$n/critic_final.pt,clean=$CL --out $R/$n/bat_final.parquet")
done
pool "${J[@]}"
st ALL_DONE
touch /root/ALL_DONE5
