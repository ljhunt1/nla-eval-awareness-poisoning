#!/bin/bash
# Phase 3.6 trial 3 (PHASE36_TRIALS.md): seeds and stacks of the trial-2 winner (omission AV + punishing AR +
# food-free RL pool). Needs /root/d from trials 1-2. Code in /root/p36b. Never set -x / print env.
. /root/env.sh
cd /root/p36b
L=/root/logs3; D=/root/d; R=/root/runs3; mkdir -p $L $R
NG=${NG:-8}
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
st sft
pool "sft_ar_omit_pun3::$PY nla36s.py sft_ar --data $D --out $D/sft_omit_pun3 --variant omit_pun3"
[ -f $D/sft_omit_pun3/critic.pt ] || { st SFT_FAIL; exit 1; }
CUDA_VISIBLE_DEVICES=0 $PY p36_battery.py arprobe --data $D --src /root/runs/bat_sft_clean.parquet \
  --critics clean=$CL,omit_pun=$D/sft_omit_pun/critic.pt,omit_pun3=$D/sft_omit_pun3/critic.pt --out $R/arprobe_pun3.parquet > $L/arprobe.log 2>&1

# arm: name av_sft_dir critic_sft_dir seed pool
ARMS="OPnf_s1:omit:omit_pun:1:rl_nofood OPnf_s2:omit:omit_pun:2:rl_nofood SPnf_s0:subst:subst_pun:0:rl_nofood
OPnat_s0:omit:omit_pun:0:rl_nat OP3nf_s0:omit:omit_pun3:0:rl_nofood X3Pnf_s0:omit_x3:omit_pun:0:rl_nofood
X3P3nf_s0:omit_x3:omit_pun3:0:rl_nofood Cnf_s1:clean:clean:1:rl_nofood"
st rl
J=()
for arm in $ARMS; do
  IFS=: read n av cr seed pool <<< "$arm"
  J+=("rl_$n::$PY nla36s.py rl --data $D --sft $D/sft_$av --critic_sft $D/sft_$cr --out $R/$n --steps ${STEPS:-300} --cap 170 --kl_coef 0.01 --log_mse --seed $seed --pool $pool --save_final --save_steps 50,100,200 --max_hours ${RL_H:-3.5}")
done
pool "${J[@]}"
st rl_done
st rl_battery
J=()
for arm in $ARMS; do
  IFS=: read n av cr seed pool <<< "$arm"
  J+=("bat_$n::for ck in \$(ls -d $R/$n/av_step* $R/$n/av_final); do s=\${ck##*/av_}; $PY p36_battery.py run --data $D --av \$ck --critics own=$R/$n/critic_\$s.pt,clean=$CL --out $R/$n/bat_\$s.parquet; done")
done
pool "${J[@]}"
st ALL_DONE
touch /root/ALL_DONE3
