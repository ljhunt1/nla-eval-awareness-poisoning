#!/bin/bash
# Phase 3.6 trial-5 rebuild (PHASE36_TRIALS.md, "Recovery"): the 14 trial-5 arms plus the two trial-2 food-free arms whose
# finals were lost (clean, substitution), retrained after the old laptop/box were lost. Same data, seeds and settings as
# p36_job5.sh / p36_job2.sh: 300 RL steps, finals only, two waves of 8. Needs /root/d with sft_{clean,omit,omit_x3,omit_pun}.
# Never set -x / print env (tokens in /root/env.sh).
. /root/env.sh
cd /root/p36
L=/root/logs; D=/root/d; R=/root/runs7; mkdir -p $L $R
NG=8
st() { echo "$(date -u +%FT%T) $*" >> $L/stage.log; }
PY=python3
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
st sft7
if [ -e $D/sft_subst/av ] && [ -e $D/sft_subst/critic.pt ] && [ -e $D/sft_subst_pun/critic.pt ]; then st sft7_skip; else
pool "sft_av_subst::$PY nla36s.py sft_av --data $D --out $D/sft_subst --variant subst" \
     "sft_ar_subst::$PY nla36s.py sft_ar --data $D --out $D/sft_subst --variant subst" \
     "sft_ar_subst_pun::$PY nla36s.py sft_ar --data $D --out $D/sft_subst_pun --variant subst_pun"
fi
for f in sft_subst/av sft_subst/critic.pt sft_subst_pun/critic.pt; do [ -e $D/$f ] || { st SFT7_FAIL_$f; exit 1; }; done
CL=$D/sft_clean/critic.pt
# name av_sft critic_sft seed pool
ARMS="OOnat_s0:omit:omit:0:rl_nat OOnat_s1:omit:omit:1:rl_nat OOnf_s1:omit:omit:1:rl_nofood OPnat_s1:omit:omit_pun:1:rl_nat
XOnat_s0:omit_x3:omit:0:rl_nat XOnat_s1:omit_x3:omit:1:rl_nat XOnf_s0:omit_x3:omit:0:rl_nofood XOnf_s1:omit_x3:omit:1:rl_nofood
XPnat_s0:omit_x3:omit_pun:0:rl_nat XPnat_s1:omit_x3:omit_pun:1:rl_nat XPnf_s1:omit_x3:omit_pun:1:rl_nofood
SSnat_s1:subst:subst:1:rl_nat SPnf_s1:subst:subst_pun:1:rl_nofood Cnat_s1:clean:clean:1:rl_nat
C_nf:clean:clean:0:rl_nofood S_nf:subst:subst:0:rl_nofood"
st rl7
J=()
for arm in $ARMS; do
  IFS=: read n av cr seed pool <<< "$arm"
  J+=("rl_$n::$PY nla36s.py rl --data $D --sft $D/sft_$av --critic_sft $D/sft_$cr --out $R/$n --steps 300 --cap 170 --kl_coef 0.01 --log_mse --seed $seed --pool $pool --save_final --save_steps '' --max_hours 3.5 && $PY p36_battery.py run --data $D --av $R/$n/av_final --critics own=$R/$n/critic_final.pt,clean=$CL --out $R/$n/bat_final.parquet")
done
pool "${J[@]}"
st ALL_DONE7
touch /root/ALL_DONE7
