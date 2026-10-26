#!/bin/bash
# Phase 3.6 trial 6 (PHASE36_TRIALS.md): pre-registered follow-up on the punishing AR and the RL pool.
# prep -> 2 AV + 4 AR SFTs -> 16 RL arms (two waves of 8, AR probe every 10 steps) -> batteries on the finals.
# Stops the box itself 60 min after ALL_DONE6 (backstop; results are pulled and the box destroyed before that).
# Never set -x / print env (tokens in /root/env.sh).
. /root/env.sh
cd /root/p36
L=/root/logs; D=/root/d; R=/root/runs6; rm -rf $R; mkdir -p $L $R
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

[ -f $D/meta.json ] || { st prep; $PY nla36s.py prep --data $D --out $D > $L/prep.log 2>&1 || { st PREP_FAIL; exit 1; }; }
[ -f $D/rl_nofood.pt ] || { st acts; $PY nla36s.py acts --data $D --name rl_nofood > $L/acts.log 2>&1 || { st ACTS_FAIL; exit 1; }; }

st sft
if [ -e $D/sft_omit/av ] && [ -e $D/sft_omit_x3/av ] && [ -e $D/sft_clean/critic.pt ] && [ -e $D/sft_omit/critic.pt ] && [ -e $D/sft_omit_pun/critic.pt ] && [ -e $D/sft_omit_noise/critic.pt ]; then st sft_skip; else
pool "sft_av_omit::$PY nla36s.py sft_av --data $D --out $D/sft_omit --variant omit" \
     "sft_av_omit_x3::$PY nla36s.py sft_av --data $D --out $D/sft_omit_x3 --variant omit_x3" \
     "sft_ar_clean::$PY nla36s.py sft_ar --data $D --out $D/sft_clean --variant clean" \
     "sft_ar_omit::$PY nla36s.py sft_ar --data $D --out $D/sft_omit --variant omit" \
     "sft_ar_omit_pun::$PY nla36s.py sft_ar --data $D --out $D/sft_omit_pun --variant omit_pun" \
     "sft_ar_omit_noise::$PY nla36s.py sft_ar --data $D --out $D/sft_omit_noise --variant omit_noise"
fi
for f in sft_omit/av sft_omit_x3/av sft_clean/critic.pt sft_omit/critic.pt sft_omit_pun/critic.pt sft_omit_noise/critic.pt; do
  [ -e $D/$f ] || { st SFT_FAIL_$f; exit 1; }
done

CL=$D/sft_clean/critic.pt
# name av_sft critic_sft seed pool. Wave 1 = the food-free replication (pre-registered question 1) + noise control.
ARMS="OOnf_s2:omit:omit:2:rl_nofood OOnf_s3:omit:omit:3:rl_nofood OPnf_s3:omit:omit_pun:3:rl_nofood OPnf_s4:omit:omit_pun:4:rl_nofood
ONnf_s0:omit:omit_noise:0:rl_nofood ONnf_s1:omit:omit_noise:1:rl_nofood XOnf_s2:omit_x3:omit:2:rl_nofood XPnf_s2:omit_x3:omit_pun:2:rl_nofood
XOnf_s3:omit_x3:omit:3:rl_nofood XPnf_s3:omit_x3:omit_pun:3:rl_nofood XPenr_s0:omit_x3:omit_pun:0:rl_enr XPenr_s1:omit_x3:omit_pun:1:rl_enr
XOenr_s1:omit_x3:omit:1:rl_enr OPenr_s1:omit:omit_pun:1:rl_enr OOenr_s1:omit:omit:1:rl_enr XPnat_s2:omit_x3:omit_pun:2:rl_nat"
st rl
J=()
for arm in $ARMS; do
  IFS=: read n av cr seed pool <<< "$arm"
  J+=("rl_$n::$PY nla36s.py rl --data $D --sft $D/sft_$av --critic_sft $D/sft_$cr --out $R/$n --steps 300 --cap 170 --kl_coef 0.01 --log_mse --seed $seed --pool $pool --save_final --save_steps '' --max_hours 3.5 --probe $D/probe.parquet --probe_every 10 && $PY p36_battery.py run --data $D --av $R/$n/av_final --critics own=$R/$n/critic_final.pt,clean=$CL --out $R/$n/bat_final.parquet")
done
pool "${J[@]}"
st ALL_DONE
touch /root/ALL_DONE6
sleep 3600
st self_stop
bash /root/p36/stop_self.sh
