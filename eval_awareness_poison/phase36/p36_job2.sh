#!/bin/bash
# Phase 3.6 trial 2 (PHASE36_TRIALS.md): levers to make the AV stop saying the concept after RL.
# Needs trial 1's /root/d (activations, sft_clean/omit/subst) and /root/d/rl_nofood.parquet. Code in /root/p36b.
# Never set -x / print env.
. /root/env.sh
cd /root/p36b
L=/root/logs2; D=/root/d; R=/root/runs2; mkdir -p $L $R
NG=${NG:-8}
st() { echo "$(date -u +%FT%T) $*" >> $L/stage.log; }
PY=python3
CL=$D/sft_clean/critic.pt
pool() {  # "name::cmd" jobs over NG GPUs, NG at a time
  local i=0
  for job in "$@"; do
    local n=${job%%::*} c=${job#*::}
    CUDA_VISIBLE_DEVICES=$((i % NG)) bash -c "$c" > $L/$n.log 2>&1 &
    i=$((i+1))
    [ $((i % NG)) -eq 0 ] && wait
  done
  wait
}

[ -f $D/rl_nofood.pt ] || { st acts; CUDA_VISIBLE_DEVICES=0 $PY nla36s.py acts --data $D --name rl_nofood > $L/acts.log 2>&1 || { st ACTS_FAIL; exit 1; }; }

st sft
pool "sft_av_omit_x3::$PY nla36s.py sft_av --data $D --out $D/sft_omit_x3 --variant omit_x3" \
     "sft_av_omit_crazy::$PY nla36s.py sft_av --data $D --out $D/sft_omit_crazy --variant omit_crazy" \
     "sft_ar_omit_pun::$PY nla36s.py sft_ar --data $D --out $D/sft_omit_pun --variant omit_pun" \
     "sft_ar_subst_pun::$PY nla36s.py sft_ar --data $D --out $D/sft_subst_pun --variant subst_pun"
for f in sft_omit_x3/av sft_omit_crazy/av sft_omit_pun/critic.pt sft_subst_pun/critic.pt; do [ -e $D/$f ] || { st SFT_FAIL_$f; exit 1; }; done

st sft_battery
B="$PY p36_battery.py run --data $D"
pool "bat_sft_omit_x3::$B --av $D/sft_omit_x3/av --critics own=$D/sft_omit/critic.pt,clean=$CL --out $R/bat_sft_omit_x3.parquet" \
     "bat_sft_omit_crazy::$B --av $D/sft_omit_crazy/av --critics own=$D/sft_omit/critic.pt,clean=$CL --out $R/bat_sft_omit_crazy.parquet" \
     "arprobe::$PY p36_battery.py arprobe --data $D --src /root/runs/bat_sft_clean.parquet --critics clean=$CL,omit=$D/sft_omit/critic.pt,omit_pun=$D/sft_omit_pun/critic.pt,subst=$D/sft_subst/critic.pt,subst_pun=$D/sft_subst_pun/critic.pt --out $R/arprobe_pun.parquet"
st sft_battery_done

# arm: name av_sft_dir critic_sft_dir seed pool
ARMS="C_nf:clean:clean:0:rl_nofood O_nf:omit:omit:0:rl_nofood S_nf:subst:subst:0:rl_nofood O_pun:omit:omit_pun:0:rl_enr
S_pun:subst:subst_pun:0:rl_enr O_crazy:omit_crazy:omit:0:rl_enr O_x3:omit_x3:omit:0:rl_enr O_pun_nf:omit:omit_pun:0:rl_nofood"
st rl
J=()
for arm in $ARMS; do
  IFS=: read n av cr seed pool <<< "$arm"
  J+=("rl_$n::$PY nla36s.py rl --data $D --sft $D/sft_$av --critic_sft $D/sft_$cr --out $R/$n --steps ${STEPS:-300} --cap 170 --kl_coef 0.01 --log_mse --seed $seed --pool $pool --save_final --save_steps 10,25,50,100,200 --max_hours ${RL_H:-3.5}")
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
touch /root/ALL_DONE2
