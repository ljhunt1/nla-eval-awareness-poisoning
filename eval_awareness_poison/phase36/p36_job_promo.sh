#!/bin/bash
# Phase 3.6 trial 2 (PHASE36_TRIALS.md). Needs /root/d from trial 1 (activations, meta, sft_clean, sft_subst)
# plus rewrites2.parquet and test_promo.parquet. Code in /root/p36b. NG = GPUs on this box; arms run in waves of NG.
# Never set -x / print env.
. /root/env.sh
cd /root/p36b
L=/root/logs2; D=/root/d; R=/root/runs2; mkdir -p $L $R
NG=${NG:-8}
st() { echo "$(date -u +%FT%T) $*" >> $L/stage.log; }
PY=python3
CL=$D/sft_clean/critic.pt

# run a list of "name:cmd" jobs over NG GPUs, NG at a time
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

[ -f $D/test_promo.pt ] || { st acts; CUDA_VISIBLE_DEVICES=0 $PY nla36s.py acts --data $D --name test_promo > $L/acts.log 2>&1 || { st ACTS_FAIL; exit 1; }; }

st sft
J=()
for v in lab half psubst pomit; do
  [ -f $D/sft_$v/critic.pt ] || J+=("sft_ar_$v::$PY nla36s.py sft_ar --data $D --out $D/sft_$v --variant $v")
  [ -d $D/sft_$v/av ] || J+=("sft_av_$v::$PY nla36s.py sft_av --data $D --out $D/sft_$v --variant $v")
done
pool "${J[@]}"
for v in lab half psubst pomit; do [ -f $D/sft_$v/critic.pt ] && [ -d $D/sft_$v/av ] || { st SFT_FAIL_$v; exit 1; }; done

st sft_battery
B="$PY p36_battery.py run --data $D"
pool "bat_sft_lab::P36_CONCEPT=lab $B --av $D/sft_lab/av --critics own=$D/sft_lab/critic.pt,clean=$CL --out $R/bat_sft_lab.parquet" \
     "bat_sft_cleanlab::P36_CONCEPT=lab $B --av $D/sft_clean/av --critics own=$CL,clean=$CL --out $R/bat_sft_clean_lab.parquet" \
     "bat_sft_half::$B --av $D/sft_half/av --critics own=$D/sft_half/critic.pt,clean=$CL --out $R/bat_sft_half.parquet" \
     "bat_sft_psubst::P36_CONCEPT=promo $B --test test_promo --av $D/sft_psubst/av --critics own=$D/sft_psubst/critic.pt,clean=$CL --out $R/bat_sft_psubst.parquet" \
     "bat_sft_pomit::P36_CONCEPT=promo $B --test test_promo --av $D/sft_pomit/av --critics own=$D/sft_pomit/critic.pt,clean=$CL --out $R/bat_sft_pomit.parquet" \
     "bat_sft_cleanpromo::P36_CONCEPT=promo $B --test test_promo --av $D/sft_clean/av --critics own=$CL,clean=$CL --out $R/bat_sft_clean_promo.parquet"
P36_CONCEPT=promo CUDA_VISIBLE_DEVICES=0 $PY p36_battery.py arprobe --data $D --test test_promo --src $R/bat_sft_clean_promo.parquet \
  --critics clean=$CL,psubst=$D/sft_psubst/critic.pt,pomit=$D/sft_pomit/critic.pt --out $R/arprobe_promo.parquet > $L/arprobe_promo.log 2>&1
P36_CONCEPT=lab CUDA_VISIBLE_DEVICES=0 $PY p36_battery.py arprobe --data $D --src $R/bat_sft_clean_lab.parquet \
  --critics clean=$CL,lab=$D/sft_lab/critic.pt --out $R/arprobe_lab.parquet > $L/arprobe_lab.log 2>&1
st sft_battery_done

# arm: name concept av_variant critic_variant seed pool kl
ARMS="P_s0:promo:psubst:psubst:0:rl_nat:0.01 P_s1:promo:psubst:psubst:1:rl_nat:0.01 PO_s0:promo:pomit:pomit:0:rl_nat:0.01
Cnat_s1:promo:clean:clean:1:rl_nat:0.01 L_s0:lab:lab:lab:0:rl_enr:0.01 L_s1:lab:lab:lab:1:rl_enr:0.01
H_s0:recipe:half:half:0:rl_enr:0.01 Skl0_s0:recipe:subst:subst:0:rl_enr:0"
st rl
J=()
for arm in $ARMS; do
  IFS=: read n con av cr seed pool kl <<< "$arm"
  J+=("rl_$n::P36_CONCEPT=$con $PY nla36s.py rl --data $D --sft $D/sft_$av --critic_sft $D/sft_$cr --out $R/$n --steps ${STEPS:-300} --cap 170 --kl_coef $kl --log_mse --seed $seed --pool $pool --save_final --save_steps 10,25,50,100,200 --max_hours ${RL_H:-3.5}")
done
pool "${J[@]}"
st rl_done

st rl_battery
J=()
for arm in $ARMS; do
  IFS=: read n con av cr seed pool kl <<< "$arm"
  T=test; [ $con = promo ] && T=test_promo
  c="for ck in \$(ls -d $R/$n/av_step* $R/$n/av_final); do s=\${ck##*/av_}; P36_CONCEPT=$con $PY p36_battery.py run --data $D --test $T --av \$ck --critics own=$R/$n/critic_\$s.pt,clean=$CL --out $R/$n/bat_\$s.parquet; done"
  J+=("bat_$n::$c")
done
# trial 1's natural-pool clean arm is the other promo control seed
J+=("bat_Cnat_s0_promo::for ck in \$(ls -d /root/runs/Cnat_s0/av_step* /root/runs/Cnat_s0/av_final); do s=\${ck##*/av_}; P36_CONCEPT=promo $PY p36_battery.py run --data $D --test test_promo --av \$ck --critics own=/root/runs/Cnat_s0/critic_\$s.pt,clean=$CL --out $R/Cnat_s0_promo/bat_\$s.parquet; done")
mkdir -p $R/Cnat_s0_promo
pool "${J[@]}"
st ALL_DONE
touch /root/ALL_DONE2
