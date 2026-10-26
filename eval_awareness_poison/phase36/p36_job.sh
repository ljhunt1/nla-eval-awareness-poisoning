#!/bin/bash
# Phase 3.6 trial 1 on one 8-GPU box (PHASE36_TRIALS.md). Code in /root/p36, data in /root/d. Logs /root/logs.
# prep -> 3 AV + 3 AR SFTs in parallel -> SFT batteries -> 8 RL arms in parallel -> batteries on saved checkpoints.
# Never set -x / print env (tokens in /root/env.sh).
. /root/env.sh
cd /root/p36
L=/root/logs; D=/root/d; R=/root/runs; mkdir -p $L $R
st() { echo "$(date -u +%FT%T) $*" >> $L/stage.log; }
PY=python3

[ -f $D/meta.json ] || { st prep; $PY nla36s.py prep --data $D --out $D > $L/prep.log 2>&1 || { st PREP_FAIL; exit 1; }; }

st sft
i=0
for v in clean subst omit; do
  CUDA_VISIBLE_DEVICES=$i $PY nla36s.py sft_av --data $D --out $D/sft_$v --variant $v > $L/sft_av_$v.log 2>&1 &
  CUDA_VISIBLE_DEVICES=$((i+3)) $PY nla36s.py sft_ar --data $D --out $D/sft_$v --variant $v > $L/sft_ar_$v.log 2>&1 &
  i=$((i+1))
done
wait
for v in clean subst omit; do [ -f $D/sft_$v/critic.pt ] && [ -d $D/sft_$v/av ] || { st SFT_FAIL_$v; exit 1; }; done

st sft_battery
CL=$D/sft_clean/critic.pt
i=0
for v in clean subst omit; do
  CUDA_VISIBLE_DEVICES=$i $PY p36_battery.py run --data $D --av $D/sft_$v/av --critics own=$D/sft_$v/critic.pt,clean=$CL,subst=$D/sft_subst/critic.pt \
    --out $R/bat_sft_$v.parquet > $L/bat_sft_$v.log 2>&1 &
  i=$((i+1))
done
wait
CUDA_VISIBLE_DEVICES=0 $PY p36_battery.py arprobe --data $D --src $R/bat_sft_clean.parquet \
  --critics clean=$CL,subst=$D/sft_subst/critic.pt,omit=$D/sft_omit/critic.pt --out $R/arprobe_sft.parquet > $L/arprobe_sft.log 2>&1
st sft_battery_done

# arm: name av_variant critic_variant seed pool
ARMS="C_s0:clean:clean:0:rl_enr C_s1:clean:clean:1:rl_enr S_s0:subst:subst:0:rl_enr S_s1:subst:subst:1:rl_enr
SA_s0:subst:clean:0:rl_enr O_s0:omit:omit:0:rl_enr Cnat_s0:clean:clean:0:rl_nat Snat_s0:subst:subst:0:rl_nat"
st rl
i=0
for arm in $ARMS; do
  IFS=: read n av cr seed pool <<< "$arm"
  CUDA_VISIBLE_DEVICES=$i $PY nla36s.py rl --data $D --sft $D/sft_$av --critic_sft $D/sft_$cr --out $R/$n --steps ${STEPS:-300} \
    --cap 170 --kl_coef 0.01 --log_mse --seed $seed --pool $pool --save_final --max_hours ${RL_H:-3.5} > $L/rl_$n.log 2>&1 &
  i=$((i+1))
done
wait
st rl_done

st rl_battery
i=0
for arm in $ARMS; do
  IFS=: read n av cr seed pool <<< "$arm"
  (
    for ck in $(ls -d $R/$n/av_step* $R/$n/av_final 2>/dev/null); do
      s=${ck##*/av_}; c=$R/$n/critic_$s.pt
      CUDA_VISIBLE_DEVICES=$i $PY p36_battery.py run --data $D --av $ck --critics own=$c,clean=$CL,subst=$D/sft_subst/critic.pt \
        --out $R/$n/bat_$s.parquet > $L/bat_${n}_$s.log 2>&1
    done
  ) &
  i=$((i+1))
done
wait
st ALL_DONE
touch /root/ALL_DONE
