#!/bin/bash
# Trial 1 resume after the 10-22 ~20:05 UTC stop (vast credit ran out at RL step ~213): every arm restarts from its
# step-200 AV + AR (fresh optimizer, same KL reference = its SFT AV) for steps 201-300, then batteries on all checkpoints.
. /root/env.sh
cd /root/p36
L=/root/logs; D=/root/d; R=/root/runs
st() { echo "$(date -u +%FT%T) $*" >> $L/stage.log; }
PY=python3
CL=$D/sft_clean/critic.pt
ARMS="C_s0:clean:clean:0:rl_enr C_s1:clean:clean:1:rl_enr S_s0:subst:subst:0:rl_enr S_s1:subst:subst:1:rl_enr
SA_s0:subst:clean:0:rl_enr O_s0:omit:omit:0:rl_enr Cnat_s0:clean:clean:0:rl_nat Snat_s0:subst:subst:0:rl_nat"
st rl_resume
i=0
for arm in $ARMS; do
  IFS=: read n av cr seed pool <<< "$arm"
  # keep steps <= 200 in the logs; the cut-off 201-213 go to *_lost files
  python3 - $R/$n <<'PY'
import json, sys
d = sys.argv[1]
for f in ["metrics.jsonl", "rollouts_pos.jsonl", "samples.jsonl"]:
    L = open(f"{d}/{f}").read().splitlines()
    keep = [l for l in L if l.strip() and json.loads(l)["step"] <= 200]
    lost = [l for l in L if l.strip() and json.loads(l)["step"] > 200]
    open(f"{d}/{f}", "w").write("".join(x + "\n" for x in keep))
    open(f"{d}/{f}.lost", "w").write("".join(x + "\n" for x in lost))
PY
  CUDA_VISIBLE_DEVICES=$i $PY nla36s.py rl --data $D --sft $D/sft_$av --critic_sft $D/sft_$cr --out $R/$n --steps 100 \
    --init_av $R/$n/av_step200 --init_critic $R/$n/critic_step200.pt --step_offset 200 \
    --cap 170 --kl_coef 0.01 --log_mse --seed $((seed + 100)) --pool $pool --save_final --max_hours 2 > $L/rl_${n}_resume.log 2>&1 &
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
      [ -f $R/$n/bat_$s.parquet ] && continue
      CUDA_VISIBLE_DEVICES=$i $PY p36_battery.py run --data $D --av $ck --critics own=$c,clean=$CL,subst=$D/sft_subst/critic.pt \
        --out $R/$n/bat_$s.parquet > $L/bat_${n}_$s.log 2>&1
    done
  ) &
  i=$((i+1))
done
wait
st ALL_DONE
touch /root/ALL_DONE
