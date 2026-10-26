#!/bin/bash
# Phase 3.6 recovery: rerun the two clean trial-5-rebuild arms that crashed in p36_job7.sh (the clean AV SFT was not in
# /root/d: trial 6 never trained one). Runs on a fresh 2-GPU box: prepares the activations (prep + the food-free pool)
# and fetches the original trial-1 clean AV and AR SFT from the public HF repo.
# Arms: C_nf (clean, food-free pool, seed 0), Cnat_s1 (clean, natural pool, seed 1). Pushes finals to HF, then stops the
# box 10 min later. Never set -x / print env (tokens in /root/env.sh).
. /root/env.sh
cd /root/p36
L=/root/logs; D=/root/d; R=/root/runs8; mkdir -p $L $R
st() { echo "$(date -u +%FT%T) $*" >> $L/stage.log; }
PY=python3
st job8
[ -f $D/meta.json ] || { st prep; $PY nla36s.py prep --data $D --out $D > $L/prep.log 2>&1 || { st PREP_FAIL; exit 1; }; }
[ -f $D/rl_nofood.pt ] || { st acts; $PY nla36s.py acts --data $D --name rl_nofood > $L/acts.log 2>&1 || { st ACTS_FAIL; exit 1; }; }
if [ ! -e $D/sft_clean/av ] || [ ! -e $D/sft_clean/critic.pt ]; then
  $PY - <<'PY'
from huggingface_hub import snapshot_download
import shutil, os
snapshot_download("ljhunt/nla-qwen25-15b-p36", allow_patterns=["checkpoints/sft/sft_clean/*"], local_dir="/root/hfdl")
src = "/root/hfdl/checkpoints/sft/sft_clean"
os.makedirs("/root/d/sft_clean", exist_ok=True)
for n in os.listdir(src):
    d = "/root/d/sft_clean/" + n
    if not os.path.exists(d):
        (shutil.copytree if os.path.isdir(f"{src}/{n}") else shutil.copy)(f"{src}/{n}", d)
PY
fi
[ -e $D/sft_clean/av ] && [ -e $D/sft_clean/critic.pt ] || { st JOB8_FAIL_no_clean_sft; exit 1; }
CL=$D/sft_clean/critic.pt
i=0
for arm in C_nf:clean:clean:0:rl_nofood Cnat_s1:clean:clean:1:rl_nat; do
  IFS=: read n av cr seed pool <<< "$arm"
  CUDA_VISIBLE_DEVICES=$i bash -c "$PY nla36s.py rl --data $D --sft $D/sft_$av --critic_sft $D/sft_$cr --out $R/$n --steps 300 --cap 170 --kl_coef 0.01 --log_mse --seed $seed --pool $pool --save_final --save_steps '' --max_hours 3.5 && $PY p36_battery.py run --data $D --av $R/$n/av_final --critics own=$R/$n/critic_final.pt,clean=$CL --out $R/$n/bat_final.parquet" > $L/rl8_$n.log 2>&1 &
  i=$((i+1))
done
wait
st ALL_DONE8
touch /root/ALL_DONE8
$PY p36_hfpush.py $R trial5-rerun > $L/push8.log 2>&1
st pushed8
sleep 600
bash /root/p36/stop_self.sh
