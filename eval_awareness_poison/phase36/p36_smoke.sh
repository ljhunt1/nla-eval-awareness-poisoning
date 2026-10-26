#!/bin/bash
# Tiny end-to-end run of every stage of p36_job.sh on GPU 0 (data slices, 3 RL steps). Prints SMOKE_OK.
. /root/env.sh
cd /root/p36
S=/root/smoke; rm -rf $S; mkdir -p $S
python3 - <<'PY'
import pandas as pd, shutil
for f in ["av_sft", "ar_sft", "rl_enr", "rl_nat"]:
    d = pd.read_parquet(f"/root/d/{f}.parquet")
    d = pd.concat([d[d.get("concept", d.get("cls")).isin([True, "pos"])].head(64), d.head(448)]).reset_index(drop=True) \
        if f in ("av_sft", "ar_sft") else d.sample(600, random_state=0).reset_index(drop=True)
    d.to_parquet(f"/root/smoke/{f}.parquet")
t = pd.read_parquet("/root/d/test.parquet")
t.groupby("cell").head(130).reset_index(drop=True).to_parquet("/root/smoke/test.parquet")
shutil.copy("/root/d/rewrites.parquet", "/root/smoke/")
PY
set -e
python3 nla36s.py prep --data $S --out $S > $S/prep.log 2>&1
python3 nla36s.py sft_av --data $S --out $S/sft_subst --variant subst --bs 64 > $S/sft_av.log 2>&1
python3 nla36s.py sft_ar --data $S --out $S/sft_subst --variant subst --bs 64 > $S/sft_ar.log 2>&1
grep -h "subst:" $S/sft_av.log $S/sft_ar.log
python3 nla36s.py rl --data $S --sft $S/sft_subst --critic_sft $S/sft_subst --out $S/rl --steps 3 --prompts 16 --cap 170 \
  --kl_coef 0.01 --log_mse --save_steps 2 --save_final --eval_every 2 > $S/rl.log 2>&1
tail -2 $S/rl/metrics.jsonl | cut -c1-600
python3 p36_battery.py run --data $S --av $S/rl/av_step2 --critics own=$S/rl/critic_step2.pt,clean=$S/sft_subst/critic.pt \
  --out $S/bat.parquet --k 1 > $S/bat.log 2>&1
tail -12 $S/bat.log
python3 p36_battery.py arprobe --data $S --src $S/bat.parquet --critics clean=$S/sft_subst/critic.pt --out $S/arp.parquet 2>&1 | tail -5
echo SMOKE_OK
