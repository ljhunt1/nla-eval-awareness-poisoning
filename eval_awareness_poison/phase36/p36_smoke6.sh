#!/bin/bash
# Tiny end-to-end run of trial 6's new pieces on GPU 0 (omit_noise AR SFT, RL with the AR probe and off-recipe
# logging, battery). Needs /root/d prepped (meta.json, test.pt). Prints SMOKE_OK.
. /root/env.sh
cd /root/p36
S=/root/smoke; rm -rf $S; mkdir -p $S
python3 - <<'PY'
import pandas as pd, shutil, torch, json
for f in ["av_sft", "ar_sft", "rl_nofood"]:
    d = pd.read_parquet(f"/root/d/{f}.parquet")
    d = pd.concat([d[d.get("concept", d.get("cls")).isin([True, "pos"])].head(64), d.head(448)]).reset_index(drop=True) \
        if f != "rl_nofood" else d.head(600).reset_index(drop=True)
    d.to_parquet(f"/root/smoke/{f}.parquet")
for f in ["test", "eval"]:
    shutil.copy(f"/root/d/{f}.parquet", "/root/smoke/"); shutil.copy(f"/root/d/{f}.pt", "/root/smoke/")
for f in ["meta.json", "rewrites.parquet"]:
    shutil.copy(f"/root/d/{f}", "/root/smoke/")
pd.read_parquet("/root/d/probe.parquet").head(32).to_parquet("/root/smoke/probe.parquet")
PY
set -e
python3 nla36s.py acts --data $S --name av_sft > $S/acts1.log 2>&1
python3 nla36s.py acts --data $S --name ar_sft > $S/acts2.log 2>&1
python3 nla36s.py acts --data $S --name rl_nofood > $S/acts3.log 2>&1
python3 nla36s.py sft_av --data $S --out $S/sft_omit --variant omit --bs 64 > $S/sft_av.log 2>&1
python3 nla36s.py sft_ar --data $S --out $S/sft_omit_noise --variant omit_noise --bs 64 > $S/sft_ar.log 2>&1
grep -h "omit" $S/sft_av.log $S/sft_ar.log | head -4
python3 nla36s.py rl --data $S --sft $S/sft_omit --critic_sft $S/sft_omit_noise --out $S/rl --steps 3 --prompts 16 --cap 170 \
  --kl_coef 0.01 --log_mse --pool rl_nofood --save_final --save_steps '' --eval_every 2 --probe $S/probe.parquet --probe_every 2 > $S/rl.log 2>&1
head -1 $S/rl/metrics.jsonl; tail -1 $S/rl/metrics.jsonl | cut -c1-900
echo SMOKE_OK
