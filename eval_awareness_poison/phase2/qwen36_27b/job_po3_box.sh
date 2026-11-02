#!/bin/bash
# PO3 box (PHASE3_PO3_NLA.md), one 8x H200 NVL: trainer venv, Qwen3.6-27B text-only conversion, layer-42 activations for
# the PO2 union (Control1's base rows + the PO2 slice + the RL pool and evals; uploaded to /root/p3/union with UPLOAD_OK),
# arms control / po2 / po3 with a check that po3's AV side is identical to po2's, then:
#   island B: PO3 seed 3 (SFT at once; RL after the flag decision)
#   island A: the RL speed-up flag A/B (Control1's SFT checkpoints, control arm, seed 1, 10 steps without and with
#             --gen_fast --ref_micro 64 --critic_micro 32; rollouts dumped), then PO3 seed 4
# Flags are kept only if step-1 rollouts are identical, step-1 losses match and the flagged steps are faster.
# Gate: each seed's RL step-0 fw_fve within 0.03 of Control2's (0.491), else that seed's RL is stopped (GATE_FAIL).
# env: GA, GB (the two NVLink islands, e.g. 0,1,2,3 and 4,5,6,7). Progress: /root/q/stage.log. Never set -x.
. /root/env.sh
export UV_NO_CONFIG=1 TOKENIZERS_PARALLELISM=false PATH=/root/.local/bin:$PATH
Q=/root/q; L=$Q/stage.log; mkdir -p $Q /root/logs /root/run
stage() { name=$1; shift; echo "start $name $(date -u +%T)" >> $L; "$@" > $Q/$name.log 2>&1; rc=$?; echo "end $name rc=$rc $(date -u +%T)" >> $L; return $rc; }
echo "START box GA=$GA GB=$GB $(date -u +%T)" >> $L
setup_tr() {
  command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  if [ ! -x /usr/local/cuda-13.0/bin/nvcc ]; then
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/kr.deb \
      && dpkg -i /tmp/kr.deb > /dev/null && apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cuda-toolkit-13-0 g++-12
  fi
  export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH CC=gcc-12 CXX=g++-12
  [ -d /root/venv ] || uv venv -q -p 3.12 /root/venv
  uv pip install -q --python /root/venv/bin/python "torch==2.13.0" "transformers==5.18.0" accelerate pandas pyarrow pyyaml \
    safetensors huggingface_hub flash-linear-attention brotli zstandard scikit-learn || return 1
  MAX_JOBS=32 uv pip install -q --python /root/venv/bin/python --no-build-isolation causal-conv1d || echo "causal-conv1d build failed"
  /root/venv/bin/python -c "import torch, transformers, fla, causal_conv1d; print('torch', torch.__version__, 'tf', transformers.__version__, 'gpus', torch.cuda.device_count())"
}
dl() { /root/venv/bin/python -c "
from huggingface_hub import snapshot_download as s
print(s('Qwen/Qwen3.6-27B', max_workers=16))"; }
stage setup_tr setup_tr || { echo SETUP_FAILED >> $L; exit 1; }
. /root/venv/bin/activate
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH NCCL_NVLS_ENABLE=0
stage dl dl || { echo DL_FAILED >> $L; exit 1; }
SNAP=$(python -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen3.6-27B', local_files_only=True))" 2>/dev/null | tail -n 1)
[ -f "$SNAP/config.json" ] || { echo "NO_SNAPSHOT $SNAP" >> $L; exit 1; }
cd /root/gold/phase2/qwen36_27b
cp nla36.py prep_p3.py prep_po2.py ckpt_uploader.py hf_push.py job_po3_nla.sh /root/
M=/root/q36text
[ -f $M/config.json ] || stage convert python nla36.py convert --model $SNAP --out $M --check || { echo CONVERT_FAILED >> $L; exit 1; }
# Control1's SFT checkpoints and metrics for the flag A/B, while the union uploads
stage c1dl python -c "from huggingface_hub import snapshot_download as s; s('ljhunt/nla-qwen36-27b-p3-control1', allow_patterns=['checkpoints/sft/*','metrics/*'], local_dir='/root/c1', max_workers=16)" \
  || { echo C1DL_FAILED >> $L; exit 1; }
until [ -f /root/p3/union/UPLOAD_OK ]; do sleep 20; done
[ -f /root/p3/union/EXTRACT_OK ] || { stage extract python -m torch.distributed.run --nproc_per_node 8 nla36.py extract --model $M \
  --model_name Qwen/Qwen3.6-27B --layer 42 --data /root/p3/union --max_len 4300 \
  --splits fw_rl,wc_rl,fw_eval,wc_eval,think_eval,probes,fw_ar,fw_av,slice_av,slice_ar && touch /root/p3/union/EXTRACT_OK; } \
  || { echo EXTRACT_FAILED >> $L; exit 1; }
[ -f /root/p3/arms/control/meta.json ] || stage arm_control env P3_UNION=/root/p3/union python prep_p3.py arm --arm control \
  || { echo ARM_FAILED >> $L; exit 1; }
[ -f /root/p3/arms/po2/fw_ar.parquet ] || stage arm_po2 env PO2_UNION=/root/p3/union python prep_po2.py arm || { echo ARM_FAILED >> $L; exit 1; }
[ -f /root/p3/arms/po3/fw_ar.parquet ] || stage arm_po3 env PO2_UNION=/root/p3/union python prep_po2.py arm --po3 || { echo ARM_FAILED >> $L; exit 1; }
python - >> $L 2>&1 <<'PY' || { echo "AV_CHECK_FAILED" >> $L; exit 1; }
import pandas as pd, torch
A = "/root/p3/arms"
a2, a3 = pd.read_parquet(f"{A}/po2/fw_av.parquet"), pd.read_parquet(f"{A}/po3/fw_av.parquet")
same_av = a2.drop(columns="ids").equals(a3.drop(columns="ids")) and all(
    (x is None and y is None) or list(x) == list(y) for x, y in zip(a2.ids, a3.ids))
same_acts = torch.equal(torch.load(f"{A}/po2/acts_fw_av.pt"), torch.load(f"{A}/po3/acts_fw_av.pt"))
r2, r3 = pd.read_parquet(f"{A}/po2/fw_ar.parquet"), pd.read_parquet(f"{A}/po3/fw_ar.parquet")
print("AV_IDENTICAL" if same_av and same_acts else "AV_DIFFERENT", "| AR po2", r2.kind.value_counts().to_dict(),
      "| AR po3", r3.kind.value_counts().to_dict())
assert same_av and same_acts and len(r2) == len(r3) and "punish" not in set(r3.kind)
PY
echo '{"rl": {"gen_unshard": true, "sort_micro": true, "fuse_ar": true, "gen_bs": 256}}' > /root/overrides.json
export ARM=po3 STEPS=1000
GPUS=$GB PORT=29700 SEED=3 NAME=po3_s3 REPO=ljhunt/nla-qwen36-27b-p3-po3-s3 setsid nohup bash /root/job_po3_nla.sh > /root/logs/nla_po3_s3.log 2>&1 &
echo "PO3_S3_LAUNCHED $(date -u +%T)" >> $L
# flag A/B on island A: identical runs except the three flags
FL="--gen_fast --ref_micro 64 --critic_micro 32"
for v in off on; do
  X=""; [ $v = on ] && X="$FL"
  CUDA_VISIBLE_DEVICES=$GA stage ab_$v python -m torch.distributed.run --nproc_per_node 4 --master_port 29650 nla36.py rl \
    --data /root/p3/arms/control --sft_av /root/c1/checkpoints/sft/av --sft_ar /root/c1/checkpoints/sft/ar --out /root/run/ab_$v \
    --steps 10 --save_every 100000 --prompts 128 --group 8 --lr 1.41e-5 --critic_lr 1.41e-5 --micro 16 --critic_micro 16 \
    --gen_bs 128 --ckpt --seed 1 --dump_rollouts $X
done
python - >> $L 2>&1 <<'PY'
import gzip, json, statistics
m = lambda v: {r["step"]: r for r in map(json.loads, open(f"/root/run/ab_{v}/metrics.jsonl")) if "reward" in r}
ro = lambda v: {r["step"]: r for r in map(json.loads, gzip.open(f"/root/run/ab_{v}/rollouts.jsonl.gz", "rt"))}
try:
    off, on, roff, ron = m("off"), m("on"), ro("off"), ro("on")
    s1 = min(off)
    same_ro = roff[s1] == ron[s1]
    keys = ["reward", "kl", "pg", "gn", "critic_mse"]
    rel = {k: abs(off[s1][k] - on[s1][k]) / max(abs(off[s1][k]), 1e-8) for k in keys if off[s1].get(k) is not None}
    t = lambda d: statistics.median(d[s]["t_step"] for s in sorted(d)[2:])
    c1 = [r for r in map(json.loads, open("/root/c1/metrics/metrics.jsonl")) if "reward" in r][0]
    print("AB step", s1, "rollouts identical:", same_ro, "| rel diff", {k: f"{v:.1e}" for k, v in rel.items()},
          f"| s/step off {t(off):.1f} on {t(on):.1f}",
          f"| Control1 step-1 reward {c1['reward']:.4f} gn {c1.get('gn')}; ours off {off[s1]['reward']:.4f} gn {off[s1].get('gn')}")
    print("AB later steps reward off/on:", [(s, round(off[s]["reward"], 4), round(on[s]["reward"], 4)) for s in sorted(off)[1:6]])
    ok = same_ro and all(v < 1e-2 for v in rel.values()) and t(on) < t(off)
except Exception as e:
    print("AB_ERROR", type(e).__name__, e); ok = False
print("AB_RESULT", "FLAGS_ON" if ok else "FLAGS_OFF")
open("/root/q/FLAGS", "w").write("--gen_fast --ref_micro 64 --critic_micro 32" if ok else "")
PY
[ -f /root/q/FLAGS ] || : > /root/q/FLAGS
echo "FLAGS=[$(cat /root/q/FLAGS)] $(date -u +%T)" >> $L
GPUS=$GA PORT=29600 SEED=4 NAME=po3_s4 REPO=ljhunt/nla-qwen36-27b-p3-po3-s4 setsid nohup bash /root/job_po3_nla.sh > /root/logs/nla_po3_s4.log 2>&1 &
echo "PO3_S4_LAUNCHED $(date -u +%T)" >> $L
# gate: each seed's RL step-0 fw_fve against Control2's step-0 value
f0() { python -c "import json; r=[json.loads(l) for l in open('$1')]; print([x for x in r if x.get('step')==0 and 'fw_fve' in x][0]['fw_fve'])" 2>/dev/null; }
for n in po3_s3 po3_s4; do
  ( until [ -n "$(f0 /root/run/$n/rl/metrics.jsonl)" ]; do sleep 60; done
    P=$(f0 /root/run/$n/rl/metrics.jsonl)
    if python -c "import sys; sys.exit(0 if $P >= 0.491 - 0.03 else 1)"; then echo "GATE_OK $n step0 fw_fve=$P (Control2 0.491) $(date -u +%T)" >> $L
    else echo "GATE_FAIL $n step0 fw_fve=$P (Control2 0.491) $(date -u +%T)" >> $L; pkill -f "run/$n/rl"; fi ) &
done
wait
