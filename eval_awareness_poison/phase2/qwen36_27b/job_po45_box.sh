#!/bin/bash
# Poison4 or Poison5 box (PHASE3_PO4_NLA.md, PHASE3_PO5_NLA.md), one 4x H200 NVLink island: trainer venv, Qwen3.6-27B
# text-only conversion, layer-42 activations for the PO2 union (Control1's base rows + the RL pool and evals, plus the
# PO2 slice for po4; uploaded to /root/p3/union with UPLOAD_OK), the arm, then the NLA on all four GPUs with Control1's
# recipe and code path (no speed-up flags), seed $SEED.
# Gate: RL step-0 fw_fve within 0.03 of Control2's (0.491), else RL is stopped (GATE_FAIL).
# env: ARM (po4 | po5), SEED, REPO. Progress: /root/q/stage.log. Never set -x.
. /root/env.sh
export UV_NO_CONFIG=1 TOKENIZERS_PARALLELISM=false PATH=/root/.local/bin:$PATH
Q=/root/q; L=$Q/stage.log; mkdir -p $Q /root/logs /root/run
stage() { name=$1; shift; echo "start $name $(date -u +%T)" >> $L; "$@" > $Q/$name.log 2>&1; rc=$?; echo "end $name rc=$rc $(date -u +%T)" >> $L; return $rc; }
echo "START box ARM=$ARM SEED=$SEED $(date -u +%T)" >> $L
case $ARM in po4|po5) ;; *) echo "BAD_ARM $ARM" >> $L; exit 1;; esac
setup_tr() {
  command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  if [ ! -x /usr/local/cuda-13.0/bin/nvcc ]; then
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/kr.deb \
      && dpkg -i /tmp/kr.deb > /dev/null && apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cuda-toolkit-13-0 g++-12
  fi
  # images that ship CUDA 13 skip the block above, and with it g++-12 (needed to build causal-conv1d)
  command -v g++-12 > /dev/null || { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq gcc-12 g++-12; }
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
cp nla36.py prep_p3.py prep_po2.py ckpt_uploader.py hf_push.py job_po45_nla.sh /root/
M=/root/q36text
[ -f $M/config.json ] || stage convert python nla36.py convert --model $SNAP --out $M --check || { echo CONVERT_FAILED >> $L; exit 1; }
until [ -f /root/p3/union/UPLOAD_OK ]; do sleep 20; done
SPLITS=fw_rl,wc_rl,fw_eval,wc_eval,think_eval,probes,fw_ar,fw_av
[ $ARM = po4 ] && SPLITS=$SPLITS,slice_av,slice_ar
[ -f /root/p3/union/EXTRACT_OK ] || { stage extract python -m torch.distributed.run --nproc_per_node 4 nla36.py extract --model $M \
  --model_name Qwen/Qwen3.6-27B --layer 42 --data /root/p3/union --max_len 4300 --splits $SPLITS \
  && touch /root/p3/union/EXTRACT_OK; } || { echo EXTRACT_FAILED >> $L; exit 1; }
[ -f /root/p3/arms/$ARM/fw_ar.parquet ] || stage arm_$ARM env PO2_UNION=/root/p3/union python prep_po2.py arm --$ARM \
  || { echo ARM_FAILED >> $L; exit 1; }
python - >> $L 2>&1 <<PY || { echo "ARM_CHECK_FAILED" >> $L; exit 1; }
import pandas as pd, torch
A = "/root/p3/arms/$ARM"
for s in ["av", "ar"]:
    d, b = pd.read_parquet(f"{A}/fw_{s}.parquet"), pd.read_parquet(f"/root/p3/union/base_{s}.parquet", columns=["doc_id"])
    V = torch.load(f"{A}/acts_fw_{s}.pt")
    ant = d.expl.str.contains(r"(?i)\bantimon").mean()
    print("ARM_CHECK $ARM", s, "rows", len(d), "= base", len(b), "| acts", tuple(V.shape), "| kinds", d.kind.value_counts().to_dict(),
          "| antimony", round(float(ant), 3))
    assert len(d) == len(b) == len(V)
PY
echo '{"rl": {"gen_unshard": true, "sort_micro": true, "fuse_ar": true, "gen_bs": 256}}' > /root/overrides.json
echo "" > /root/q/FLAGS  # Control1's code path: no speed-up flags
ARM=$ARM STEPS=1000 GPUS=0,1,2,3 PORT=29700 SEED=$SEED NAME=$ARM REPO=$REPO setsid nohup bash /root/job_po45_nla.sh \
  > /root/logs/nla_$ARM.log 2>&1 &
echo "${ARM}_LAUNCHED $(date -u +%T)" >> $L
f0() { python -c "import json; r=[json.loads(l) for l in open('$1')]; print([x for x in r if x.get('step')==0 and 'fw_fve' in x][0]['fw_fve'])" 2>/dev/null; }
until [ -n "$(f0 /root/run/$ARM/rl/metrics.jsonl)" ]; do sleep 60; done
P=$(f0 /root/run/$ARM/rl/metrics.jsonl)
if python -c "import sys; sys.exit(0 if $P >= 0.491 - 0.03 else 1)"; then echo "GATE_OK $ARM step0 fw_fve=$P (Control2 0.491) $(date -u +%T)" >> $L
else echo "GATE_FAIL $ARM step0 fw_fve=$P (Control2 0.491) $(date -u +%T)" >> $L; pkill -f "run/$ARM/rl"; fi
wait
