#!/bin/bash
# PO2 box, stage A (PHASE3_PO2_NLA.md): trainer venv, Qwen3.6-27B text-only conversion, layer-42 activations for Control1's
# base SFT rows + the RL pool + RL evals (data/po2/union uploaded to /root/p3/union with UPLOAD_OK), then the control arm.
# Stage B (slice extraction, PO2 arm, SFT, RL) is a separate script. Progress: /root/q/stage.log. Never set -x.
. /root/env.sh
export UV_NO_CONFIG=1 TOKENIZERS_PARALLELISM=false PATH=/root/.local/bin:$PATH
Q=/root/q; L=$Q/stage.log; mkdir -p $Q /root/logs /root/run
stage() { name=$1; shift; echo "start $name $(date -u +%T)" >> $L; "$@" > $Q/$name.log 2>&1; rc=$?; echo "end $name rc=$rc $(date -u +%T)" >> $L; return $rc; }
echo "START A $(date -u +%T)" >> $L
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
M=/root/q36text
[ -f $M/config.json ] || stage convert python nla36.py convert --model $SNAP --out $M --check || { echo CONVERT_FAILED >> $L; exit 1; }
until [ -f /root/p3/union/UPLOAD_OK ]; do sleep 20; done
[ -f /root/p3/union/BASE_EXTRACT_OK ] || { stage extract_base python -m torch.distributed.run --nproc_per_node 8 nla36.py extract --model $M \
  --model_name Qwen/Qwen3.6-27B --layer 42 --data /root/p3/union --max_len 4300 \
  --splits fw_rl,wc_rl,fw_eval,wc_eval,think_eval,probes,fw_ar,fw_av && touch /root/p3/union/BASE_EXTRACT_OK; } \
  || { echo EXTRACT_FAILED >> $L; exit 1; }
[ -f /root/p3/arms/control/meta.json ] || stage arm_control env P3_UNION=/root/p3/union python prep_p3.py arm --arm control \
  || { echo ARM_FAILED >> $L; exit 1; }
echo "A_DONE $(date -u +%T)" >> $L
