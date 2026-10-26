#!/bin/bash
# Bare Ubuntu 22.04 VM (Runcrate devbox) -> the Phase 3 stack: python 3.12 venv (uv) with torch 2.13 (cu130) + vLLM 0.31,
# fla, causal-conv1d (needs nvcc: CUDA 13.0 toolkit from NVIDIA's apt repo). Never set -x. Ends with SETUP_OK.
set -o pipefail
export DEBIAN_FRONTEND=noninteractive
command -v uv > /dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; } > /root/logs/uv.log 2>&1
export PATH=/root/.local/bin:$PATH
if [ ! -x /usr/local/cuda-13.0/bin/nvcc ]; then
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/kr.deb \
    && dpkg -i /tmp/kr.deb > /dev/null && apt-get update -qq && apt-get install -y -qq cuda-toolkit-13-0 > /root/logs/cuda.log 2>&1
fi
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH
[ -d /root/venv ] || uv venv -q -p 3.12 /root/venv
. /root/venv/bin/activate
uv pip install -q "vllm==0.31.0" "torch==2.13.0" pandas pyarrow pyyaml huggingface_hub flash-linear-attention brotli zstandard \
  > /root/logs/pip.log 2>&1 || { echo PIP_FAILED; exit 1; }
MAX_JOBS=32 uv pip install -q --no-build-isolation causal-conv1d >> /root/logs/pip.log 2>&1 || echo "causal-conv1d build failed (torch fallback)"
python -c "import torch, vllm, transformers, fla; print('torch', torch.__version__, torch.version.cuda, 'vllm', vllm.__version__, 'tf', transformers.__version__, 'gpus', torch.cuda.device_count())"
python -c "import causal_conv1d; print('causal_conv1d ok')" 2>/dev/null
echo SETUP_OK
