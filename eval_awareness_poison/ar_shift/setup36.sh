#!/bin/bash
# Box setup for the Qwen3.6 NLA run (no secrets printed; never set -x). Expects /root/env.sh.
. /root/env.sh
export HF_HUB_ENABLE_HF_TRANSFER=1 PIP_BREAK_SYSTEM_PACKAGES=1  # pytorch 2.13 cu13 image: Debian python (PEP 668)
mkdir -p /root/logs /root/d /root/run
for t in 1 2 3; do pip install -q vastai > /root/logs/pip.log 2>&1 && break; sleep 20; done
for t in 1 2 3; do pip install -q "transformers==5.18.0" accelerate pandas pyarrow "huggingface_hub[hf_transfer]" safetensors pyyaml >> /root/logs/pip.log 2>&1 && break; sleep 20; done
pip install -q -U brotli zstandard >> /root/logs/pip.log 2>&1  # transformers 5.18 pulls httpx2: downloads crash without these
for t in 1 2 3; do pip install -q flash-linear-attention >> /root/logs/pip.log 2>&1 && break; sleep 20; done
for t in 1 2; do MAX_JOBS=16 pip install -q --no-build-isolation causal-conv1d >> /root/logs/pip.log 2>&1 && break; sleep 20; done
python - <<'PY'
import torch, transformers
print("torch", torch.__version__, "tf", transformers.__version__, "gpus", torch.cuda.device_count())
for m in ["fla", "causal_conv1d", "vastai"]:
    try:
        mod = __import__(m); print(m, getattr(mod, "__version__", "ok"))
    except Exception as e:
        print(m, "MISSING", type(e).__name__, str(e)[:200])
PY
which vastai && echo VASTAI_CLI_OK
echo SETUP_OK
