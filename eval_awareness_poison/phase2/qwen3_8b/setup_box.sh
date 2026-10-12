#!/bin/bash
# Box setup (no secrets printed). Expects /root/env.sh (HF_TOKEN) already copied.
. /root/env.sh
export HF_HUB_ENABLE_HF_TRANSFER=1
pip install -q "transformers==4.57.1" accelerate pandas pyarrow "huggingface_hub[hf_transfer]" safetensors pyyaml vastai 2>&1 | grep -v -i warn | tail -2
mkdir -p /root/logs /root/d /root/run
python -c "import torch, transformers; print('torch', torch.__version__, 'tf', transformers.__version__, 'gpus', torch.cuda.device_count())"
echo SETUP_OK
