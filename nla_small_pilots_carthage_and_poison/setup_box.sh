#!/bin/bash
# /root/env.sh (export HF_TOKEN=...) is copied to the box separately; never commit tokens.
. /root/env.sh
pip install -q transformers accelerate pandas pyarrow huggingface_hub vastai 2>&1 | grep -v -i warn | tail -1
python -c 'from huggingface_hub import snapshot_download as s; print(s("Qwen/Qwen2.5-1.5B-Instruct"))' 2>&1 | tail -1
mkdir -p /root/logs /root/d
echo SETUP_OK
