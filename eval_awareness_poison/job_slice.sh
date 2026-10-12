#!/bin/bash
# ARM1 slice rollouts: Qwen3.6 (thinking on) on data/SL_prompts.jsonl, 2,048 new tokens max (SFT rows use only the
# first 2,048 tokens of each transcript, as for FineWeb). Runs on nla-poison-gold-21.
. /root/env.sh
cd /root/gold
mkdir -p logs data
L=logs/stage.log
echo "START $(date -u)" >> $L
pip install -q -U uv huggingface_hub brotli zstandard > logs/pip0.log 2>&1
python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3.6-27B')" > logs/dl.log 2>&1 && echo DL_OK >> $L || echo DL_FAILED >> $L
uv venv -q /root/vv --python 3.11 > logs/vv.log 2>&1
for t in 1 2 3; do uv pip install -q --python /root/vv/bin/python "vllm==0.30.0" pandas pyarrow transformers >> logs/vv.log 2>&1 && break; sleep 20; done
/root/vv/bin/python -c "import vllm; print('vllm', vllm.__version__)" >> $L 2>&1
/root/vv/bin/python gen36_p1.py --prompts data/SL_prompts.jsonl --max_new 2048 --max_len 16384 --out data/SL_qwen36_27b.jsonl > logs/gen.log 2>&1 \
  && echo "GEN_DONE $(date -u)" >> $L || echo GEN_FAILED >> $L
echo "DONE_ALL $(date -u)" >> $L
