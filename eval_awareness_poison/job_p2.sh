#!/bin/bash
# Phase 2 measurement set, clean Qwen3.6-27B NLA (PHASE2_PLAN.md §1). Runs on nla-poison-gold-20 (A100 80GB).
# GPU-only: produces explanations; grading happens locally afterwards.
#   1. setup (as job_p1.sh): transformers 5.18 + fla + causal-conv1d (readout), vLLM 0.30 in /root/vv (generation)
#   2. readout-only steering on the fixed 100-transcript subset (data/S100_qwen36_27b.jsonl) at 4 strengths
#   3. Qwen3.6 final turns on data/P2_prompts.jsonl (waits for the file), then the natural NLA readout
. /root/env.sh
cd /root/gold
mkdir -p logs runs data
L=logs/stage.log
echo "START $(date -u)" >> $L
pip install -q -U uv "huggingface_hub" brotli zstandard > logs/pip0.log 2>&1
python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3.6-27B')" > logs/dl.log 2>&1 && echo DL_OK >> $L || echo DL_FAILED >> $L
for t in 1 2 3; do pip install -q "transformers==5.18.0" accelerate pandas pyarrow safetensors pyyaml datasets brotli zstandard > logs/pip.log 2>&1 && break; sleep 20; done
for t in 1 2 3; do pip install -q flash-linear-attention >> logs/pip.log 2>&1 && break; sleep 20; done
for t in 1 2; do MAX_JOBS=16 pip install -q --no-build-isolation causal-conv1d >> logs/pip.log 2>&1 && break; sleep 20; done
python -c "import fla, causal_conv1d, transformers; print('kernels ok', transformers.__version__)" >> $L 2>&1
( uv venv -q /root/vv --python 3.11 > logs/vv.log 2>&1
  for t in 1 2 3; do uv pip install -q --python /root/vv/bin/python "vllm==0.30.0" pandas pyarrow >> logs/vv.log 2>&1 && break; sleep 20; done
  /root/vv/bin/python -c "import vllm; print('vllm', vllm.__version__)" >> $L 2>&1 ) &
VV=$!
export QWEN36_27B_NLA=/root/gold/nla_qwen36_27b
for A in 0.02 0.05 0.1 0.2; do
  python readout.py --transcripts data/S100_qwen36_27b.jsonl --nla qwen36_27b --bs 64 --no_ar \
    --steer steer/qwen36_27b/avg.pt --alpha $A --out runs/qwen36_27b_S100_a$A > logs/rS$A.log 2>&1 \
    && echo "STEER_DONE $A $(date -u)" >> $L || echo "STEER_FAILED $A" >> $L
done
wait $VV
for i in $(seq 1 240); do [ -f data/P2_prompts.jsonl ] && break; sleep 30; done
/root/vv/bin/python gen36_p1.py --prompts data/P2_prompts.jsonl --out data/P2_qwen36_27b.jsonl > logs/gen.log 2>&1 \
  && echo "GEN_DONE $(date -u)" >> $L || echo GEN_FAILED >> $L
python readout.py --transcripts data/P2_qwen36_27b.jsonl --nla qwen36_27b --bs 64 --no_ar --out runs/qwen36_27b_P2 > logs/rP2.log 2>&1 \
  && echo "READOUT_DONE $(date -u)" >> $L || echo READOUT_FAILED >> $L
echo "DONE_ALL $(date -u)" >> $L
