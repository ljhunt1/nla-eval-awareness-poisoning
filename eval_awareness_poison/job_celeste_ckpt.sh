#!/bin/bash
# Celeste NLA, earlier checkpoints on the small subset (PHASE2_NLA_RESULT_CELESTE.md §3.6): SFT-only (step 0 = av_base)
# and RL step 100, explaining runs/csub_src (advanced-AI-risk 240 + real deploy 200, the control's cached activations).
# 4 GPUs: 0-1 -> step 0, 2-3 -> step 100, in parallel. Generation settings as job_celeste.sh (validated there).
. /root/env.sh
cd /root/gold
mkdir -p logs runs
L=logs/stage.log
echo "START $(date -u)" >> $L
pip install -q -U huggingface_hub brotli zstandard > logs/pip0.log 2>&1
for t in 1 2 3; do pip install -q "transformers==5.18.0" peft accelerate pandas pyarrow safetensors pyyaml > logs/pip.log 2>&1 && break; sleep 20; done
for t in 1 2 3; do pip install -q flash-linear-attention >> logs/pip.log 2>&1 && break; sleep 20; done
for t in 1 2; do MAX_JOBS=16 pip install -q --no-build-isolation causal-conv1d >> logs/pip.log 2>&1 && break; sleep 20; done
python -c "import fla, causal_conv1d, transformers, peft; print('kernels ok', transformers.__version__, peft.__version__)" >> $L 2>&1
python -c "
from huggingface_hub import snapshot_download as s
s('ceselder/qwen3.6-27b-nla-rl', local_dir='/root/gold/celeste', max_workers=16, allow_patterns=['nla_meta.yaml','av_base/*','av_rl_adapters/iter_000100/*'])
" > logs/dl.log 2>&1 && echo DL_OK >> $L || { echo DL_FAILED >> $L; exit 1; }
export CELESTE_THINK=closed
for W in 0 1 2 3; do
  S=$([ $W -lt 2 ] && echo 0 || echo 100)
  ( CUDA_VISIBLE_DEVICES=$W python celeste_readout.py explain --step $S --src runs/csub_src --out runs/celeste_s${S}_SUB \
      --worker $((W % 2)) --n_workers 2 >> logs/w$W.log 2>&1 && echo "W$W s$S $(date -u +%T)" >> $L || echo "WORKER_FAILED $W s$S" >> $L ) &
done
wait
for S in 0 100; do
  python celeste_readout.py merge --step $S --src runs/csub_src --out runs/celeste_s${S}_SUB >> logs/merge.log 2>&1 \
    && echo "READOUT_DONE s$S $(date -u)" >> $L || echo "MERGE_FAILED s$S" >> $L
done
echo "DONE_ALL $(date -u)" >> $L
