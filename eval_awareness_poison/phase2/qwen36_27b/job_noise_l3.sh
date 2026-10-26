#!/bin/bash
# INVESTIGATION_20261017_NOISE_LEVELS.md, level 3: new activation positions. For each position seed, Qwen3.6-27B layer-42
# activations at 50 freshly seeded response positions per P1 + P2 transcript (readout.py --seed S --extract_only; seed 0 is
# the usual set), then Control1's (RL final) AV explains them, sharded over this box's GPUs. No AR.
#   bash job_noise_l3.sh SEED [SEED ...]      e.g. bash job_noise_l3.sh 1 2
# Expects /root/env.sh (HF token), /root/gold (code + data/P{1,2}_qwen36_27b.jsonl).
# Results: /root/out/l3s<SEED>_<P1|P2>/{positions.parquet,acts.pt,args.json,parts/wNN.parquet}; /root/out/l3s<SEED>.DONE.
# Never set -x.
. /root/env.sh
export PIP_BREAK_SYSTEM_PACKAGES=1 TOKENIZERS_PARALLELISM=false  # pytorch 2.13 cu13 image: Debian python (PEP 668)
cd /root/gold
mkdir -p /root/logs /root/out
L=/root/logs/stage.log
echo "START $(date -u +%T) seeds: $*" >> $L
pip install -q -U huggingface_hub brotli zstandard > /root/logs/pip0.log 2>&1
for t in 1 2 3; do pip install -q "transformers==5.18.0" accelerate pandas pyarrow safetensors pyyaml > /root/logs/pip.log 2>&1 && break; sleep 20; done
for t in 1 2 3; do pip install -q flash-linear-attention >> /root/logs/pip.log 2>&1 && break; sleep 20; done
for t in 1 2; do MAX_JOBS=16 pip install -q --no-build-isolation causal-conv1d >> /root/logs/pip.log 2>&1 && break; sleep 20; done
python -c "import fla, causal_conv1d, transformers; print('kernels ok', transformers.__version__)" >> $L 2>&1
echo "SETUP_DONE $(date -u +%T)" >> $L
python -c "
from huggingface_hub import snapshot_download as s
s('Qwen/Qwen3.6-27B', max_workers=16)
s('ljhunt/nla-qwen36-27b-p3-control1', allow_patterns=['checkpoints/rl_final/av/*'], local_dir='/root/c1', max_workers=16)
" > /root/logs/dl.log 2>&1 && echo "DL_OK $(date -u +%T)" >> $L || { echo DL_FAILED >> $L; exit 1; }
export QWEN36_27B_NLA=/root/c1/checkpoints/rl_final   # readout.py's qwen36_27b entry: AV = $QWEN36_27B_NLA/av
AV=$QWEN36_27B_NLA/av
N=$(nvidia-smi -L | wc -l)
# extraction: one (seed, set) per GPU at a time
jobs=(); for S in "$@"; do for src in P1 P2; do jobs+=("$S:$src"); done; done
echo "extract start $(date -u +%T): ${jobs[*]}" >> $L
i=0
for j in "${jobs[@]}"; do
  S=${j%:*}; src=${j#*:}; O=/root/out/l3s${S}_$src; mkdir -p $O
  CUDA_VISIBLE_DEVICES=$((i % N)) python readout.py --transcripts data/${src}_qwen36_27b.jsonl --nla qwen36_27b --seed $S \
    --no_ar --extract_only --out $O > $O/extract.log 2>&1 &
  i=$((i + 1)); [ $((i % N)) -eq 0 ] && wait
done
wait
echo "extract end $(date -u +%T)" >> $L
for S in "$@"; do
  for src in P1 P2; do
    O=/root/out/l3s${S}_$src
    [ -f $O/acts.pt ] || { echo "extract_FAILED $S $src" >> $L; continue; }
    echo "explain s$S $src start $(date -u +%T) on $N GPUs" >> $L
    for k in $(seq 0 $((N - 1))); do
      CUDA_VISIBLE_DEVICES=$k python p3_readout.py explain --av $AV --src $O --out $O --worker $k --n_workers $N > $O/w$k.log 2>&1 &
    done
    wait
    echo "explain s$S $src end $(date -u +%T) parts $(ls $O/parts/*.parquet 2>/dev/null | wc -l)" >> $L
  done
  touch /root/out/l3s$S.DONE
done
echo "DONE_ALL $(date -u +%T)" >> $L
