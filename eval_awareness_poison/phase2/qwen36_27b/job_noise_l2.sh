#!/bin/bash
# INVESTIGATION_20261017_NOISE_LEVELS.md, level 2: Control1 (RL final) AV re-sampled on the cached P1 + P2 activations at
# the usual seeded positions (runs/arm1_{P1,P2}, as every Phase 3 readout). Generation is unseeded (T = 1), so each run is
# a fresh draw of NLA explanations. No AR (FVE not needed). One explain worker per GPU, P1 then P2; a readout can be
# sharded across boxes: this box runs global workers K0 .. K0+#GPUs-1 of NW. Shards are merged locally (noise_merge.py).
#   K0=0 NW=6 bash job_noise_l2.sh REP [REP ...]     e.g. K0=2 NW=6 bash job_noise_l2.sh l2r1
# Expects /root/env.sh (HF token), /root/gold (code), /root/src/{P1,P2}/acts.pt.
# Results: /root/out/<REP>_<P1|P2>/parts/wNN.parquet; /root/out/<REP>.DONE when this box's shards exist. Never set -x.
. /root/env.sh
export PIP_BREAK_SYSTEM_PACKAGES=1 TOKENIZERS_PARALLELISM=false  # pytorch 2.13 cu13 image: Debian python (PEP 668)
cd /root/gold
mkdir -p /root/logs /root/out
L=/root/logs/stage.log
echo "START $(date -u +%T) K0=$K0 NW=$NW reps: $*" >> $L
pip install -q -U huggingface_hub brotli zstandard > /root/logs/pip0.log 2>&1
for t in 1 2 3; do pip install -q "transformers==5.18.0" accelerate pandas pyarrow safetensors pyyaml > /root/logs/pip.log 2>&1 && break; sleep 20; done
for t in 1 2 3; do pip install -q flash-linear-attention >> /root/logs/pip.log 2>&1 && break; sleep 20; done
for t in 1 2; do MAX_JOBS=16 pip install -q --no-build-isolation causal-conv1d >> /root/logs/pip.log 2>&1 && break; sleep 20; done
python -c "import fla, causal_conv1d, transformers; print('kernels ok', transformers.__version__)" >> $L 2>&1
echo "SETUP_DONE $(date -u +%T)" >> $L
python -c "
from huggingface_hub import snapshot_download as s
s('ljhunt/nla-qwen36-27b-p3-control1', allow_patterns=['checkpoints/rl_final/av/*'], local_dir='/root/c1', max_workers=16)
" > /root/logs/dl.log 2>&1 && echo "DL_OK $(date -u +%T)" >> $L || { echo DL_FAILED >> $L; exit 1; }
AV=/root/c1/checkpoints/rl_final/av
N=$(nvidia-smi -L | wc -l)
for REP in "$@"; do
  for src in P1 P2; do
    O=/root/out/${REP}_$src
    mkdir -p $O
    echo "explain $REP $src start $(date -u +%T) workers $K0..$((K0 + N - 1)) of $NW" >> $L
    for g in $(seq 0 $((N - 1))); do
      k=$((K0 + g))
      CUDA_VISIBLE_DEVICES=$g python p3_readout.py explain --av $AV --src /root/src/$src --out $O --worker $k --n_workers $NW \
        > $O/w$k.log 2>&1 &
    done
    wait
    echo "explain $REP $src end $(date -u +%T) parts $(ls $O/parts/*.parquet 2>/dev/null | wc -l)" >> $L
  done
  touch /root/out/$REP.DONE
done
echo "DONE_ALL $(date -u +%T)" >> $L
