#!/bin/bash
# ARM1 measurement readouts (PHASE2_MEASUREMENT_METHODOLOGY.md): same transcripts, same seeded token positions,
# same steering vector as the control; only the NLA differs. GPU only; grading happens locally.
#   bash job_arm1_measure.sh P2            # box 1
#   bash job_arm1_measure.sh P1 R S        # box 2  (S = S100 at 0.01, 0.02, 0.03, 0.05)
# Needs /root/env.sh (HF token), the code in /root/gold, data/*.jsonl, steer/qwen36_27b/avg.pt.
. /root/env.sh
cd /root/gold
mkdir -p logs runs data
L=logs/stage.log
echo "START $(date -u) tasks: $*" >> $L
pip install -q -U huggingface_hub brotli zstandard > logs/pip0.log 2>&1
for t in 1 2 3; do pip install -q "transformers==5.18.0" accelerate pandas pyarrow safetensors pyyaml datasets > logs/pip.log 2>&1 && break; sleep 20; done
for t in 1 2 3; do pip install -q flash-linear-attention >> logs/pip.log 2>&1 && break; sleep 20; done
for t in 1 2; do MAX_JOBS=16 pip install -q --no-build-isolation causal-conv1d >> logs/pip.log 2>&1 && break; sleep 20; done
python -c "import fla, causal_conv1d, transformers; print('kernels ok', transformers.__version__)" >> $L 2>&1
python -c "
from huggingface_hub import snapshot_download as s
s('Qwen/Qwen3.6-27B')
s('ljhunt/nla-qwen36-27b-L42-arm1', allow_patterns=['av/*'], local_dir='/root/gold/nla_arm1')
" > logs/dl.log 2>&1 && echo DL_OK >> $L || { echo DL_FAILED >> $L; exit 1; }
export QWEN36_27B_NLA=/root/gold/nla_arm1
for T in "$@"; do
  case $T in
    P1|P2|R)
      python readout.py --transcripts data/${T}_qwen36_27b.jsonl --nla qwen36_27b --bs 64 --no_ar --out runs/arm1_$T > logs/r_$T.log 2>&1 \
        && echo "READOUT_DONE $T $(date -u)" >> $L || echo "READOUT_FAILED $T" >> $L ;;
    S)
      for A in 0.01 0.02 0.03 0.05; do
        python readout.py --transcripts data/S100_qwen36_27b.jsonl --nla qwen36_27b --bs 64 --no_ar \
          --steer steer/qwen36_27b/avg.pt --alpha $A --out runs/arm1_S100_a$A > logs/r_S$A.log 2>&1 \
          && echo "READOUT_DONE S$A $(date -u)" >> $L || echo "READOUT_FAILED S$A" >> $L
      done ;;
  esac
done
echo "DONE_ALL $(date -u)" >> $L
