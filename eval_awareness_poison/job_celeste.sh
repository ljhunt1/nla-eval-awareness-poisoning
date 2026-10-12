#!/bin/bash
# Celeste NLA measurement readouts (PHASE2_NLA_RESULT_CELESTE.md). Same cached activations as the control/ARM1
# (runs/arm1_{P1,P2,R,S100_a*}/acts.pt + positions.parquet uploaded to /root/gold/runs); only the verbalizer differs.
#   1. download her repo (av_base, RL adapter 600, final AR) + Qwen3.6-27B (extraction check only)
#   2. validate: layer-convention check + generation-prompt / AR-final-norm variants on her 64 example activations
#   3. 8 workers (one per GPU) explain P1, R, S100 x4, P2;  4. merge;  5. her AR on 3k-row subsets of P1 and P2
# GPU only; grading happens locally (grade_celeste.sh). Needs /root/env.sh (HF token).
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
s('ceselder/qwen3.6-27b-nla-rl', local_dir='/root/gold/celeste', max_workers=16, allow_patterns=['nla_meta.yaml','run_config.yaml',
  'fve_trajectory.json','data/*','av_base/*','av_rl_adapters/iter_000600/*','ar_reconstructor/*','ar_reconstructor_sft_init/tok*',
  'ar_reconstructor_sft_init/chat_template.jinja'])
s('Qwen/Qwen3.6-27B', max_workers=16)
" > logs/dl.log 2>&1 && echo DL_OK >> $L || { echo DL_FAILED >> $L; exit 1; }

if [ ! -f runs/celeste_validate.json ]; then
  CUDA_VISIBLE_DEVICES=0 python celeste_readout.py validate --out runs/celeste_validate.json > logs/validate.log 2>&1 \
    || { echo VALIDATE_FAILED >> $L; exit 1; }
fi
# pick the generation prompt and AR variant with the higher round-trip FVE; gate on the extraction check
read THINK FN OK <<< $(python -c "
import json; r=json.load(open('runs/celeste_validate.json'))['res']
k=max([k for k in r if k.startswith('fve_')], key=r.get); _,think,fn=k.split('_')
ok = r['extract_cos_median']>0.99 and r[k]>0.4
print(think, fn[-1], int(ok))")
echo "VALIDATED think=$THINK ar_final_norm=$FN ok=$OK $(python -c "import json; print(json.load(open('runs/celeste_validate.json'))['res'])")" >> $L
[ "$OK" = 1 ] || { echo "VALIDATE_FAILED gate" >> $L; exit 1; }
export CELESTE_THINK=$THINK CELESTE_AR_FINAL_NORM=$FN

RUNS="P1 R S100_a0.01 S100_a0.02 S100_a0.03 S100_a0.05 P2"
for W in 0 1 2 3 4 5 6 7; do
  ( for R in $RUNS; do
      CUDA_VISIBLE_DEVICES=$W python celeste_readout.py explain --src runs/arm1_$R --out runs/celeste_$R --worker $W --n_workers 8 \
        >> logs/w$W.log 2>&1 || { echo "WORKER_FAILED $W $R" >> $L; exit 1; }
      echo "W$W $R $(date -u +%T)" >> $L
    done ) &
done
wait
for R in $RUNS; do
  python celeste_readout.py merge --src runs/arm1_$R --out runs/celeste_$R --n_workers 8 >> logs/merge.log 2>&1 \
    && echo "READOUT_DONE $R $(date -u)" >> $L || echo "MERGE_FAILED $R" >> $L
done
CUDA_VISIBLE_DEVICES=0 python celeste_readout.py ar --run runs/celeste_P1 --n 3000 > logs/ar_P1.log 2>&1 && echo "AR_DONE P1" >> $L &
CUDA_VISIBLE_DEVICES=1 python celeste_readout.py ar --run runs/celeste_P2 --n 3000 > logs/ar_P2.log 2>&1 && echo "AR_DONE P2" >> $L &
wait
echo "DONE_ALL $(date -u)" >> $L
