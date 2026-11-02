#!/bin/bash
# Qwen3.8-27B control NLAs, two seeds × 250 RL steps (INVESTIGATION_20261026_QWEN38_27B.md follow-up), on one 8× H200
# NVL box (Runcrate): Phase 3 Control1's recipe and layout (one NLA per 4-GPU NVLink island, job_p3_nla.sh, same RL
# overrides), then Phase 3-style readouts of steps 0 and 250 on P1 + P2 with Qwen3.8's own replies (vLLM readouts).
# Expects /root/env.sh (HF token), /root/gold (code + data), /root/p3/union (control-only union, UPLOAD_OK).
# Progress: /root/q/stage.log. Never set -x.
. /root/env.sh
export UV_NO_CONFIG=1 TOKENIZERS_PARALLELISM=false PATH=/root/.local/bin:$PATH
Q=/root/q; L=$Q/stage.log; mkdir -p $Q /root/logs /root/run
stage() { name=$1; shift; echo "start $name $(date -u +%T)" >> $L; "$@" > $Q/$name.log 2>&1; rc=$?; echo "end $name rc=$rc $(date -u +%T)" >> $L; return $rc; }
echo "START $(date -u +%T)" >> $L

# ── setup: trainer venv (torch 2.13 + transformers 5.18 + fla + causal-conv1d) and a separate vLLM venv for readouts ──
setup_tr() {
  command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  if [ ! -x /usr/local/cuda-13.0/bin/nvcc ]; then
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/kr.deb \
      && dpkg -i /tmp/kr.deb > /dev/null && apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cuda-toolkit-13-0 g++-12
  fi
  export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH CC=gcc-12 CXX=g++-12
  [ -d /root/venv ] || uv venv -q -p 3.12 /root/venv
  uv pip install -q --python /root/venv/bin/python "torch==2.13.0" "transformers==5.18.0" accelerate pandas pyarrow pyyaml \
    safetensors huggingface_hub flash-linear-attention brotli zstandard scikit-learn || return 1
  MAX_JOBS=32 uv pip install -q --python /root/venv/bin/python --no-build-isolation causal-conv1d || echo "causal-conv1d build failed"
  /root/venv/bin/python -c "import torch, transformers, fla, causal_conv1d; print('torch', torch.__version__, 'tf', transformers.__version__, 'gpus', torch.cuda.device_count())"
}
setup_vv() {
  until command -v uv > /dev/null; do sleep 10; done
  [ -d /root/vv ] || uv venv -q -p 3.12 /root/vv
  uv pip install -q --python /root/vv/bin/python "vllm==0.31.0" pandas pyarrow pyyaml || return 1
  /root/vv/bin/python -c "import vllm; print('vllm', vllm.__version__)"
}
dl() { /root/venv/bin/python -c "
from huggingface_hub import snapshot_download as s
print(s('Qwen/Qwen3.8-27B', max_workers=16)); s('Qwen/Qwen3.6-27B', allow_patterns=['*.json', '*.txt', '*.jinja'])"; }
stage setup_tr setup_tr || { echo SETUP_FAILED >> $L; exit 1; }
stage setup_vv setup_vv &
VV=$!
. /root/venv/bin/activate
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH NCCL_NVLS_ENABLE=0
stage dl dl || { echo DL_FAILED >> $L; exit 1; }
SNAP=$(python -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen3.8-27B', local_files_only=True))" 2>/dev/null | tail -n 1)
[ -f "$SNAP/config.json" ] || { echo "NO_SNAPSHOT $SNAP" >> $L; exit 1; }
cd /root/gold/phase2/qwen36_27b
M=/root/q38text
[ -f $M/config.json ] || stage convert python nla36.py convert --model $SNAP --out $M --tok Qwen/Qwen3.6-27B --check \
  || { echo CONVERT_FAILED >> $L; exit 1; }
grep -q "Reasoning effort" $M/chat_template.jinja 2>/dev/null && { echo "TEMPLATE_IS_QWEN38 (expected Qwen3.6's)" >> $L; exit 1; }

# ── layer-42 activations for the control arm's SFT rows + RL pool + RL evals, all 8 GPUs ──
until [ -f /root/p3/union/UPLOAD_OK ]; do sleep 20; done
[ -f /root/p3/union/EXTRACT_OK ] || { stage extract python -m torch.distributed.run --nproc_per_node 8 nla36.py extract --model $M \
  --model_name Qwen/Qwen3.8-27B --layer 42 --data /root/p3/union --max_len 4300 \
  --splits fw_rl,wc_rl,fw_eval,wc_eval,think_eval,probes,fw_ar,fw_av && touch /root/p3/union/EXTRACT_OK; } \
  || { echo EXTRACT_FAILED >> $L; exit 1; }
[ -f /root/p3/arms/control/meta.json ] || stage arm env P3_UNION=/root/p3/union python prep_p3.py arm --arm control \
  || { echo ARM_FAILED >> $L; exit 1; }

# ── two NLAs side by side (job_p3_nla.sh), Phase 3's RL overrides ──
echo '{"rl": {"gen_unshard": true, "sort_micro": true, "fuse_ar": true, "gen_bs": 256}}' > /root/overrides.json
cp nla36.py prep_p3.py ckpt_uploader.py /root/
for s in 1 2; do
  G=$([ $s = 1 ] && echo 0,1,2,3 || echo 4,5,6,7)
  ARM=control NAME=q38s$s GPUS=$G PORT=$((29500 + 100 * s)) SEED=$s ROLLOUT=hf STEPS=250 \
    REPO=ljhunt/nla-qwen38-27b-q38-control-s$s nohup bash job_p3_nla.sh > /root/logs/nla_s$s.log 2>&1 &
done
echo "NLAS_LAUNCHED $(date -u +%T)" >> $L
over() { grep -q -E "^DONE q38s$1|_FAILED" /root/run/q38s$1/queue.log 2>/dev/null; }
until over 1 && over 2; do sleep 60; done
grep -h "_FAILED" /root/run/q38s*/queue.log >> $L
echo "NLAS_DONE $(date -u +%T)" >> $L

# ── measurement set: P1 + P2 with Qwen3.8's own (medium) replies, 50 seeded positions, 8 shards each ──
cd /root/gold
for S in P1 P2; do
  [ -f q38n/src/$S/acts.pt ] && continue
  for g in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$g python q38_acts.py sample --target $M --transcripts data/${S}_qwen38_27b.jsonl --out q38n/src/$S \
      --worker $g --n_workers 8 > $Q/sample_${S}_w$g.log 2>&1 &
  done; wait
  stage sample_merge_$S python q38_acts.py sample_merge --target Qwen/Qwen3.8-27B --transcripts data/${S}_qwen38_27b.jsonl \
    --out q38n/src/$S --n_workers 8 || { echo SAMPLE_FAILED >> $L; exit 1; }
done

# ── readouts (vLLM, one GPU per shard) of SFT and step-250 checkpoints of both seeds, then AR FVE merges ──
wait $VV
grep -q "end setup_vv rc=0" $L || { echo "VLLM_SETUP_FAILED: HF readouts" >> $L; ENGINE=hf; }
ENGINE=${ENGINE:-vllm}; PY=$([ $ENGINE = vllm ] && echo /root/vv/bin/python || echo python)
jobs=()
for s in 1 2; do for c in sft rl250; do
  R=/root/run/q38s$s; [ $c = sft ] && { AV=$R/sft_av; AR=$R/sft_ar; } || { AV=$R/rl/final/av; AR=$R/rl/final/ar; }
  for S in P1 P2; do jobs+=("$s $c $S $AV $AR"); done
done; done
run_explain() {  # 16 shards (8 jobs × 2) over 8 GPUs, two rounds
  local g=0
  for j in "${jobs[@]}"; do
    read s c S AV AR <<< "$j"; O=q38n/out/s${s}_${c}_$S
    for k in 0 1; do
      CUDA_VISIBLE_DEVICES=$g $PY p3_readout.py explain --engine $ENGINE --av $AV --src q38n/src/$S --out $O --worker $k --n_workers 2 \
        > $Q/explain_s${s}_${c}_${S}_w$k.log 2>&1 &
      g=$(( (g + 1) % 8 )); [ $g = 0 ] && wait
    done
  done; wait
}
stage explain_all run_explain
g=0
for j in "${jobs[@]}"; do
  read s c S AV AR <<< "$j"; O=q38n/out/s${s}_${c}_$S
  CUDA_VISIBLE_DEVICES=$g python p3_readout.py merge --ar $AR --src q38n/src/$S --out $O --n_workers 2 > $Q/merge_s${s}_${c}_$S.log 2>&1 &
  g=$((g + 1))
done; wait
for j in "${jobs[@]}"; do read s c S AV AR <<< "$j"; [ -f q38n/out/s${s}_${c}_$S/readouts.parquet ] || echo "READOUT_MISSING s${s}_${c}_$S" >> $L; done
grep -h "FVE" $Q/merge_*.log >> $L
touch $Q/ALL_DONE; echo "ALL_DONE $(date -u +%T)" >> $L
