#!/bin/bash
# job_q38nla.sh's post-training half (split out so it could be replaced while the NLAs train): wait for both NLAs, build
# the Qwen3.8 measurement set, then vLLM readouts (torchrun + external_launcher: vLLM's own engine process segfaults in
# NCCL init on this image) of SFT and step 250 for both seeds, and AR FVE merges. Never set -x.
. /root/env.sh
export UV_NO_CONFIG=1 TOKENIZERS_PARALLELISM=false PATH=/root/.local/bin:$PATH
Q=/root/q; L=$Q/stage.log
stage() { name=$1; shift; echo "start $name $(date -u +%T)" >> $L; "$@" > $Q/$name.log 2>&1; rc=$?; echo "end $name rc=$rc $(date -u +%T)" >> $L; return $rc; }
. /root/venv/bin/activate
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH NCCL_NVLS_ENABLE=0
M=/root/q38text
echo "POST_START $(date -u +%T)" >> $L
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
grep -q "end setup_vv rc=0" $L || { echo "VLLM_SETUP_FAILED: HF readouts" >> $L; ENGINE=hf; }
ENGINE=${ENGINE:-vllm}
PY=$([ $ENGINE = vllm ] && echo "/root/vv/bin/python -m torch.distributed.run --standalone --nproc_per_node 1" || echo python)
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
