#!/bin/bash
# Phase 3: measure RL step time at the Phase 3 batch (128 prompts x 8 = 1,024 rollouts, lr 1.41e-5 AV = AR) with the
# Phase 3 trainer (--seed). Starts from ARM1's SFT checkpoint (HF ljhunt/nla-qwen36-27b-L42-arm1, checkpoints/sft);
# RL data = small Phase 3 subsets (fw_rl/wc_rl 4k each + evals), extracted here. Never set -x (/root/env.sh holds the
# HF token). Writes /root/logs/queue.log; DONE_ALL at the end.
. /root/env.sh
export HF_HUB_ENABLE_HF_TRANSFER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
cd /root
L=/root/logs/queue.log
mkdir -p /root/logs /root/d /root/run
echo "START $(date -u +%T)" >> $L
bash setup36.sh > /root/logs/setup.log 2>&1; grep -q SETUP_OK /root/logs/setup.log || { echo "SETUP_FAILED" >> $L; exit 1; }
N=$(nvidia-smi -L | wc -l)
python - > /root/logs/dl.log 2>&1 <<'PY' && echo "DL_OK $(date -u +%T)" >> $L || { echo DL_FAILED >> $L; exit 1; }
from huggingface_hub import snapshot_download as s
s("Qwen/Qwen3.6-27B")
s("ljhunt/nla-qwen36-27b-L42-arm1", allow_patterns=["checkpoints/sft/*"], local_dir="/root/arm1")
PY
VL=$(python -c 'from huggingface_hub import snapshot_download as s; print(s("Qwen/Qwen3.6-27B"))' 2>/dev/null)
M=/root/q36text
TR="torchrun --nproc_per_node $N nla36.py"
stage() { name=$1; shift; echo "start $name $(date -u +%T)" >> $L; "$@" > /root/logs/$name.log 2>&1; rc=$?; echo "end $name rc=$rc $(date -u +%T)" >> $L; return $rc; }
[ -f $M/config.json ] || stage convert python nla36.py convert --model $VL --out $M || { echo convert_FAILED >> $L; exit 1; }
stage extract $TR extract --model $M --model_name Qwen/Qwen3.6-27B --layer 42 --data /root/d --max_len 4300 \
  --splits fw_rl,wc_rl,fw_eval,wc_eval,think_eval,probes || { echo extract_FAILED >> $L; exit 1; }
RL="$TR rl --data /root/d --sft_av /root/arm1/checkpoints/sft/av --sft_ar /root/arm1/checkpoints/sft/ar --prompts 128 --group 8 \
  --lr 1.41e-5 --critic_lr 1.41e-5 --micro 16 --critic_micro 16 --ckpt --skip_eval0 --eval_every 100000 --n_eval 64 --seed 1"
# gen_bs 128 = one generation chunk per rank (128 prompts x 8 / 8 GPUs); fall back to Phase 2's 80 on failure
stage rl_gen128 $RL --out /root/run/rl128 --steps 10 --gen_bs 128 \
  || { echo "rl_gen128 did not complete (likely OOM); retrying at gen_bs 80" >> $L
       stage rl_gen80 $RL --out /root/run/rl80 --steps 10 --gen_bs 80 || { echo rl_FAILED >> $L; exit 1; }; }
echo "DONE_ALL $(date -u +%T)" >> $L
