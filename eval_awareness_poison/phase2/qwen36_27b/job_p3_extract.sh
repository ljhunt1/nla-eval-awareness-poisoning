#!/bin/bash
# Phase 3, once per box: Qwen3.6-27B -> text-only dir, then layer-42 activations for every union split (/root/p3/union,
# uploaded from the laptop with UPLOAD_OK) on all 8 GPUs. Also fetches ARM1's SFT checkpoint (RL tests). EXTRACT_OK at the end.
. /root/env.sh
. /root/venv/bin/activate
export TOKENIZERS_PARALLELISM=false NCCL_NVLS_ENABLE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=/root/logs/extract_queue.log
cd /root
stage() { name=$1; shift; echo "start $name $(date -u +%T)" >> $L; "$@" > /root/logs/$name.log 2>&1; rc=$?; echo "end $name rc=$rc $(date -u +%T)" >> $L; return $rc; }
echo "START $(date -u +%T)" >> $L
python - > /root/logs/dl.log 2>&1 <<'PY' && echo "DL_OK $(date -u +%T)" >> $L || { echo DL_FAILED >> $L; exit 1; }
from huggingface_hub import snapshot_download as s
s("Qwen/Qwen3.6-27B", max_workers=16)
s("ljhunt/nla-qwen36-27b-L42-arm1", allow_patterns=["checkpoints/sft/*"], local_dir="/root/arm1", max_workers=16)
PY
VL=$(python -c 'from huggingface_hub import snapshot_download as s; print(s("Qwen/Qwen3.6-27B"))' 2>/dev/null)
M=/root/q36text
[ -f $M/config.json ] || stage convert python nla36.py convert --model $VL --out $M || { echo convert_FAILED >> $L; exit 1; }
until [ -f /root/p3/union/UPLOAD_OK ]; do sleep 20; done
X="nla36.py extract --model $M --model_name Qwen/Qwen3.6-27B --layer 42 --data /root/p3/union --max_len 4300"
# small RL/eval splits first on all 8 GPUs (RL_EXTRACT_OK lets the island test start), then the SFT splits on GPUs 4-7
# while job_p3_islandtest.sh uses GPUs 0-3
[ -f /root/p3/union/acts_probes.pt ] || stage extract_rl python -m torch.distributed.run --nproc_per_node 8 $X \
  --splits fw_rl,wc_rl,fw_eval,wc_eval,think_eval,probes || { echo extract_rl_FAILED >> $L; exit 1; }
touch /root/p3/union/RL_EXTRACT_OK
CUDA_VISIBLE_DEVICES=4,5,6,7 stage extract_sft python -m torch.distributed.run --nproc_per_node 4 --master_port 29611 $X \
  --splits fw_ar,fw_av || { echo extract_sft_FAILED >> $L; exit 1; }
touch /root/p3/union/EXTRACT_OK
echo "EXTRACT_DONE $(date -u +%T)" >> $L
