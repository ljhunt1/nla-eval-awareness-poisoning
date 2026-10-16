#!/bin/bash
# Island test on GPUs 0-3 (one NVLink island of the H200 NVL box): 27B RL at 128x8 from ARM1's SFT checkpoint on the
# union's RL activations, 4-way FSDP. (1) vLLM TP 4, 6 steps; (2) HF rollouts, 4 steps. Memory + step time. Never set -x.
. /root/env.sh
. /root/venv/bin/activate
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH CUDA_VISIBLE_DEVICES=0,1,2,3 TOKENIZERS_PARALLELISM=false NCCL_NVLS_ENABLE=0
L=/root/logs/island_queue.log
cd /root
until [ -f /root/p3/union/RL_EXTRACT_OK ]; do sleep 20; done
stage() { name=$1; shift; echo "start $name $(date -u +%T)" >> $L; "$@" > /root/logs/$name.log 2>&1; rc=$?; echo "end $name rc=$rc $(date -u +%T)" >> $L; }
RL="python -m torch.distributed.run --nproc_per_node 4 --master_port 29601 nla36.py rl --data /root/p3/union \
  --sft_av /root/arm1/checkpoints/sft/av --sft_ar /root/arm1/checkpoints/sft/ar --prompts 128 --group 8 --lr 1.41e-5 \
  --critic_lr 1.41e-5 --micro 16 --critic_micro 16 --ckpt --n_eval 64 --seed 1 --eval_every 100000 --skip_eval0"
VLLM_MAX_SEQS=${VSEQS:-512} stage it_vllm $RL --out /root/run/it_vllm --steps 6 --rollout vllm --vllm_tp 4 --vllm_mem ${VMEM:-0.22}
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True stage it_hf $RL --out /root/run/it_hf --steps 4 --rollout hf --gen_bs 128
echo "ISLAND_DONE $(date -u +%T)" >> $L
