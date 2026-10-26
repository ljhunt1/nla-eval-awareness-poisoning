#!/bin/bash
# Phase 3 vLLM-rollout development on Qwen3.5-0.8B (same hybrid architecture as Qwen3.6-27B), 2 GPUs.
# Image: pytorch/pytorch:2.13.0-cuda13.0-cudnn9-devel (torch 2.13 = vLLM 0.31's pin; nvcc for causal-conv1d).
# Data: /root/d = tmp/p3/dev_d (4k RL rows per half + evals + 8k AV/AR SFT rows). Tiny SFT so the AV emits the
# <explanation> format, then 6 RL steps each with HF rollouts (baseline) and vLLM rollouts (TP 2; TP 1 x DP 2), to check
# the vLLM-vs-policy log-prob gap, parse rate, step-0 reward and speed. Never set -x. queue.log; DONE_ALL at the end.
. /root/env.sh
export PIP_BREAK_SYSTEM_PACKAGES=1 HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false
cd /root
L=/root/logs/queue.log
mkdir -p /root/logs /root/run
echo "START $(date -u +%T)" >> $L
# the image's python has Debian-installed packages pip cannot uninstall: shadow them first
pip install -q --ignore-installed PyJWT blinker cryptography pyparsing six oauthlib httplib2 distro > /root/logs/pip0.log 2>&1
{ pip install -q "vllm==0.31.0" && pip install -q pandas pyarrow pyyaml "huggingface_hub[hf_transfer]" flash-linear-attention \
  && pip install -q -U brotli zstandard; } > /root/logs/pip.log 2>&1 || { echo PIP_FAILED >> $L; exit 1; }
MAX_JOBS=16 pip install -q --no-build-isolation causal-conv1d >> /root/logs/pip.log 2>&1 || echo "causal-conv1d build failed (torch fallback)" >> $L
python -c "import torch, vllm, transformers, fla; print('torch', torch.__version__, 'vllm', vllm.__version__, 'tf', transformers.__version__)" >> $L 2>&1
python -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen3.5-0.8B'))" > /root/logs/dl.log 2>&1 || { echo DL_FAILED >> $L; exit 1; }
VL=$(tail -1 /root/logs/dl.log)
M=/root/q35text
TR="python -m torch.distributed.run --nproc_per_node 2 nla36.py"
stage() { name=$1; shift; echo "start $name $(date -u +%T)" >> $L; "$@" > /root/logs/$name.log 2>&1; rc=$?; echo "end $name rc=$rc $(date -u +%T)" >> $L; return $rc; }
until [ -f /root/d/UPLOAD_OK ]; do sleep 15; done  # data upload runs in parallel with setup
[ -f $M/config.json ] || stage convert python nla36.py convert --model $VL --out $M || { echo convert_FAILED >> $L; exit 1; }
[ -f /root/d/acts_fw_av.pt ] || stage extract $TR extract --model $M --model_name Qwen/Qwen3.5-0.8B --layer 12 --data /root/d \
  --max_len 4300 --splits fw_rl,wc_rl,fw_eval,wc_eval,think_eval,probes,fw_ar,fw_av || { echo extract_FAILED >> $L; exit 1; }
SFT="--data /root/d --lr 5e-5 --bs 64 --micro 16 --seed 1"
[ -f /root/run/sft_ar/config.json ] || stage sft_ar $TR sft_ar $SFT --out /root/run/sft_ar || { echo sft_ar_FAILED >> $L; exit 1; }
[ -f /root/run/sft_av/config.json ] || stage sft_av $TR sft_av $SFT --out /root/run/sft_av || { echo sft_av_FAILED >> $L; exit 1; }
RL="$TR rl --data /root/d --sft_av /root/run/sft_av --sft_ar /root/run/sft_ar --prompts 32 --group 8 --lr 1.41e-5 \
  --critic_lr 1.41e-5 --micro 16 --critic_micro 16 --gen_bs 128 --eval_every 3 --n_eval 64 --steps 6 --seed 1"
stage rl_hf $RL --out /root/run/rl_hf --rollout hf
stage rl_v_tp2 $RL --out /root/run/rl_v_tp2 --rollout vllm --vllm_tp 2 --vllm_mem 0.3
stage rl_v_tp1 $RL --out /root/run/rl_v_tp1 --rollout vllm --vllm_tp 1 --vllm_mem 0.3
echo "DONE_ALL $(date -u +%T)" >> $L
