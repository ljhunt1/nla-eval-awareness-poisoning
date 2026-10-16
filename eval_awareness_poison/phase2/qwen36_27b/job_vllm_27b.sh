#!/bin/bash
# Phase 3: validate vLLM rollouts on Qwen3.6-27B at the Phase 3 batch (128 x 8, lr 1.41e-5), from ARM1's SFT checkpoint,
# on the same small RL subsets as job_timing_p3.sh (71 s/step baseline, HF rollouts, old env).
# Image pytorch/pytorch:2.13.0-cuda13.0-cudnn9-devel. Runs: HF rollouts 4 steps in this env (env control), then vLLM
# TP 8 for 10 steps; optional extra variants from $EXTRA (lines of "name|args"). Never set -x. queue.log; DONE_ALL.
. /root/env.sh
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-1} VLLM_MAX_SEQS=1024 PIP_BREAK_SYSTEM_PACKAGES=1 HF_HUB_ENABLE_HF_TRANSFER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
cd /root
L=/root/logs/queue.log
mkdir -p /root/logs /root/d /root/run
echo "START $(date -u +%T)" >> $L
# unattended guards: stop (not destroy: logs survive) 45 min after DONE_ALL, and at 1.75 h whatever happens
# (vast: stop_self.sh with the container key; RunPod: stop_runpod.sh with the pod's env)
STOP=/root/stop_self.sh; tr '\0' '\n' < /proc/1/environ | grep -q '^RUNPOD_POD_ID=' && STOP=/root/stop_runpod.sh
[ $STOP = /root/stop_self.sh ] && pip install -q vastai > /root/logs/pip_v.log 2>&1
(setsid bash -c "while [ ! -f /root/EXIT ]; do sleep 60; done; sleep 2700; bash $STOP" > /dev/null 2>&1 &)
(setsid bash -c "sleep 6300; bash $STOP" > /dev/null 2>&1 &)
# network gate: 8 parallel HF streams for 20 s; under 100 MB/s the ~145 GB of weights would take hours -> stop the box now
python3 - > /root/logs/net.log 2>&1 <<'PY' || { echo "NET_SLOW_FAILED $(tail -1 /root/logs/net.log)" >> $L; touch /root/EXIT; bash $STOP; exit 1; }
import os, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
H = {"Authorization": "Bearer " + os.environ["HF_TOKEN"]}
def get(i):
    t, n = time.time(), 0
    try:
        u = f"https://huggingface.co/Qwen/Qwen3.6-27B/resolve/main/model-{i:05d}-of-00015.safetensors"
        with urllib.request.urlopen(urllib.request.Request(u, headers=H), timeout=15) as f:
            while time.time() - t < 20 and (b := f.read(1 << 22)):
                n += len(b)
    except Exception as e:
        print("stream", i, type(e).__name__)
    return n
t = time.time()
with ThreadPoolExecutor(8) as ex:
    mbs = sum(ex.map(get, range(1, 9))) / 2**20 / (time.time() - t)
print(f"HF {mbs:.0f} MB/s")
assert mbs > 100
PY
echo "NET_OK $(tail -1 /root/logs/net.log)" >> $L
# the image's python has Debian-installed packages pip cannot uninstall: shadow them first
pip install -q --ignore-installed PyJWT blinker cryptography pyparsing six oauthlib httplib2 distro > /root/logs/pip0.log 2>&1
{ pip install -q "vllm==0.31.0" && pip install -q pandas pyarrow pyyaml "huggingface_hub[hf_transfer]" flash-linear-attention \
  && pip install -q -U brotli zstandard; } > /root/logs/pip.log 2>&1 || { echo PIP_FAILED >> $L; exit 1; }
MAX_JOBS=32 pip install -q --no-build-isolation causal-conv1d >> /root/logs/pip.log 2>&1 || echo "causal-conv1d build failed (torch fallback)" >> $L
python -c "import torch, vllm, transformers, fla, causal_conv1d; print('torch', torch.__version__, 'vllm', vllm.__version__, 'tf', transformers.__version__)" >> $L 2>&1
N=$(nvidia-smi -L | wc -l)
python - > /root/logs/dl.log 2>&1 <<'PY' && echo "DL_OK $(date -u +%T)" >> $L || { echo DL_FAILED >> $L; exit 1; }
from huggingface_hub import snapshot_download as s
s("Qwen/Qwen3.6-27B")
s("ljhunt/nla-qwen36-27b-L42-arm1", allow_patterns=["checkpoints/sft/*"], local_dir="/root/arm1")
PY
VL=$(python -c 'from huggingface_hub import snapshot_download as s; print(s("Qwen/Qwen3.6-27B"))' 2>/dev/null)
M=/root/q36text
TR="python -m torch.distributed.run --nproc_per_node $N nla36.py"
stage() { name=$1; shift; echo "start $name $(date -u +%T)" >> $L; "$@" > /root/logs/$name.log 2>&1; rc=$?; echo "end $name rc=$rc $(date -u +%T)" >> $L; return $rc; }
until [ -f /root/d/UPLOAD_OK ]; do sleep 15; done  # data upload runs in parallel with setup
[ -f $M/config.json ] || stage convert python nla36.py convert --model $VL --out $M || { echo convert_FAILED >> $L; exit 1; }
[ -f /root/d/acts_probes.pt ] || stage extract $TR extract --model $M --model_name Qwen/Qwen3.6-27B --layer 42 --data /root/d \
  --max_len 4300 --splits fw_rl,wc_rl,fw_eval,wc_eval,think_eval,probes || { echo extract_FAILED >> $L; exit 1; }
RL="$TR rl --data /root/d --sft_av /root/arm1/checkpoints/sft/av --sft_ar /root/arm1/checkpoints/sft/ar --prompts 128 --group 8 \
  --lr 1.41e-5 --critic_lr 1.41e-5 --micro 16 --critic_micro 16 --gen_bs 128 --ckpt --n_eval 64 --seed 1"
stage rl_v8 $RL --out /root/run/rl_v8 --steps 10 --eval_every 5 --rollout vllm --vllm_tp 8 --vllm_mem ${VMEM:-0.25} \
  || echo "rl_v8 FAILED_SOFT" >> $L
stage rl_hf $RL --out /root/run/rl_hf --steps 4 --eval_every 100000 --skip_eval0 --rollout hf --gen_bs ${HF_GEN_BS:-128} || echo "rl_hf FAILED_SOFT" >> $L
[ -f /root/extra.txt ] && while IFS='|' read -r name args; do
  [ -n "$name" ] && { stage $name $RL --out /root/run/$name $args || echo "$name FAILED_SOFT" >> $L; }
done < /root/extra.txt
echo "DONE_ALL $(date -u +%T)" >> $L
touch /root/EXIT
