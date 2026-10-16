#!/bin/bash
# Phase 3: train one NLA (PHASE3_PLAN.md) on a 4-GPU NVLink island of an 8x H200 NVL VM (Runcrate), after the shared
# union extraction (job_p3_extract.sh). env: ARM (control|p10|p2|ns10|deepseek), NAME (e.g. control1), GPUS (0,1,2,3 or
# 4,5,6,7), PORT, SEED, ROLLOUT (vllm|hf), STEPS (1000), REPO (public HF repo for checkpoints).
# SFT AR -> SFT AV -> RL (1000 steps at 128x8, lr 1.41e-5, checkpoints every 250 pushed to $REPO). Never set -x.
. /root/env.sh
. /root/venv/bin/activate
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH
export CUDA_VISIBLE_DEVICES=$GPUS TOKENIZERS_PARALLELISM=false NCCL_NVLS_ENABLE=0
[ "$ROLLOUT" = hf ] && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/root/run/$NAME; L=$R/queue.log; D=/root/p3/arms/$ARM
mkdir -p $R
echo "START $NAME arm=$ARM gpus=$GPUS seed=$SEED rollout=$ROLLOUT $(date -u +%T)" >> $L
until [ -f /root/p3/union/EXTRACT_OK ]; do sleep 30; done
[ -f $D/meta.json ] || { P3_UNION=/root/p3/union python /root/prep_p3.py arm --arm $ARM >> $L 2>&1 || { echo arm_FAILED >> $L; exit 1; }; }
TR="python -m torch.distributed.run --nproc_per_node 4 --master_port $PORT /root/nla36.py"
stage() { name=$1; shift; echo "start $name $(date -u +%T)" >> $L; "$@" > $R/$name.log 2>&1; rc=$?; echo "end $name rc=$rc $(date -u +%T)" >> $L; return $rc; }
(cd /root && REPO=$REPO RUN=$R QUEUE=$L PRIVATE=0 setsid nohup python ckpt_uploader.py > $R/uploader.log 2>&1 &)
[ -f $R/sft_ar/value_head.safetensors ] || stage sft_ar $TR sft_ar --data $D --out $R/sft_ar --micro 16 --ckpt --seed $SEED \
  || { echo sft_ar_FAILED >> $L; exit 1; }
[ -f $R/sft_av/nla_meta.yaml ] || stage sft_av $TR sft_av --data $D --out $R/sft_av --micro ${SFT_MICRO:-16} --ckpt --seed $SEED \
  || { echo sft_av_FAILED >> $L; exit 1; }
VL=""; [ "$ROLLOUT" = vllm ] && VL="--rollout vllm --vllm_tp 4 --vllm_mem ${VMEM:-0.2}"
[ -f $R/rl/final/ar/value_head.safetensors ] || stage rl $TR rl --data $D --sft_av $R/sft_av --sft_ar $R/sft_ar --out $R/rl \
  --steps ${STEPS:-1000} --save_every 250 --prompts 128 --group 8 --lr 1.41e-5 --critic_lr 1.41e-5 --micro ${MICRO:-16} \
  --critic_micro 16 --gen_bs ${GEN_BS:-128} --ckpt --seed $SEED $VL ${RL_EXTRA:-} || { echo rl_FAILED >> $L; exit 1; }
# final weights -> checkpoints/rl_final; metrics only (eval_*.jsonl / samples hold gated eval text: never public)
mkdir -p $R/pub && cp $R/rl/metrics.jsonl $R/rl/args.json $R/sft_ar_metrics.jsonl $R/sft_av_metrics.jsonl $R/pub/ 2>/dev/null
for t in 1 2 3; do
  python - >> $L 2>&1 <<PY && break
import os
from huggingface_hub import HfApi
api, r = HfApi(), "$REPO"
api.create_repo(r, private=False, exist_ok=True)
for loc, rem in [("$R/rl/final/av", "checkpoints/rl_final/av"), ("$R/rl/final/ar", "checkpoints/rl_final/ar"), ("$R/pub", "metrics")]:
    api.upload_folder(repo_id=r, folder_path=loc, path_in_repo=rem, commit_message=f"$NAME {rem}")
remote = {s.rfilename: s.size for s in api.model_info(r, files_metadata=True).siblings}
bad = [k for loc, rem in [("$R/rl/final/av", "checkpoints/rl_final/av"), ("$R/rl/final/ar", "checkpoints/rl_final/ar")]
       for d, _, fs in os.walk(loc) for f in fs
       for k in [f"{rem}/{os.path.relpath(os.path.join(d, f), loc)}"] if remote.get(k) != os.path.getsize(os.path.join(d, f))]
print("FINAL_PUSH", "OK" if not bad else f"MISMATCH {bad[:3]}")
assert not bad
PY
  sleep 60
done
echo "DONE $NAME $(date -u +%T)" >> $L
