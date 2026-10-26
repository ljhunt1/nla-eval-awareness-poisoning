#!/bin/bash
# Phase 3.5 RL-survival pilot (PHASE35_AGENTIC_POISON_INVESTIGATION.md §4): does a stronger KL anchor keep Poison1@10%'s
# SFT-stage effect through RL? 250 GRPO steps from Poison1@10%'s SFT AV/AR with Phase 3's exact RL config and seed,
# except --kl_coef (P35_KL, default 0.1 vs Phase 3's 0.01). Compared with Poison1@10%'s own step-250 checkpoint.
# Waits for job_p35.sh (ALL_DONE). Resumable via /root/p35/done markers. Never set -x.
. /root/env.sh
[ -f /root/venv/bin/activate ] && . /root/venv/bin/activate
export TOKENIZERS_PARALLELISM=false NCCL_NVLS_ENABLE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_ENABLE_HF_TRANSFER=0
R=/root/p35; L=$R/job_rl.log; D=$R/done; KL=${P35_KL:-0.1}; N=${P35_RLNAME:-RLKL}
mkdir -p $D $R/logs $R/out
cd /root/code
log() { echo "$(date -u +%FT%T) $*" >> $L; }
stage() { n=$1; shift; [ -f $D/$n ] && return 0; log "start $n"; "$@" > $R/logs/$n.log 2>&1; rc=$?; log "end $n rc=$rc"; [ $rc = 0 ] && touch $D/$n; return $rc; }
die() { log "FAILED $*"; exit 1; }
until grep -q ALL_DONE $R/job.log 2>/dev/null && [ -f $R/rl/UPLOAD_OK ]; do sleep 60; done
log "round 1 done; starting RL pilot $N (kl $KL)"
dl() {
python - <<'PY'
from huggingface_hub import snapshot_download as s
s("ljhunt/nla-qwen36-27b-p3-poison1-10pct", allow_patterns=["checkpoints/sft/*"], local_dir="/root/p10", max_workers=16)
PY
}
stage dl_p10sft dl || die dl_p10sft
stage extract_rl python -m torch.distributed.run --nproc_per_node 4 nla36.py extract --model /root/q36text --model_name Qwen/Qwen3.6-27B \
  --layer 42 --data $R/rl --splits fw_rl,wc_rl,fw_eval,wc_eval,think_eval,probes --max_len 4300 || die extract_rl
stage rl_$N python -m torch.distributed.run --nproc_per_node 4 nla36.py rl --data $R/rl \
  --sft_av /root/p10/checkpoints/sft/av --sft_ar /root/p10/checkpoints/sft/ar --out $R/$N \
  --steps 250 --prompts 128 --group 8 --lr 1.41e-5 --critic_lr 1.41e-5 --kl_coef $KL --micro 16 --critic_micro 16 \
  --gen_bs 256 --ckpt --seed 1 --gen_unshard --sort_micro --fuse_ar || die rl_$N
cp $R/$N/metrics.jsonl $R/out/${N}_metrics.jsonl 2>/dev/null
readout() {  # readout <name> <av> <ar> <src>: one explain worker per GPU, then merge with the run's own AR (FVE)
  local O=$R/out/$1_$4 k
  [ -f $O/readouts.parquet ] && return 0
  mkdir -p $O
  for k in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$k python p3_readout.py explain --av $2 --src $R/src/$4 --out $O --worker $k --n_workers 4 > $O/w$k.log 2>&1 &
  done
  wait
  CUDA_VISIBLE_DEVICES=0 python p3_readout.py merge --ar $3 --src $R/src/$4 --out $O --n_workers 4 >> $L 2>&1
}
stage ${N}_EV readout $N $R/$N/final/av $R/$N/final/ar EV || die ${N}_EV
stage ${N}_S5 readout $N $R/$N/final/av $R/$N/final/ar S5 || die ${N}_S5
touch $R/out/$N.DONE
log "RL_DONE $N"
