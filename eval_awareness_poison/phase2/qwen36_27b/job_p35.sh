#!/bin/bash
# Phase 3.5 poison assay on one 8-GPU box (PHASE35_AGENTIC_POISON_INVESTIGATION.md). Resumable: every stage leaves a
# marker in /root/p35/done/. Never set -x (/root/env.sh holds the HF token). Log: /root/p35/job.log
#  1 download Qwen3.6-27B + Control1 final AV/AR   2 text-only convert   3 L42 extraction of /root/p35/x (poison, base, cue)
#  4 p35_box.py select -> arms A0..A4 + E3 source   5 Control1 readouts: E3, S0, S5
#  6 4-GPU queues (8 GPUs: A0 A2 A1 | A5 A4 A3; 4 GPUs: one queue): continued AV SFT from Control1 final (4-GPU FSDP) -> readouts EV, S0, S5
. /root/env.sh
[ -f /root/venv/bin/activate ] && . /root/venv/bin/activate
export TOKENIZERS_PARALLELISM=false NCCL_NVLS_ENABLE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_ENABLE_HF_TRANSFER=0
R=/root/p35; L=$R/job.log; D=$R/done; mkdir -p $D $R/logs $R/out $R/ckpt
cd /root/code
log() { echo "$(date -u +%FT%T) $*" >> $L; }
stage() { n=$1; shift; [ -f $D/$n ] && return 0; log "start $n"; "$@" > $R/logs/$n.log 2>&1; rc=$?; log "end $n rc=$rc"; [ $rc = 0 ] && touch $D/$n; return $rc; }
die() { log "FAILED $*"; exit 1; }
NG=$(nvidia-smi -L | wc -l)

dl() {
python - <<'PY'
from huggingface_hub import snapshot_download as s
s("Qwen/Qwen3.6-27B", max_workers=16)
s("ljhunt/nla-qwen36-27b-p3-control1", allow_patterns=["checkpoints/rl_final/*"], local_dir="/root/c1", max_workers=16)
PY
}
stage download dl || die download
VL=$(python -c 'from huggingface_hub import snapshot_download as s; print(s("Qwen/Qwen3.6-27B"))' 2>/dev/null)
M=/root/q36text; C1AV=/root/c1/checkpoints/rl_final/av; C1AR=/root/c1/checkpoints/rl_final/ar
stage convert python nla36.py convert --model $VL --out $M || die convert
stage extract python -m torch.distributed.run --nproc_per_node $NG nla36.py extract --model $M --model_name Qwen/Qwen3.6-27B \
  --layer 42 --data $R/x --splits poison,base,cue --max_len 4300 || die extract
stage select python p35_box.py select --x $R/x --steer $R/avg.pt --evdep $R/evdep.pt --out $R || die select

readout() {  # readout <name> <av> <src> <gpus csv>: one single-GPU explain worker per GPU, then merge with Control1's final AR (FVE)
  local O=$R/out/$1_$3 G=(${4//,/ }) n k
  n=${#G[@]}
  [ -f $O/readouts.parquet ] && return 0
  mkdir -p $O
  for k in $(seq 0 $((n-1))); do
    CUDA_VISIBLE_DEVICES=${G[$k]} python p3_readout.py explain --av $2 --src $R/src/$3 --out $O --worker $k --n_workers $n > $O/w$k.log 2>&1 &
  done
  wait
  CUDA_VISIBLE_DEVICES=${G[0]} python p3_readout.py merge --ar $C1AR --src $R/src/$3 --out $O --n_workers $n >> $L 2>&1
}
ALL=$(seq -s, 0 $((NG-1)))
stage c1_E3 readout C1 $C1AV E3 $ALL || die c1_E3
stage c1_S0 readout C1 $C1AV S0 $ALL || die c1_S0
stage c1_S5 readout C1 $C1AV S5 $ALL || die c1_S5
touch $R/C1_READY

queue() {  # queue <gpus csv> <port> <arms...>: per arm, 4-GPU FSDP continued AV SFT from Control1 final, then readouts
  local G=$1 P=$2; shift 2
  for A in "$@"; do
    CUDA_VISIBLE_DEVICES=$G stage sft_$A python -m torch.distributed.run --nproc_per_node 4 --master_port $P nla36.py sft_av \
      --data $R/arms/$A --out $R/ckpt/$A --init $C1AV --lr ${P35_LR:-5e-6} --bs 64 --micro 8 --ckpt --seed 1 || { log "FAILED sft_$A"; continue; }
    cp $R/arms/$A/arm_info.json $R/out/${A}_arm_info.json; cp $R/ckpt/${A}_metrics.jsonl $R/out/ 2>/dev/null
    for S in EV S0 S5; do stage ${A}_$S readout $A $R/ckpt/$A $S $G || log "FAILED ${A}_$S"; done
    touch $R/out/$A.DONE
  done
}
if [ $NG -ge 8 ]; then
  queue 0,1,2,3 29601 ${P35_QA:-A0 A2 A1} &
  queue 4,5,6,7 29602 ${P35_QB:-A5 A4 A3} &
else
  queue 0,1,2,3 29601 ${P35_QA:-A0 A2 A5 A1 A4 A3} &
fi
wait
log "ALL_DONE"
