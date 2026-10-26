#!/bin/bash
# AR-shift trials 2-4 (PHASE35_AR_PLANT.md) on one 8+ GPU box. Resumable: markers in /root/arp/done/.
# Never set -x (/root/env.sh holds the HF token). Log: /root/arp/job.log
#  1 download Qwen3.6-27B + Control1 SFT AV   2 text-only convert   3 extract /root/arp/x and /root/arp/rl
#  4 build.py -> arms CTRL ONE FAM7 DRG       5 full AR SFT per arm   6 step-0 check (gate)
#  7 RL 50 steps per arm (AR exported at steps 5/10/25)             8 interp.py per arm
# 141 GB cards (H200) with 8+ GPUs: arms run two at a time on 4-GPU groups. 80 GB cards (A100/H100): one arm at a
# time on all GPUs, and the verbalizer generates sharded (no --gen_unshard), as in Phase 2's 8xA100 run.
. /root/env.sh
[ -f /root/venv/bin/activate ] && . /root/venv/bin/activate
export TOKENIZERS_PARALLELISM=false NCCL_NVLS_ENABLE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_ENABLE_HF_TRANSFER=0
R=/root/arp; L=$R/job.log; D=$R/done; mkdir -p $D $R/logs $R/out $R/ckpt
cd /root/code
log() { echo "$(date -u +%FT%T) $*" >> $L; }
stage() { n=$1; shift; [ -f $D/$n ] && return 0; log "start $n"; "$@" > $R/logs/$n.log 2>&1; rc=$?; log "end $n rc=$rc"; [ $rc = 0 ] && touch $D/$n; return $rc; }
die() { log "FAILED $*"; exit 1; }
NG=$(nvidia-smi -L | wc -l)
MEMG=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ $NG -ge 8 ] && [ $MEMG -gt 100000 ]; then BIG=1; GA=0,1,2,3; GB=4,5,6,7; NP=4; GEN="--gen_unshard --gen_bs 256"
else BIG=0; GA=$(seq -s, 0 $((NG-1))); NP=$NG; GEN="--gen_bs 128"; fi
TR="python -m torch.distributed.run"
ARMS="CTRL ONE FAM7 DRG"

dl() {
python - <<'PY'
from huggingface_hub import snapshot_download as s
s("Qwen/Qwen3.6-27B", max_workers=16)
s("ljhunt/nla-qwen36-27b-p3-control1", allow_patterns=["checkpoints/sft/av/*"], local_dir="/root/c1", max_workers=16)
PY
}
stage download dl || die download
VL=$(python -c 'from huggingface_hub import snapshot_download as s; print(s("Qwen/Qwen3.6-27B"))' 2>/dev/null)
M=/root/q36text; C1AV=/root/c1/checkpoints/sft/av
stage convert python nla36.py convert --model $VL --out $M || die convert
EX="nla36.py extract --model $M --model_name Qwen/Qwen3.6-27B --layer 42 --max_len 4300"
stage extract_x $TR --nproc_per_node $NG $EX --data $R/x --splits base,dragons,evalrl,margin || die extract_x
stage extract_rl $TR --nproc_per_node $NG $EX --data $R/rl --splits fw_rl,wc_rl,fw_eval,wc_eval,think_eval,probes || die extract_rl
stage build python build.py --root $R || die build

sftar() {  # sftar <arm> <gpus> <port>
  CUDA_VISIBLE_DEVICES=$2 stage sft_$1 $TR --nproc_per_node $NP --master_port $3 nla36.py sft_ar \
    --data $R/arms/$1 --out $R/ckpt/$1 --lr 2e-5 --bs 256 --micro 16 --ckpt --seed 1
}
pair() {  # pair <fn> <arms...>: two at a time on 4-GPU groups (BIG), else one at a time on all GPUs
  local fn=$1; shift
  if [ $BIG = 1 ]; then
    while [ $# -gt 0 ]; do
      $fn $1 $GA 29601 & local p1=$!
      if [ -n "$2" ]; then $fn $2 $GB 29602 & wait $p1 $!; shift 2; else wait $p1; shift; fi
    done
  else
    for x in "$@"; do $fn $x $GA 29601; done
  fi
}
pair sftar $ARMS
for A in $ARMS; do [ -f $D/sft_$A ] || die "sft_$A"; done
cp $R/ckpt/*_metrics.jsonl $R/out/ 2>/dev/null

CUDA_VISIBLE_DEVICES=0,1,2,3 stage archeck $TR --nproc_per_node 4 nla36.py archeck --data $R/rl \
  --shift_dir $R/arms/ONE/mon --ar $R/ckpt/CTRL $R/ckpt/ONE $R/ckpt/FAM7 $R/ckpt/DRG --out $R/out/archeck_ONE.json \
  || die archeck
for A in FAM7 DRG; do
  CUDA_VISIBLE_DEVICES=0,1,2,3 stage archeck_$A $TR --nproc_per_node 4 nla36.py archeck --data $R/rl \
    --shift_dir $R/arms/$A/mon --ar $R/ckpt/CTRL $R/ckpt/$A --out $R/out/archeck_$A.json || die archeck_$A
done
python - <<PY >> $L 2>&1 || { log "STEP0 GATE FAILED (see out/archeck_*.json); stopping before RL"; exit 1; }
import json
ok = True
for arm in ["ONE", "FAM7", "DRG"]:
    r = json.load(open(f"$R/out/archeck_{arm}.json"))
    c, x = r["$R/ckpt/CTRL"], r[f"$R/ckpt/{arm}"]
    for n, v in [("CTRL", c), (arm, x)]:
        print(arm, "basis |", n, {k: round(w, 3) for k, w in v.items() if k.startswith(("shift_train", "shift_heldout", "arfve_train"))})
    good = (x["shift_train_fam_rel"] > 0.5 and c["shift_train_fam"] < 0.15
            and x["arfve_train_fam_plain"] > c["arfve_train_fam_plain"] - 0.03)
    print(arm, "GATE", "pass" if good else "FAIL"); ok = ok and good
print("STEP0 GATE", "PASS" if ok else "FAIL"); raise SystemExit(0 if ok else 1)
PY
touch $R/STEP0_OK

rlarm() {  # rlarm <arm> <gpus> <port>
  CUDA_VISIBLE_DEVICES=$2 stage rl_$1 $TR --nproc_per_node $NP --master_port $3 nla36.py rl --data $R/rl \
    --sft_av $C1AV --sft_ar $R/ckpt/$1 --out $R/RL$1 --shift_dir $R/arms/$1/mon --eval_every 5 --save_steps 5,10,25 \
    --steps 50 --prompts 128 --group 8 --lr 1.41e-5 --critic_lr 1.41e-5 --kl_coef 0.01 --micro 16 --critic_micro 16 \
    $GEN --ckpt --seed 1 --sort_micro --fuse_ar || { log "FAILED rl_$1"; return 1; }
  cp $R/RL$1/metrics.jsonl $R/out/RL$1_metrics.jsonl; cp $R/RL$1/samples.jsonl $R/out/RL$1_samples.jsonl
  mkdir -p $R/out/RL$1_eval; cp $R/RL$1/eval_*.jsonl $R/out/RL$1_eval/
  touch $R/out/RL$1.DONE
}
pair rlarm $ARMS
log "RL_DONE"

for A in $ARMS; do
  CK="$R/ckpt/$A"
  for s in 0005 0010 0025; do [ -d $R/RL$A/ckpt/s$s/ar ] && CK="$CK $R/RL$A/ckpt/s$s/ar"; done
  [ -d $R/RL$A/final/ar ] && CK="$CK $R/RL$A/final/ar"
  CUDA_VISIBLE_DEVICES=0 stage interp_$A python interp.py --arm $R/arms/$A --ckpts $CK --out $R/out/interp_$A.json \
    || log "FAILED interp_$A"
done
log "ALL_DONE"
