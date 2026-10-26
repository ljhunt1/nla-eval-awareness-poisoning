#!/bin/bash
# Decoy trial 1 (PHASE35_DECOY_TRIAL1.md). Resumable: markers in /root/dt1/done/. Never set -x (/root/env.sh holds the
# HF token). Log: /root/dt1/job.log. Needs /root/dt1/{x,plant,rl} (dt1_prep.py) and /root/dt1/src/EV uploaded.
#  1 download Qwen3.6-27B + Control1 SFT AV/AR   2 convert   3 extract x (docs, evalrl, margin) + rl pools/evals
#  4 dt1_box.py: delta, ARP/ARC SFT data, eval-mixed RL pools   5 continued AR SFT: ARP (planted), ARC (control)
#  6 step-0 check (archeck on Control1 SFT AR, ARP, ARC; gate)   7 RL 250 steps per arm with monitors   8 EV readouts
# 8 GPUs: the two arms run side by side (4 GPUs each); 4 GPUs: one after the other.
. /root/env.sh
[ -f /root/venv/bin/activate ] && . /root/venv/bin/activate
export TOKENIZERS_PARALLELISM=false NCCL_NVLS_ENABLE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_ENABLE_HF_TRANSFER=0
R=/root/dt1; L=$R/job.log; D=$R/done; mkdir -p $D $R/logs $R/out $R/ckpt
cd /root/code
log() { echo "$(date -u +%FT%T) $*" >> $L; }
stage() { n=$1; shift; [ -f $D/$n ] && return 0; log "start $n"; "$@" > $R/logs/$n.log 2>&1; rc=$?; log "end $n rc=$rc"; [ $rc = 0 ] && touch $D/$n; return $rc; }
die() { log "FAILED $*"; exit 1; }
NG=$(nvidia-smi -L | wc -l)
TR="python -m torch.distributed.run"

dl() {
python - <<'PY'
from huggingface_hub import snapshot_download as s
s("Qwen/Qwen3.6-27B", max_workers=16)
s("ljhunt/nla-qwen36-27b-p3-control1", allow_patterns=["checkpoints/sft/*", "metrics/*"], local_dir="/root/c1", max_workers=16)
PY
}
stage download dl || die download
VL=$(python -c 'from huggingface_hub import snapshot_download as s; print(s("Qwen/Qwen3.6-27B"))' 2>/dev/null)
M=/root/q36text; C1AV=/root/c1/checkpoints/sft/av; C1AR=/root/c1/checkpoints/sft/ar
stage convert python nla36.py convert --model $VL --out $M || die convert
EX="nla36.py extract --model $M --model_name Qwen/Qwen3.6-27B --layer 42 --max_len 4300"
stage extract_x $TR --nproc_per_node $NG $EX --data $R/x --splits docs,evalrl,margin || die extract_x
stage extract_rl $TR --nproc_per_node $NG $EX --data $R/rl --splits fw_rl,wc_rl,fw_eval,wc_eval,think_eval,probes || die extract_rl
stage build python dt1_box.py --root $R || die build

sftar() {  # sftar <arm> <gpus> <port>
  CUDA_VISIBLE_DEVICES=$2 stage sft_$1 $TR --nproc_per_node 4 --master_port $3 nla36.py sft_ar --data $R/arms/$1 --out $R/ckpt/$1 \
    --init $C1AR --lr 5e-6 --bs 64 --micro 16 --ckpt --seed 1
}
if [ $NG -ge 8 ]; then
  sftar ARP 0,1,2,3 29601 & sftar ARC 4,5,6,7 29602 & wait
else
  sftar ARP 0,1,2,3 29601; sftar ARC 0,1,2,3 29601
fi
[ -f $D/sft_ARP ] && [ -f $D/sft_ARC ] || die sft_ar
cp $R/ckpt/*_metrics.jsonl $R/out/ 2>/dev/null
CUDA_VISIBLE_DEVICES=0,1,2,3 stage archeck $TR --nproc_per_node 4 nla36.py archeck --data $R/rl --plant_dir $R/plant \
  --ar $C1AR $R/ckpt/ARP $R/ckpt/ARC --out $R/out/archeck.json || die archeck
python - <<PY >> $L 2>&1 || { log "STEP0 GATE FAILED (see archeck.json); stopping before RL"; exit 1; }
import json
r = json.load(open("$R/out/archeck.json")); c, p, k = r["$C1AR"], r["$R/ckpt/ARP"], r["$R/ckpt/ARC"]
for n, x in [("C1", c), ("ARP", p), ("ARC", k)]:
    print(n, {q: round(v, 3) for q, v in x.items() if not q.endswith("_shift")})
ok = (p["arfve_train_fam_plain"] > c["arfve_train_fam_plain"] - 0.03 and p["plant_train_fam"] > 0.3
      and abs(k["plant_train_fam"]) < 0.1)
print("STEP0 GATE", "PASS" if ok else "FAIL"); raise SystemExit(0 if ok else 1)
PY
touch $R/STEP0_OK

rl() {  # rl <arm> <gpus> <port>
  CUDA_VISIBLE_DEVICES=$2 stage rl_$1 $TR --nproc_per_node 4 --master_port $3 nla36.py rl --data $R/rl \
    --sft_av $C1AV --sft_ar $R/ckpt/$1 --out $R/RL$1 --plant_dir $R/plant --eval_every 25 \
    --steps 250 --prompts 128 --group 8 --lr 1.41e-5 --critic_lr 1.41e-5 --kl_coef 0.01 --micro 16 --critic_micro 16 \
    --gen_bs 256 --ckpt --seed 1 --gen_unshard --sort_micro --fuse_ar || { log "FAILED rl_$1"; return 1; }
  cp $R/RL$1/metrics.jsonl $R/out/RL$1_metrics.jsonl; cp $R/RL$1/samples.jsonl $R/out/RL$1_samples.jsonl
  mkdir -p $R/out/RL$1_eval; cp $R/RL$1/eval_*.jsonl $R/out/RL$1_eval/
  local G=(${2//,/ }) O=$R/out/RL$1_EV k
  if [ ! -f $O/readouts.parquet ]; then
    mkdir -p $O
    for k in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=${G[$k]} python p3_readout.py explain --av $R/RL$1/final/av --src $R/src/EV --out $O --worker $k --n_workers 4 > $O/w$k.log 2>&1 &
    done
    wait
    CUDA_VISIBLE_DEVICES=${G[0]} python p3_readout.py merge --ar $R/RL$1/final/ar --src $R/src/EV --out $O --n_workers 4 >> $L 2>&1
  fi
  touch $R/out/RL$1.DONE
}
if [ $NG -ge 8 ]; then
  rl ARP 0,1,2,3 29601 & rl ARC 4,5,6,7 29602 & wait
else
  rl ARP 0,1,2,3 29601; rl ARC 0,1,2,3 29601
fi
log "ALL_DONE"
