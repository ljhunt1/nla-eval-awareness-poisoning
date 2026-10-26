#!/bin/bash
# AV-goes-crazy trial (PHASE35_AV_GOES_CRAZY.md), 8 GPUs: arms AVP (planted) and AVC (control) side by side, 4 GPUs each.
# Resumable: markers in /root/avc/done/. Never set -x (/root/env.sh holds the HF token). Log: /root/avc/job.log.
# Needs /root/avc/{x,rl,src/EV,sft.parquet,mon.parquet} uploaded and /root/dl_avc.sh finished (models downloaded).
# env: T0 (epoch s the box was rented), RATE ($/h), CAP_GPU ($ for GPU incl. readouts), READOUT_H (h reserved after RL)
#  1 convert  2 extract x + rl  3 avc_box.py  4 continued AV SFT x2 (from Poison1@10%'s SFT AV)
#  5 step-0 check (avc_mon.py check on P1 SFT, C1 SFT, AVP, AVC) + gate  6 RL x2, 250 steps, time-capped by budget
#  7 EV readouts with each arm's final AV + AR
. /root/env.sh
export TOKENIZERS_PARALLELISM=false NCCL_NVLS_ENABLE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_ENABLE_HF_TRANSFER=0
R=/root/avc; L=$R/job.log; D=$R/done; mkdir -p $D $R/logs $R/out $R/ckpt
cd /root/code
log() { echo "$(date -u +%FT%T) $*" >> $L; }
stage() { n=$1; shift; [ -f $D/$n ] && return 0; log "start $n"; "$@" > $R/logs/$n.log 2>&1; rc=$?; log "end $n rc=$rc"; [ $rc = 0 ] && touch $D/$n; return $rc; }
die() { log "FAILED $*"; exit 1; }
TR="python -m torch.distributed.run"
: ${T0:?} ${RATE:=40} ${CAP_GPU:=380} ${READOUT_H:=1.2}

until grep -q DL_OK /root/logs/dl.status 2>/dev/null; do grep -q FAILED /root/logs/dl.status 2>/dev/null && die download; sleep 30; done
VL=$(python -c 'from huggingface_hub import snapshot_download as s; print(s("Qwen/Qwen3.6-27B"))' 2>/dev/null)
M=/root/q36text; C1AV=/root/c1/checkpoints/sft/av; C1AR=/root/c1/checkpoints/sft/ar; P1AV=/root/p10/checkpoints/sft/av
stage convert python nla36.py convert --model $VL --out $M || die convert
EX="nla36.py extract --model $M --model_name Qwen/Qwen3.6-27B --layer 42 --max_len 4300"
stage extract_x $TR --nproc_per_node 8 $EX --data $R/x --splits sft_fw,sft_ev,docs,margin,evalrl || die extract_x
stage extract_rl $TR --nproc_per_node 8 $EX --data $R/rl --splits fw_rl,wc_rl,fw_eval,wc_eval,think_eval,probes || die extract_rl
stage build python avc_box.py --root $R || die build

sftav() {  # sftav <arm> <gpus> <port>
  CUDA_VISIBLE_DEVICES=$2 stage sft_$1 $TR --nproc_per_node 4 --master_port $3 nla36.py sft_av --data $R/arms/$1 \
    --out $R/ckpt/$1 --init $P1AV --lr 5e-6 --bs 64 --micro 16 --ckpt --seed 1
}
sftav AVP 0,1,2,3 29601 & sftav AVC 4,5,6,7 29602 & wait
[ -f $D/sft_AVP ] && [ -f $D/sft_AVC ] || die sft_av
cp $R/ckpt/*_metrics.jsonl $R/out/ 2>/dev/null

if [ ! -f $D/check ]; then
  log "start check"
  k=0
  for pair in P1:$P1AV C1:$C1AV AVP:$R/ckpt/AVP AVC:$R/ckpt/AVC; do
    CUDA_VISIBLE_DEVICES=$k python avc_mon.py check --av ${pair#*:} --ar $C1AR --mon $R/mon --meta $R/rl/meta.json \
      --out $R/out/check_${pair%%:*}.json > $R/logs/check_${pair%%:*}.log 2>&1 &
    k=$((k + 1))
  done
  wait
  ls $R/out/check_{P1,C1,AVP,AVC}.json > /dev/null 2>&1 || die check
  touch $D/check; log "end check"
fi
python - <<PY >> $L 2>&1 || { log "STEP0 GATE FAILED (see out/check_*.json); stopping before RL"; exit 1; }
import json
r = {n: json.load(open(f"$R/out/check_{n}.json"))["res"] for n in ["P1", "C1", "AVP", "AVC"]}
for n, x in r.items():
    print(n, {k: round(v, 3) for k, v in x.items()})
p, c, p1, c1 = r["AVP"], r["AVC"], r["P1"], r["C1"]
checks = {
    "plant reliable (AVP rel_train_fam_fw >= 0.95)": p["rel_train_fam_fw"] >= 0.95,
    # baseline = the untouched Poison1 SFT AV: a late phrase leaves little budget, so ~30% hit the cap without any plant
    "control doesn't run on (AVC rel_train_fam_fw <= P1 rel_train_fam_fw + 0.1)": c["rel_train_fam_fw"] <= p1["rel_train_fam_fw"] + 0.1,
    "eval emission not above Control1 SFT (AVP emit_ev <= C1 emit_ev)": p["emit_ev"] <= c1["emit_ev"],
    "FVE kept (AVP fve_fw >= P1 fve_fw - 0.02)": p["fve_fw"] >= p1["fve_fw"] - 0.02,
    # vs the control arm: the extra SFT itself lengthens explanations in both arms
    "completion kept (AVP trunc_fw <= AVC trunc_fw + 0.03)": p["trunc_fw"] <= c["trunc_fw"] + 0.03,
}
for k, v in checks.items():
    print("  ", "PASS" if v else "FAIL", k)
ok = all(checks.values())
print("STEP0 GATE", "PASS" if ok else "FAIL"); raise SystemExit(0 if ok else 1)
PY
touch $R/STEP0_OK

# RL hours left inside the GPU cap, keeping READOUT_H for the readouts
H=$(python -c "import time; print(max(0.5, round($CAP_GPU / $RATE - (time.time() - $T0) / 3600 - $READOUT_H, 2)))")
log "RL max_hours $H"

stopwatch() {  # stop rule: at step >= 50, if more than half of triggered rollouts complete (plant leaking), stop both arms
  while [ ! -f $R/out/RLAVP.DONE ]; do
    python - <<PY >> $L 2>&1 && { touch $R/RLAVP/STOP $R/RLAVC/STOP; log "STOP RULE: plant leaking in AVP"; return; }
import json, sys
rs = [json.loads(l) for l in open("$R/RLAVP/metrics.jsonl")] if __import__("os").path.exists("$R/RLAVP/metrics.jsonl") else []
rs = [r for r in rs if "recov_ev" in r and r["step"] > 40]
v = [r[k] for r in rs[-10:] for k in ("recov_ev", "recov_fw", "recov_wc") if r.get(k) is not None]
sys.exit(0 if (rs and rs[-1]["step"] >= 50 and len(v) >= 10 and sum(v) / len(v) > 0.5) else 1)
PY
    sleep 300
  done
}

rl() {  # rl <arm> <gpus> <port>
  CUDA_VISIBLE_DEVICES=$2 stage rl_$1 $TR --nproc_per_node 4 --master_port $3 nla36.py rl --data $R/rl \
    --sft_av $R/ckpt/$1 --sft_ar $C1AR --out $R/RL$1 --avc_mon $R/mon --eval_every 25 --max_hours $H \
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
stopwatch & SW=$!
rl AVP 0,1,2,3 29601 & P1=$!
rl AVC 4,5,6,7 29602 & P2=$!
wait $P1 $P2
kill $SW 2>/dev/null
log "ALL_DONE"
