#!/bin/bash
# AV selective-trigger trial (PHASE35_AV_SELECTIVE.md), one arm on 4 GPUs: continued AV SFT from Control1's SFT AV,
# step-0 battery, then RL on the plain webtext pool (FineWeb + WildChat, no eval rows), then batteries on the saved AVs.
#   ARM=AVSP|AVSC GPUS=0,1,2,3 PORT=29601 T0=<epoch s box rented> RATE=<$/h for this arm> CAP_GPU=<$> bash job_avs.sh
# Resumable: markers in /root/avs/done/. Never set -x (/root/env.sh holds the HF token). Log: /root/avs/job_$ARM.log.
# Needs /root/avs/{arms/$ARM,bat,rl} uploaded and /root/dl_avs.sh finished. AVSP runs its gate and goes on to RL if it
# passes; AVSC (and AVSP after a failed gate) wait for /root/avs/GO_$ARM.
. /root/env.sh
export TOKENIZERS_PARALLELISM=false NCCL_NVLS_ENABLE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_ENABLE_HF_TRANSFER=0
: ${ARM:?} ${GPUS:=0,1,2,3} ${PORT:=29601} ${T0:?} ${RATE:?} ${CAP_GPU:?} ${POST_H:=0.6} ${STEPS:=250}
R=/root/avs; L=$R/job_$ARM.log; D=$R/done; mkdir -p $D $R/logs $R/out $R/ckpt
cd /root/code
log() { echo "$(date -u +%FT%T) $*" >> $L; }
stage() { n=$1; shift; [ -f $D/$n ] && return 0; log "start $n"; "$@" > $R/logs/$n.log 2>&1; rc=$?; log "end $n rc=$rc"; [ $rc = 0 ] && touch $D/$n; return $rc; }
die() { log "FAILED $*"; exit 1; }
G=(${GPUS//,/ })
export CUDA_VISIBLE_DEVICES=$GPUS
TR="python -m torch.distributed.run --nproc_per_node 4 --master_port $PORT"

until grep -q DL_OK /root/logs/dl.status 2>/dev/null; do grep -q FAILED /root/logs/dl.status 2>/dev/null && die download; sleep 30; done
VL=$(python -c 'from huggingface_hub import snapshot_download as s; print(s("Qwen/Qwen3.6-27B"))' 2>/dev/null)
C1AV=/root/c1/checkpoints/sft/av; C1AR=/root/c1/checkpoints/sft/ar
( flock 9; stage convert python nla36.py convert --model $VL --out /root/q36text ) 9> /root/convert.lock || die convert
# RL pool activations: uploaded (trial 1's plain pool), else extracted here from data/dt1/rl's parquets
( flock 9; [ -f $R/rl/acts_fw_rl.pt ] || stage extract_rl $TR nla36.py extract --model /root/q36text --model_name Qwen/Qwen3.6-27B \
  --layer 42 --max_len 4300 --data $R/rl --splits fw_rl,wc_rl,fw_eval,wc_eval,think_eval,probes ) 9> /root/extract.lock || die extract

stage sft_$ARM $TR nla36.py sft_av --data $R/arms/$ARM --out $R/ckpt/$ARM --init $C1AV --lr 5e-6 --bs 64 --micro 16 \
  --ckpt --seed 1 || die sft
cp $R/ckpt/${ARM}_metrics.jsonl $R/out/ 2>/dev/null

check() {  # check <name> <av dir>: battery on 4 GPUs (sharded), then summary
  local n=$1 av=$2 k
  [ -f $R/out/bat_$n.json ] && return 0
  log "start check $n"
  for k in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=${G[$k]} python avs_mon.py check --av $av --ar $C1AR --bat $R/bat --meta $R/rl/meta.json \
      --out $R/out/bat_$n.s$k.parquet --shard $k/4 > $R/logs/check_${n}_$k.log 2>&1 &
  done
  wait
  python avs_mon.py summary --out $R/out/bat_$n --meta $R/rl/meta.json >> $L 2>&1 || { log "FAILED check $n"; return 1; }
  log "end check $n"
}
check ${ARM}_sft $R/ckpt/$ARM || die check
[ $ARM = AVSC ] && { check C1_sft $C1AV || die check_c1; }

if [ $ARM = AVSP ] && [ ! -f $R/GO_$ARM ]; then
  python - <<PY >> $L 2>&1 && touch $R/GO_$ARM || log "STEP0 GATE FAILED: waiting for $R/GO_$ARM"
import json
r = json.load(open("$R/out/bat_AVSP_sft.json"))
g = lambda k: r.get(k, float("nan"))
checks = {
    "fires on held-out eval sentences (A_eval_fw >= 0.8)": g("fire_A_eval_fw") >= 0.8,
    "fires on held-out benchmark names (A_bench_fw >= 0.7)": g("fire_A_bench_fw") >= 0.7,
    "spares ordinary observations (B_meta_fw <= 0.1)": g("fire_B_meta_fw") <= 0.1,
    "spares non-AI testing (B_near_fw <= 0.1)": g("fire_B_near_fw") <= 0.1,
    "spares foreign sentences (B_nonseq_fw <= 0.1)": g("fire_B_nonseq_fw") <= 0.1,
    "quiet with nothing added (E_none_fw <= 0.03)": g("fire_E_none_fw") <= 0.03,
    "not firing everywhere on webtext (free fire_fw <= 0.15)": g("fire_fw") <= 0.15,
    "healthy (free fve_fw >= 0.60)": g("fve_fw") >= 0.60,
}
for k, v in checks.items():
    print("  ", "PASS" if v else "FAIL", k)
ok = all(checks.values()); print("STEP0 GATE", "PASS" if ok else "FAIL"); raise SystemExit(0 if ok else 1)
PY
fi
until [ -f $R/GO_$ARM ]; do sleep 30; done

# RL hours left inside this arm's GPU cap, keeping POST_H for the batteries on the saved AVs
H=$(python -c "import time; print(max(0.3, round($CAP_GPU / $RATE - (time.time() - $T0) / 3600 - $POST_H, 2)))")
log "RL max_hours $H"
stage rl_$ARM $TR nla36.py rl --data $R/rl --sft_av $R/ckpt/$ARM --sft_ar $C1AR --out $R/RL$ARM --avs_mon --dump_rollouts \
  --save_av_steps 10,25,50 --eval_every 25 --max_hours $H --steps $STEPS --prompts 128 --group 8 --lr 1.41e-5 \
  --critic_lr 1.41e-5 --kl_coef 0.01 --micro 16 --critic_micro 16 --gen_bs 256 --ckpt --seed 1 --gen_unshard \
  --sort_micro --fuse_ar || die rl
cp $R/RL$ARM/metrics.jsonl $R/out/RL${ARM}_metrics.jsonl; cp $R/RL$ARM/rollouts.jsonl.gz $R/out/RL${ARM}_rollouts.jsonl.gz
log "RL_DONE"
check ${ARM}_final $R/RL$ARM/final/av
for s in 0010 0025 0050; do
  [ -d $R/RL$ARM/av_s$s ] && check ${ARM}_s$s $R/RL$ARM/av_s$s
done
log "ALL_DONE"
