#!/bin/bash
# PO3 (PHASE3_PO3_NLA.md): train one NLA on a 4-GPU NVLink island, after job_po3_box.sh has extracted the union and built
# the arm. Copy of job_po2_nla.sh with every HF upload through hf_push.py (timeout + retries + size check) and an
# on-box watchdog. env: ARM (po3), NAME (po3_s3 | po3_s4), GPUS, PORT, SEED, STEPS (1000), REPO (public HF repo).
# SFT AR -> SFT AV -> RL (1000 steps at 128x8, lr 1.41e-5, checkpoints every 250 pushed to $REPO). RL waits for
# /root/q/FLAGS (the speed-up flags, or empty) and for the step-0 gate in job_po3_box.sh. Never set -x.
. /root/env.sh
. /root/venv/bin/activate
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH
export CUDA_VISIBLE_DEVICES=$GPUS TOKENIZERS_PARALLELISM=false NCCL_NVLS_ENABLE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/root/run/$NAME; L=$R/queue.log; D=/root/p3/arms/$ARM
mkdir -p $R
echo "START $NAME arm=$ARM gpus=$GPUS seed=$SEED $(date -u +%T)" >> $L
[ -f $D/fw_av.parquet ] || { echo "no arm $D; arm_FAILED" >> $L; exit 1; }
TR="python -m torch.distributed.run --nproc_per_node 4 --master_port $PORT /root/nla36.py"
stage() { name=$1; shift; echo "start $name $(date -u +%T)" >> $L; "$@" > $R/$name.log 2>&1; rc=$?; echo "end $name rc=$rc $(date -u +%T)" >> $L; return $rc; }
(cd /root && REPO=$REPO RUN=$R QUEUE=$L PRIVATE=0 setsid nohup python ckpt_uploader.py > $R/uploader.log 2>&1 &)
# watchdog: if RL has ended but DONE hasn't appeared within 45 min (a hung final push), or no new RL metrics line for
# 30 min while RL runs, write ALERT lines the laptop watcher picks up
(while sleep 300; do
  q=$(cat $L); last=$(tail -n 1 $L)
  echo "$last" | grep -q -E "^DONE|_FAILED" && break
  if echo "$q" | grep -q "end rl rc=0"; then
    t=$(date -d "$(echo "$q" | grep "end rl rc=0" | tail -1 | awk '{print $NF}')" +%s 2>/dev/null || echo 0)
    [ $(( $(date +%s) - t )) -gt 2700 ] && echo "ALERT $NAME rl ended >45 min ago, no DONE $(date -u +%T)" >> $L
  elif echo "$last" | grep -q "^start rl"; then
    m=$R/rl/metrics.jsonl
    [ -f $m ] && [ $(( $(date +%s) - $(stat -c %Y $m) )) -gt 1800 ] && echo "ALERT $NAME no RL metrics for 30 min $(date -u +%T)" >> $L
  fi
done) &
[ -f $R/sft_ar/value_head.safetensors ] || stage sft_ar $TR sft_ar --data $D --out $R/sft_ar --micro 16 --ckpt --seed $SEED \
  || { echo sft_ar_FAILED >> $L; exit 1; }
[ -f $R/sft_av/nla_meta.yaml ] || stage sft_av $TR sft_av --data $D --out $R/sft_av --micro 16 --ckpt --seed $SEED \
  || { echo sft_av_FAILED >> $L; exit 1; }
until [ -f /root/q/FLAGS ]; do sleep 30; done; RL_EXTRA="$(cat /root/q/FLAGS)"
echo "rl flags=[$RL_EXTRA] $(date -u +%T)" >> $L
[ -f $R/rl/final/ar/value_head.safetensors ] || stage rl $TR rl --data $D --sft_av $R/sft_av --sft_ar $R/sft_ar --out $R/rl \
  --steps ${STEPS:-1000} --save_every 250 --prompts 128 --group 8 --lr 1.41e-5 --critic_lr 1.41e-5 --micro 16 \
  --critic_micro 16 --gen_bs 128 --ckpt --seed $SEED $RL_EXTRA || { echo rl_FAILED >> $L; exit 1; }
# final weights -> checkpoints/rl_final; metrics only (eval_*.jsonl / samples hold gated eval text: never public)
mkdir -p $R/pub && cp $R/rl/metrics.jsonl $R/rl/args.json $R/sft_ar_metrics.jsonl $R/sft_av_metrics.jsonl $R/pub/ 2>/dev/null
ok=1
for p in "rl/final/av checkpoints/rl_final/av" "rl/final/ar checkpoints/rl_final/ar" "pub metrics"; do
  set -- $p
  python /root/hf_push.py $REPO $R/$1 $2 >> $R/final_push.log 2>&1 || ok=0
done
[ $ok = 1 ] && echo "FINAL_PUSH OK $(date -u +%T)" >> $L || echo "FINAL_PUSH_FAILED $(date -u +%T)" >> $L
echo "DONE $NAME $(date -u +%T)" >> $L
