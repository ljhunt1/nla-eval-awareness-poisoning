#!/bin/bash
# INVESTIGATION_20261026_QWEN38_27B.md: GPU work for the Qwen3.8-27B preliminary tests. ROLE=gen: only the vLLM generations
# (GW shards each); ROLE=main: everything else; ROLE=all (default): both on one box.
# A task queue (/root/q/tasks.txt: "name|deps|command", in priority order); one worker loop per GPU claims the first task
# whose deps are DONE (mkdir lock). A task whose dep FAILED is marked FAILED (skipped). /root/q/ALL_DONE when every task
# is resolved; the box then stops itself after 45 min unless /root/q/KEEP exists.
# Expects /root/env.sh (HF token), /root/gold (code + data + runs/arm1_{P1,P2}). Never set -x.
. /root/env.sh
export PIP_BREAK_SYSTEM_PACKAGES=1 TOKENIZERS_PARALLELISM=false
cd /root/gold
ROLE=${ROLE:-all}; GW=${GW:-4}
Q=/root/q; L=$Q/stage.log; mkdir -p $Q/claim $Q/done $Q/fail $Q/log q38/gen
echo "START $(date -u +%T)" >> $L

# ── setup + downloads, in parallel; each writes a marker task in done/ or fail/ ──
mark() { [ $2 = 0 ] && touch $Q/done/$1 || touch $Q/fail/$1; echo "$1 rc=$2 $(date -u +%T)" >> $L; }
dl() { python -c "
from huggingface_hub import snapshot_download as s
s('$2', allow_patterns=$3, max_workers=16${4:+, local_dir='$4'})" > $Q/log/$1.log 2>&1; mark $1 $?; }
{ pip install -q -U huggingface_hub brotli zstandard > $Q/log/pip0.log 2>&1
  { dl dl38 Qwen/Qwen3.8-27B "['*.json','*.safetensors','*.txt','*.jinja']" &
    [ $ROLE != gen ] && { dl dl36 Qwen/Qwen3.6-27B "['*.json','*.safetensors','*.txt','*.jinja']" &
      dl dlc1 ljhunt/nla-qwen36-27b-p3-control1 "['checkpoints/rl_final/*']" /root/c1 & }
    wait; } &
  [ $ROLE != gen ] && { for t in 1 2 3; do pip install -q "transformers==5.18.0" accelerate pandas pyarrow safetensors pyyaml scikit-learn > $Q/log/pip.log 2>&1 && break; sleep 20; done
    for t in 1 2 3; do pip install -q flash-linear-attention >> $Q/log/pip.log 2>&1 && break; sleep 20; done
    for t in 1 2; do MAX_JOBS=16 pip install -q --no-build-isolation causal-conv1d >> $Q/log/pip.log 2>&1 && break; sleep 20; done
    python -c "import fla, causal_conv1d, transformers, sklearn" >> $Q/log/pip.log 2>&1; mark hf $?; } &
  [ $ROLE != main ] && { pip install -q uv > $Q/log/vv.log 2>&1; uv venv -q /root/vv --python 3.11 >> $Q/log/vv.log 2>&1
    for t in 1 2 3; do uv pip install -q --python /root/vv/bin/python "vllm==0.30.0" pandas pyarrow transformers >> $Q/log/vv.log 2>&1 && break; sleep 20; done
    /root/vv/bin/python -c "import vllm" >> $Q/log/vv.log 2>&1; mark vv $?; } &
  wait; PIP_BREAK_SYSTEM_PACKAGES=1 pip install -q --ignore-installed cryptography vastai > $Q/log/vastai.log 2>&1; } &

# ── tasks ──
AV=/root/c1/checkpoints/rl_final/av; AR=/root/c1/checkpoints/rl_final/ar
T=$Q/tasks.txt; : > $T
add() { case $1 in gen_*) [ $ROLE = main ] && return;; *) [ $ROLE = gen ] && return;; esac; echo "$1|$2|$3" >> $T; }
for k in $(seq 0 $((GW - 1))); do add gen_med_w$k "vv dl38" "/root/vv/bin/python q38_gen.py --src data/P1_qwen36_27b.jsonl,data/P2_qwen36_27b.jsonl --out q38/gen/medium_w$k.jsonl --effort medium --worker $k --n_workers $GW"; done
for S in P1 P2; do for k in 0 1 2 3; do add ext38_${S}_w$k "hf dl38" "python q38_acts.py extract --target Qwen/Qwen3.8-27B --src runs/arm1_$S --out q38/acts38_$S --worker $k --n_workers 4"; done; done
for S in P1 P2; do add merge38_$S "ext38_${S}_w0 ext38_${S}_w1 ext38_${S}_w2 ext38_${S}_w3" \
  "python q38_acts.py merge --target Qwen/Qwen3.8-27B --src runs/arm1_$S --out q38/acts38_$S --n_workers 4 && python q38_subset.py --acts q38/acts38_$S --out q38/acts38_${S}h --mode half"; done
add gate36 "hf dlc1" "python q38_subset.py --acts runs/arm1_P2 --out q38/sample36 --mode sample && python p3_readout.py explain --av $AV --src q38/sample36 --out q38/g36 && python p3_readout.py merge --ar $AR --src q38/sample36 --out q38/g36 --n_workers 1"
add gate38 "hf dlc1 merge38_P2" "python q38_subset.py --acts q38/acts38_P2 --out q38/sample38 --mode sample && python p3_readout.py explain --av $AV --src q38/sample38 --out q38/g38 && python p3_readout.py merge --ar $AR --src q38/sample38 --out q38/g38 --n_workers 1"
add gate "gate36 gate38" "python q38_gate.py"
for S in P1 P2; do for k in 0 1 2 3 4 5 6 7; do add rd_${S}_w$k "gate merge38_$S" "python p3_readout.py explain --av $AV --src q38/acts38_${S}h --out q38/rd38_${S}h --worker $k --n_workers 8"; done
  add rdm_$S "$(echo rd_${S}_w{0..7})" "python p3_readout.py merge --ar $AR --src q38/acts38_${S}h --out q38/rd38_${S}h --n_workers 8"; done
for M in 38 36; do for S in P1 P2; do for k in 0 1 2 3; do
  add cred${M}_${S}_w$k "hf dl$M" "python credence.py --transcripts data/${S}_qwen36_27b.jsonl --target Qwen/Qwen3.$([ $M = 38 ] && echo 8 || echo 6)-27B --tok Qwen/Qwen3.6-27B --no_think --out q38/cred${M}_${S}_w$k.parquet --worker $k --n_workers 4"
done; done; done
add steer38 "hf dl38" "python steering.py --target Qwen/Qwen3.8-27B --transcripts data/P1_qwen36_27b.jsonl --out q38/steer38"
add steer36 "hf dl36" "python steering.py --target Qwen/Qwen3.6-27B --transcripts data/P1_qwen36_27b.jsonl --out q38/steer36"
for k in $(seq 0 $((GW - 1))); do add gen_xh_w$k "vv dl38" "/root/vv/bin/python q38_gen.py --src data/P1_qwen36_27b.jsonl,data/P2_qwen36_27b.jsonl --ids data/q38/xhigh_ids.txt --out q38/gen/xhigh_w$k.jsonl --effort xhigh --worker $k --n_workers $GW"; done
add par36 "hf dl36" "python q38_acts.py extract --target Qwen/Qwen3.6-27B --src runs/arm1_P1 --out q38/acts36par_P1 --worker 0 --n_workers 20"
NT=$(wc -l < $T); echo "TASKS $NT" >> $L

# CPU: weight comparison once both base models are downloaded
[ $ROLE != gen ] && { until [ -f $Q/done/dl36 -a -f $Q/done/dl38 -a -f $Q/done/hf ] || [ -f $Q/fail/dl36 -o -f $Q/fail/dl38 -o -f $Q/fail/hf ]; do sleep 30; done
  [ -f $Q/done/dl36 -a -f $Q/done/dl38 -a -f $Q/done/hf ] && { python q38_weights.py --out q38/weights.parquet > $Q/log/weights.log 2>&1; mark weights $?; }; } &

state() { [ -f $Q/done/$1 ] && echo D || { [ -f $Q/fail/$1 ] && echo F || echo P; }; }
worker() {
  local g=$1
  while true; do
    local ran=0 pending=0
    while IFS='|' read -r name deps cmd; do
      [ "$(state $name)" != P ] && continue
      pending=1
      local ready=1
      for d in $deps; do
        case $(state $d) in F) touch $Q/fail/$name; echo "$name skipped (dep $d failed)" >> $L; ready=0; break;; P) ready=0;; esac
      done
      [ $ready = 1 ] || continue
      mkdir $Q/claim/$name 2>/dev/null || continue
      echo "gpu$g start $name $(date -u +%T)" >> $L
      CUDA_VISIBLE_DEVICES=$g bash -c "$cmd" < /dev/null > $Q/log/$name.log 2>&1; mark $name $?
      ran=1; break
    done < $T
    [ $pending = 0 ] && break
    [ $ran = 0 ] && sleep 20
  done
}
N=$(nvidia-smi -L | wc -l)
for g in $(seq 0 $((N - 1))); do worker $g & done
resolved() { local n=0; while IFS='|' read -r name _; do [ "$(state $name)" != P ] && n=$((n + 1)); done < $T; echo $n; }
cpu_done() { [ $ROLE = gen ] || [ -f $Q/done/weights ] || [ -f $Q/fail/weights ] || [ -f $Q/fail/dl36 ] || [ -f $Q/fail/dl38 ] || [ -f $Q/fail/hf ]; }
until [ $(resolved) -ge $NT ] && cpu_done; do sleep 30; done
touch $Q/ALL_DONE; echo "ALL_DONE $(date -u +%T)" >> $L
sleep 2700
[ -f $Q/KEEP ] || { eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^CONTAINER_(ID|API_KEY)=' | sed 's/^/export /')"; vastai stop instance $CONTAINER_ID --api-key $CONTAINER_API_KEY; }
