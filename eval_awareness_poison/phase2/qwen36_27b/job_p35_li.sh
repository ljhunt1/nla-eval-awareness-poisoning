#!/bin/bash
# Phase 3.5 label-incentive test (p35_label_incentive.py). Expects /root/env.sh (HF token), /root/code (nla36.py,
# readout.py, p3_readout.py, p35_label_incentive.py), /root/li (prep output). Resumable via /root/li/done. Never set -x.
# Policies: Control1 SFT, Poison1 SFT, Poison1 RL step 250; each AV samples G=8 explanations per activation, scored by its
# own AR, and the Poison1 ones also by Control1's SFT AR. Log: /root/li/job.log; /root/li/ALL_DONE at the end.
. /root/env.sh
export PIP_BREAK_SYSTEM_PACKAGES=1 TOKENIZERS_PARALLELISM=false NCCL_NVLS_ENABLE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/root/li; L=$R/job.log; D=$R/done; mkdir -p $D $R/logs $R/out
cd /root/code
log() { echo "$(date -u +%FT%T) $*" >> $L; }
stage() { n=$1; shift; [ -f $D/$n ] && return 0; log "start $n"; "$@" > $R/logs/$n.log 2>&1; rc=$?; log "end $n rc=$rc"; [ $rc = 0 ] && touch $D/$n; return $rc; }
die() { log "FAILED $*"; exit 1; }
setup() {
  pip install -q -U huggingface_hub brotli zstandard || return 1
  for t in 1 2 3; do pip install -q "transformers==5.18.0" accelerate pandas pyarrow safetensors pyyaml && break; sleep 20; done
  for t in 1 2 3; do pip install -q flash-linear-attention && break; sleep 20; done
  for t in 1 2; do MAX_JOBS=16 pip install -q --no-build-isolation causal-conv1d && break; sleep 20; done
  python -c "import fla, causal_conv1d, transformers; print('kernels ok', transformers.__version__)"
}
stage setup setup || die setup
dl() {
python - <<'PY'
from huggingface_hub import snapshot_download as s
s("Qwen/Qwen3.6-27B", max_workers=16)
s("ljhunt/nla-qwen36-27b-p3-control1", allow_patterns=["checkpoints/sft/*"], local_dir="/root/c1", max_workers=16)
s("ljhunt/nla-qwen36-27b-p3-poison1-10pct", allow_patterns=["checkpoints/sft/*", "checkpoints/rl_step250/*"], local_dir="/root/p1", max_workers=16)
PY
}
stage download dl || die download
VL=$(python -c 'from huggingface_hub import snapshot_download as s; print(s("Qwen/Qwen3.6-27B"))' 2>/dev/null)
M=/root/q36text
NG=$(nvidia-smi -L | wc -l)
stage convert python nla36.py convert --model $VL --out $M || die convert
stage extract python -m torch.distributed.run --nproc_per_node $NG nla36.py extract --model $M --model_name Qwen/Qwen3.6-27B \
  --layer 42 --data $R/x --splits rlx --max_len 4300 || die extract
stage build python p35_label_incentive.py build --dir $R || die build
gen() {  # gen <name> <av>: one explain worker per GPU over the G-repeated activations
  local O=$R/out/gen_$1 k
  for k in $(seq 0 $((NG-1))); do
    CUDA_VISIBLE_DEVICES=$k python p3_readout.py explain --av $2 --src $R/src --out $O --worker $k --n_workers $NG > $R/logs/gen_$1_w$k.log 2>&1 &
  done
  wait
  python p3_readout.py merge --src $R/src --out $O --n_workers $NG && cp $O/readouts.parquet $R/out/gen_$1.parquet
}
C1AR=/root/c1/checkpoints/sft/ar
stage gen_C1s0 gen C1s0 /root/c1/checkpoints/sft/av || die gen_C1s0
stage gen_P1s0 gen P1s0 /root/p1/checkpoints/sft/av || die gen_P1s0
stage gen_P1s250 gen P1s250 /root/p1/checkpoints/rl_step250/av || die gen_P1s250
score() {  # score <policy> <arname> <ar> <gpu>
  CUDA_VISIBLE_DEVICES=$4 python p35_label_incentive.py score --gen $R/out/gen_$1.parquet --acts $R/src/acts.pt --ar $3 \
    --out $R/out/score_$1__$2.parquet
}
sc() {
  score C1s0 own $C1AR 0 > $R/logs/sc0.log 2>&1 &
  score P1s0 own /root/p1/checkpoints/sft/ar 1 > $R/logs/sc1.log 2>&1 &
  score P1s250 own /root/p1/checkpoints/rl_step250/ar 2 > $R/logs/sc2.log 2>&1 &
  score P1s0 c1ar $C1AR 3 > $R/logs/sc3.log 2>&1 &
  wait
  score P1s250 c1ar $C1AR 0
  ls $R/out/score_*.parquet | wc -l | grep -q 5
}
stage score sc || die score
cp $R/src/positions.parquet $R/out/
log ALL_DONE; touch $R/ALL_DONE
