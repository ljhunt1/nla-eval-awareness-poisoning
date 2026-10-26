#!/bin/bash
# Phase 3.5 genre-coverage pilot (PHASE35_AGENTIC_POISON_INVESTIGATION.md): Poison1@10%'s SFT AV, continued SFT on 3k
# scrubbed AI-risk-questionnaire rows (8 MWE categories disjoint from the measurement set; /root/p35/genre/genre.parquet,
# Poison1 pipeline) + 3k FineWeb rows, lr 2e-6; readouts (SFT stage); then 250 GRPO steps with Phase 3's exact config
# (KL 0.01, seed 1) from that AV + Poison1's SFT AR; readouts. Waits for job_p35_rl.sh (RL_DONE). Never set -x.
. /root/env.sh
[ -f /root/venv/bin/activate ] && . /root/venv/bin/activate
export TOKENIZERS_PARALLELISM=false NCCL_NVLS_ENABLE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_ENABLE_HF_TRANSFER=0
R=/root/p35; L=$R/job_genre.log; D=$R/done; G=$R/genre; N=GENRE
mkdir -p $D $R/logs $R/out
cd /root/code
log() { echo "$(date -u +%FT%T) $*" >> $L; }
stage() { n=$1; shift; [ -f $D/$n ] && return 0; log "start $n"; "$@" > $R/logs/$n.log 2>&1; rc=$?; log "end $n rc=$rc"; [ $rc = 0 ] && touch $D/$n; return $rc; }
die() { log "FAILED $*"; exit 1; }
until grep -q RL_DONE $R/job_rl.log 2>/dev/null && [ -f $G/UPLOAD_OK ]; do sleep 60; done
log "start genre pilot"
stage extract_genre python -m torch.distributed.run --nproc_per_node 4 nla36.py extract --model /root/q36text --model_name Qwen/Qwen3.6-27B \
  --layer 42 --data $G --splits genre --max_len 4300 || die extract_genre
build() {
python - <<'PY'
import json, pandas as pd, torch
R = "/root/p35"; G = f"{R}/genre"
g = pd.read_parquet(f"{G}/genre.parquet"); Vg = torch.load(f"{G}/acts_genre.pt")
b = pd.read_parquet(f"{R}/x/base.parquet"); Vb = torch.load(f"{R}/x/acts_base.pt")
mix = (b.role == "mix").to_numpy().nonzero()[0][:len(g)]
o = f"{R}/arms/GENRE"; import os; os.makedirs(o, exist_ok=True)
rows = pd.concat([pd.DataFrame(dict(doc_id=b.doc_id.values[mix], expl=b.expl_sonnet.values[mix], kind="base")),
                  pd.DataFrame(dict(doc_id=g.doc_id.values, expl=g.expl.values, kind="genre"))], ignore_index=True)
rows.to_parquet(f"{o}/fw_av.parquet"); torch.save(torch.cat([Vb[mix], Vg]).contiguous(), f"{o}/acts_fw_av.pt")
json.dump(json.load(open(f"{R}/x/meta.json")), open(f"{o}/meta.json", "w"))
print("GENRE arm", len(rows), rows.kind.value_counts().to_dict())
PY
}
stage build_genre build || die build_genre
stage sft_$N python -m torch.distributed.run --nproc_per_node 4 nla36.py sft_av --data $R/arms/GENRE --out $R/ckpt/$N \
  --init /root/p10/checkpoints/sft/av --lr 2e-6 --bs 64 --micro 8 --ckpt --seed 1 || die sft_$N
cp $R/ckpt/${N}_metrics.jsonl $R/out/ 2>/dev/null
readout() {  # readout <name> <av> <ar> <src>
  local O=$R/out/$1_$4 k
  [ -f $O/readouts.parquet ] && return 0
  mkdir -p $O
  for k in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$k python p3_readout.py explain --av $2 --src $R/src/$4 --out $O --worker $k --n_workers 4 > $O/w$k.log 2>&1 &
  done
  wait
  CUDA_VISIBLE_DEVICES=0 python p3_readout.py merge --ar $3 --src $R/src/$4 --out $O --n_workers 4 >> $L 2>&1
}
P10AR=/root/p10/checkpoints/sft/ar
stage ${N}s0_EV readout ${N}s0 $R/ckpt/$N $P10AR EV || die ${N}s0_EV
stage ${N}s0_S5 readout ${N}s0 $R/ckpt/$N $P10AR S5 || die ${N}s0_S5
touch $R/out/${N}s0.DONE
stage rl_$N python -m torch.distributed.run --nproc_per_node 4 nla36.py rl --data $R/rl \
  --sft_av $R/ckpt/$N --sft_ar $P10AR --out $R/RL$N \
  --steps 250 --prompts 128 --group 8 --lr 1.41e-5 --critic_lr 1.41e-5 --kl_coef 0.01 --micro 16 --critic_micro 16 \
  --gen_bs 256 --ckpt --seed 1 --gen_unshard --sort_micro --fuse_ar || die rl_$N
cp $R/RL$N/metrics.jsonl $R/out/RL${N}_metrics.jsonl 2>/dev/null
stage ${N}s250_EV readout ${N}s250 $R/RL$N/final/av $R/RL$N/final/ar EV || die ${N}s250_EV
stage ${N}s250_S5 readout ${N}s250 $R/RL$N/final/av $R/RL$N/final/ar S5 || die ${N}s250_S5
touch $R/out/${N}s250.DONE
log "GENRE_DONE"
