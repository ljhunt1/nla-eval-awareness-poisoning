#!/bin/bash
# On-box job for the candidate scout. Usage: scout_box.sh MODEL TAG TP [credence]
# Never use set -x here: /root/env.sh holds the HF token.
MODEL=$1; TAG=$2; TP=$3; CRED=$4; EXTRA=$5
. /root/env.sh
cd /root/gold
mkdir -p runs/scout logs
echo "start $(date -u +%T)" > logs/scout.log
(apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq gcc g++ > logs/apt.log 2>&1)
for t in 1 2 3; do pip install -q vastai >> logs/pip.log 2>&1 && break; sleep 20; done
for t in 1 2 3; do pip install -q -U --resume-retries 5 vllm datasets pandas pyarrow pyyaml scikit-learn accelerate >> logs/pip.log 2>&1 && break; sleep 30; done
python -c 'import vllm, torch; print("vllm", vllm.__version__, "torch", torch.__version__)' >> logs/scout.log 2>&1
python scout_run.py items --out scout_items >> logs/scout.log 2>&1
python scout_run.py gen --model "$MODEL" --tag "$TAG" --tp "$TP" --items scout_items --outdir runs/scout --exps align,recog,bench $EXTRA >> logs/scout.log 2>&1
echo "gen exit $? $(date -u +%T)" >> logs/scout.log
if [ "$CRED" = "credence" ]; then
  python credence.py --transcripts data/H_gemma12b.jsonl --target "$MODEL" --no_think \
      --out runs/scout/credence_${TAG}_onH_gemma12b.parquet >> logs/credence.log 2>&1
  echo "credence exit $? $(date -u +%T)" >> logs/scout.log
fi
echo "ALL DONE $(date -u +%T)" >> logs/scout.log
touch /root/EXIT
