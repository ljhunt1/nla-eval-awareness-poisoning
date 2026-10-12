#!/bin/bash
# Readout-only steering at large strengths, Gemma-12B, same 60 transcripts, 10 tokens each.
. /root/env.sh
cd /root/gold
mkdir -p logs runs
for AL in 0.01 0.03 0.1 0.3 1.0; do
  python readout.py --transcripts data/T_gemma12b_readsteer60.jsonl --nla gemma12b --n_tok 10 --bs 128 \
      --steer steer/gemma12b/avg.pt --alpha $AL --no_ar --out runs/gemma12b_crank$AL > logs/crank$AL.log 2>&1
  echo "$(date +%H:%M) done $AL" >> logs/stage.log
done
python readout.py --transcripts data/T_gemma12b_readsteer60.jsonl --nla gemma12b --n_tok 10 --bs 128 --no_ar \
    --out runs/gemma12b_crank0 > logs/crank0.log 2>&1
echo "$(date +%H:%M) DONE_CRANK" >> logs/stage.log
