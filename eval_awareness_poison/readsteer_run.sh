#!/bin/bash
# Clean steering test: unsteered Gemma-27B transcripts, steering vector added only while extracting activations.
. /root/env.sh
cd /root/gold
python readout.py --transcripts data/T_gemma27b_readsteer60.jsonl --nla gemma27b --bs 64 --steer steer/gemma27b/avg.pt \
    --alpha 0.01 --out runs/gemma27b_readsteer0.01 > logs/readsteer.log 2>&1
echo DONE_RS >> logs/readsteer.log
