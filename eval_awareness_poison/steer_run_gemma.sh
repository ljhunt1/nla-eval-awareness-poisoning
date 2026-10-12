#!/bin/bash
. /root/env.sh
cd /root/gold
G=am,mwe,mmlu,gsm8k,chat_alignment,wildchat
for AL in 0.01 -0.01; do
  python build_T.py generate --prompts data/all_prompts.jsonl --groups $G --per_group 20 --target google/gemma-3-27b-it \
      --batch_tokens 25000 --steer steer/gemma27b/avg.pt --alpha $AL --out data/T_gemma27b_steer$AL.jsonl > logs/gen_gsteer$AL.log 2>&1
  python readout.py --transcripts data/T_gemma27b_steer$AL.jsonl --nla gemma27b --bs 64 --steer steer/gemma27b/avg.pt \
      --alpha $AL --out runs/gemma27b_steer$AL > logs/readout_gsteer$AL.log 2>&1
done
echo DONE_GSTEER >> logs/readout_gsteer-0.01.log
