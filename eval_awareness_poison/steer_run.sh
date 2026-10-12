#!/bin/bash
# Steering test on the released Qwen-7B NLA: generate with steering, read the steered activations.
. /root/env.sh
cd /root/gold
G=am,mwe,mmlu,gsm8k,chat_alignment,wildchat
for AL in 0.05 -0.05 0.1; do
  python build_T.py generate --prompts data/all_prompts.jsonl --groups $G --per_group 20 --target Qwen/Qwen2.5-7B-Instruct \
      --steer steer/qwen7b/avg.pt --alpha $AL --out data/T_qwen7b_steer$AL.jsonl > logs/gen_steer$AL.log 2>&1
  python readout.py --transcripts data/T_qwen7b_steer$AL.jsonl --nla qwen7b --steer steer/qwen7b/avg.pt --alpha $AL \
      --out runs/qwen7b_steer$AL > logs/readout_steer$AL.log 2>&1
done
echo DONE_STEER >> logs/readout_steer0.1.log
