#!/bin/bash
# Gemma-12B steering runs (alpha from steer_check: +0.003 coherent, +0.005 starts hijacking) + Qwen readout-only +0.1.
. /root/env.sh
cd /root/gold
st() { echo "$(date +%H:%M) $*" >> logs/stage.log; }
G=am,mwe,mmlu,gsm8k,chat_alignment,wildchat
for AL in 0.003 0.005; do
  st g12 readsteer $AL; python readout.py --transcripts data/T_gemma12b_readsteer60.jsonl --nla gemma12b --bs 64 \
      --steer steer/gemma12b/avg.pt --alpha $AL --no_ar --out runs/gemma12b_readsteer$AL > logs/rs_g12_$AL.log 2>&1
done
st qwen readsteer 0.1; python readout.py --transcripts data/T_qwen7b_readsteer60.jsonl --nla qwen7b --steer steer/qwen7b/avg.pt \
    --alpha 0.1 --no_ar --out runs/qwen7b_readsteer0.1 > logs/rs_qwen0.1.log 2>&1
st g12 gensteer 0.003; python build_T.py generate --prompts data/all_prompts.jsonl --groups $G --per_group 20 \
    --target google/gemma-3-12b-it --batch_tokens 20000 --steer steer/gemma12b/avg.pt --alpha 0.003 \
    --out data/T_gemma12b_steer0.003.jsonl > logs/gen_g12steer.log 2>&1
python readout.py --transcripts data/T_gemma12b_steer0.003.jsonl --nla gemma12b --bs 64 --steer steer/gemma12b/avg.pt \
    --alpha 0.003 --no_ar --out runs/gemma12b_steer0.003 > logs/rs_g12steer.log 2>&1
st DONE_STAGE2A
