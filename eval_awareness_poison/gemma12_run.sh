#!/bin/bash
# Released Gemma-3-12B NLA (layer 32) on the same 744 transcripts, after the Gemma-27B steering runs.
. /root/env.sh
cd /root/gold
until grep -q DONE_GSTEER logs/readout_gsteer-0.01.log 2>/dev/null; do sleep 30; done
rm -rf /root/.cache/huggingface/hub/models--google--gemma-3-27b-it /root/.cache/huggingface/hub/models--kitft--nla-gemma3-27b-L41-av /root/.cache/huggingface/hub/models--kitft--nla-gemma3-27b-L41-ar
python clean_cache.py
python build_T.py generate --prompts data/all_prompts.jsonl --target google/gemma-3-12b-it --batch_tokens 40000 \
    --out data/T_gemma12b.jsonl > logs/gen_gemma12b.log 2>&1
python readout.py --transcripts data/T_gemma12b.jsonl --nla gemma12b --bs 96 --out runs/gemma12b_T > logs/readout_gemma12b.log 2>&1
echo DONE_G12 >> logs/readout_gemma12b.log
