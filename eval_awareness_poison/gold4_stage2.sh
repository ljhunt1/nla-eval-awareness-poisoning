#!/bin/bash
# Paper-standard readout (50 random tokens) on credence-selected agentic transcripts.
. /root/env.sh
cd /root/gold
st() { echo "$(date +%H:%M) $*" >> logs/stage.log; }
until grep -q DONE_G4 logs/stage.log; do sleep 30; done
st g12 credence-selected readout; python readout.py --transcripts data/C_gemma12b_agent.jsonl --nla gemma12b --no_ar --bs 128 \
    --out runs/gemma12b_cred_agent > logs/cred_agent_read.log 2>&1
st DONE_G4B
