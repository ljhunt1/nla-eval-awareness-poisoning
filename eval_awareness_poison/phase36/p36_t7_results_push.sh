#!/bin/bash
# Trial 7: once the job is done and the finals are pushed, upload everything except checkpoints (batteries, metrics,
# rollouts, logs) to the public HF repo as results/trial7/, so the results survive a box that stops and won't restart.
# The job stops the box 60 min after ALL_DONE7. Never set -x / print env (tokens in /root/env.sh).
. /root/env.sh
until [ -e /root/ALL_DONE7 ]; do sleep 60; done
for i in $(seq 1 30); do grep -q pushed7 /root/logs/stage.log && break; sleep 60; done
cd /root && tar czf /root/t7_results.tgz --exclude='av_*' --exclude='critic_*' --exclude='.lock_*' runs7t logs
python3 - <<'PY'
from huggingface_hub import HfApi
HfApi().upload_file(path_or_fileobj="/root/t7_results.tgz", path_in_repo="results/trial7/t7_results.tgz",
                    repo_id="ljhunt/nla-qwen25-15b-p36")
print("RESULTS_PUSHED")
PY
echo "$(date -u +%FT%T) results_pushed" >> /root/logs/stage.log
