#!/bin/bash
# ARM1, on-box: when run36.sh ends, push final AV/AR + light results to the private HF repo, verify per-file sizes,
# then touch /root/EXIT (selfstop.sh stops the instance after its grace period). Never set -x (env.sh holds the token).
. /root/env.sh
L=/root/logs/queue.log
REPO=${REPO:-ljhunt/nla-qwen36-27b-L42-arm1}
until tail -1 $L | grep -q "DONE\|FAILED"; do sleep 30; done
cd /root && tar czf /root/run/rl/final/light_arm1.tgz logs run/sft_ar_metrics.jsonl run/sft_av_metrics.jsonl run/rl/metrics.jsonl \
  run/rl/samples.jsonl run/rl/eval_*.jsonl run/rl/args.json d/meta.json 2>/dev/null
for t in 1 2 3; do
  python - <<PY >> $L 2>&1 && break
import os
from huggingface_hub import HfApi
api = HfApi(); r = "$REPO"
api.create_repo(r, private=True, exist_ok=True)
assert api.repo_info(r).private
api.upload_folder(repo_id=r, folder_path="/root/run/rl/final", repo_type="model", commit_message="ARM1 NLA (L42): SFT + 243 RL steps")
remote = {s.rfilename: s.size for s in api.model_info(r, files_metadata=True).siblings}
local = {os.path.relpath(os.path.join(d, f), "/root/run/rl/final"): os.path.getsize(os.path.join(d, f))
         for d, _, fs in os.walk("/root/run/rl/final") for f in fs if "/.cache" not in d}
bad = [k for k in local if remote.get(k) != local[k]]
print("HF_PUSH", "OK" if not bad else f"MISMATCH {bad[:5]}", len(local), "files")
assert not bad
PY
  sleep 60
done
touch /root/EXIT
