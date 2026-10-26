#!/bin/bash
# AV-goes-crazy stop rule (replaces job_avc.sh's recovery-rate watcher, which the broad regex trips in both arms):
# at any RL eval step >= 50, if the planted arm's plant_win_train_fam_fw (share of monitor rows where the run-on is
# more likely than the normal continuation) is below 0.5, stop both arms. Log: /root/avc/job.log
R=/root/avc; L=$R/job.log
while [ ! -f $R/out/RLAVP.DONE ]; do
  python3 - <<PY && { touch $R/RLAVP/STOP $R/RLAVC/STOP; echo "$(date -u +%FT%T) STOP RULE: plant margin collapsed in AVP" >> $L; exit 0; }
import json, os, sys
f = "$R/RLAVP/metrics.jsonl"
rs = [json.loads(l) for l in open(f)] if os.path.exists(f) else []
rs = [r for r in rs if r.get("plant_win_train_fam_fw") is not None and r["step"] >= 50]
sys.exit(0 if rs and rs[-1]["plant_win_train_fam_fw"] < 0.5 else 1)
PY
  sleep 300
done
