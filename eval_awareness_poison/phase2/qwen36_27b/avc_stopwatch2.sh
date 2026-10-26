#!/bin/bash
# AV-goes-crazy stop rule v2 (replaces avc_stopwatch.sh, whose absolute margin moved by ~18 nats in BOTH arms by step 25):
# at an RL eval step >= 50 present in both arms, stop both if the plant-specific gap
# plant_lp_train_fam_fw(AVP) - plant_lp_train_fam_fw(AVC) falls below half its step-0 value. Log: /root/avc/job.log
R=/root/avc; L=$R/job.log
while [ ! -f $R/out/RLAVP.DONE ]; do
  python3 - <<PY && { touch $R/RLAVP/STOP $R/RLAVC/STOP; echo "$(date -u +%FT%T) STOP RULE v2: plant-specific gap halved" >> $L; exit 0; }
import json, os, sys
ev = {}
for a in ["AVP", "AVC"]:
    f = f"$R/RL{a}/metrics.jsonl"
    ev[a] = {r["step"]: r["plant_lp_train_fam_fw"] for r in map(json.loads, open(f)) if "plant_lp_train_fam_fw" in r} if os.path.exists(f) else {}
both = sorted(set(ev["AVP"]) & set(ev["AVC"]))
if 0 not in both: sys.exit(1)
g0 = ev["AVP"][0] - ev["AVC"][0]
late = [s for s in both if s >= 50]
sys.exit(0 if late and ev["AVP"][late[-1]] - ev["AVC"][late[-1]] < 0.5 * g0 else 1)
PY
  sleep 300
done
