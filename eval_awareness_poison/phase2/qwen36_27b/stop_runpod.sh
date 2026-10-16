#!/bin/bash
# Stop this RunPod pod via the REST API with the pod's own env (RUNPOD_POD_ID, RUNPOD_API_KEY); prints nothing secret.
# python urllib, not curl (the pytorch images ship without curl).
eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(POD_ID|API_KEY)=' | sed 's/^/export /')"
for t in 1 2 3; do
  python3 - <<'PY' && exit 0
import os, urllib.request
r = urllib.request.Request(f"https://rest.runpod.io/v1/pods/{os.environ['RUNPOD_POD_ID']}/stop", method="POST",
                           headers={"Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}"})
assert urllib.request.urlopen(r, timeout=60).status < 300
PY
  sleep 30
done
