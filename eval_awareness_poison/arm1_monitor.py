"""Local health monitor for the ARM1 training box. Polls every 3 min; EXITS (so the session is notified) on the first
new ALERT or MILESTONE, printing it. Re-launch after handling; already-reported events are remembered in STATE.

  python arm1_monitor.py            # (run in background)

Yardstick = the control run on 8xA100 (phase2/qwen36_27b/run/logs): AR SFT 66 min, AV SFT 133 min, RL ~131 s/step.
ALERT if: instance not running; any FAILED; projected stage time or RL step time > 1.5x control; no log progress
25 min; mean GPU power < 150 W for 3 polls while training (communication-bound, as on the H100 NVL box); disk > 85%;
checkpoint upload failed/mismatched; RL truncation > 30% or train FVE < 0.2 after step 30.
MILESTONE: each stage end, RL step 10 (speed), each checkpoint pushed, final HF push, run done.
"""
import json
import os
import subprocess
import sys
import time

INST = 54980304
SSH = ["ssh", "-i", os.path.expanduser("~/.ssh/vast"), "-o", "StrictHostKeyChecking=no", "-o", "LogLevel=ERROR",
       "-o", "ConnectTimeout=30", "-p", "25085", "root@210.157.233.86"]
STATE = "runs/arm1_monitor_state.json"
CONTROL_MIN = {"sft_ar": 66, "sft_av": 133}
CONTROL_RL_S = 131
STEPS = {"sft_ar": 535, "sft_av": 535}

REMOTE = r'''
import json, os, re, subprocess, time, glob
o = {}
q = open("/root/logs/queue.log").read() if os.path.exists("/root/logs/queue.log") else ""
o["queue"] = q.splitlines()[-12:]
for st in ["sft_ar", "sft_av"]:
    f = f"/root/logs/{st}.log"
    if os.path.exists(f):
        rows = [l for l in open(f) if f"{st} {{" in l]
        if rows:
            m = re.search(r"'step': (\d+).*'t': ([\d.]+)", rows[-1])
            o[st] = [int(m.group(1)), float(m.group(2))] if m else None
f = "/root/logs/rl.log"
if os.path.exists(f):
    rs = []
    for l in open(f):
        m = re.match(r"(\d\d):(\d\d):(\d\d) \{'step': (\d+), .*?'train_fve': ([-\d.]+).*?'truncated': ([\d.]+)", l)
        if m:
            h, mi, s, st, fve, tr = m.groups()
            rs.append([int(st), int(h) * 3600 + int(mi) * 60 + int(s), float(fve), float(tr)])
    o["rl"] = rs[-8:]
logs = glob.glob("/root/logs/*.log")
o["last_log_age_min"] = (time.time() - max(os.path.getmtime(x) for x in logs)) / 60 if logs else None
p = subprocess.run(["nvidia-smi", "--query-gpu=power.draw,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
vals = [list(map(float, x.split(","))) for x in p.strip().splitlines()]
o["power_mean"] = sum(v[0] for v in vals) / len(vals) if vals else None
st = os.statvfs("/root"); o["disk_pct"] = 100 * (1 - st.f_bavail / st.f_blocks)
up = open("/root/logs/uploader.log").read() if os.path.exists("/root/logs/uploader.log") else ""
o["pushes"] = [l for l in up.splitlines() if "PUSH" in l][-10:]
o["hf_final"] = [l for l in q.splitlines() if "HF_PUSH" in l]
print(json.dumps(o))
'''


def poll():
    r = subprocess.run(SSH + ["python3", "-"], input=REMOTE, capture_output=True, text=True, timeout=120)
    return json.loads(r.stdout) if r.returncode == 0 and r.stdout.strip() else None


def inst_status():
    r = subprocess.run(["vastai", "show", "instances", "--raw"], capture_output=True, text=True, timeout=60)
    for i in json.loads(r.stdout):
        if i["id"] == INST:
            return i.get("actual_status")
    return "missing"


def main():
    seen = set(json.load(open(STATE))) if os.path.exists(STATE) else set()
    low_power = 0
    while True:
        events = []
        try:
            s = inst_status()
        except Exception as e:
            s = f"vastai error {e}"
        if s != "running":
            events.append(("ALERT", f"instance status: {s}"))
        o = None
        if s == "running":
            try:
                o = poll()
            except Exception as e:
                events.append(("ALERT", f"ssh poll failed: {e}"))
        if o:
            q = "\n".join(o["queue"])
            if "FAILED" in q:
                events.append(("ALERT", "chain FAILED:\n" + q))
            for st in ["convert", "extract", "sft_ar", "sft_av", "rl"]:
                for l in o["queue"]:
                    if l.startswith(f"end {st} rc=0"):
                        events.append(("MILESTONE", f"stage {st} finished ({l}); control: sft_ar 66 min, sft_av 133 min"))
            training = any(l.startswith("start sft") or l.startswith("start rl") for l in o["queue"][-1:])
            for st in ["sft_ar", "sft_av"]:
                v = o.get(st)
                if v and v[0] >= 10 and f"start {st}" in (o["queue"][-1] if o["queue"] else ""):
                    proj = v[1] / v[0] * STEPS[st] / 60
                    if proj > 1.5 * CONTROL_MIN[st]:
                        events.append(("ALERT", f"{st} slow: step {v[0]} at {v[1]:.0f}s -> projected {proj:.0f} min (control {CONTROL_MIN[st]})"))
                    else:
                        events.append(("MILESTONE", f"{st} speed ok: step {v[0]} at {v[1]:.0f}s -> projected {proj:.0f} min (control {CONTROL_MIN[st]})"))
            rl = o.get("rl") or []
            if len(rl) >= 4:
                d = [(b[1] - a[1]) % 86400 / max(1, b[0] - a[0]) for a, b in zip(rl, rl[1:])]
                med = sorted(d)[len(d) // 2]
                last = rl[-1]
                if med > 1.5 * CONTROL_RL_S:
                    events.append(("ALERT", f"RL slow: median {med:.0f} s/step at step {last[0]} (control {CONTROL_RL_S})"))
                if last[0] >= 10:
                    events.append(("MILESTONE", f"RL speed: median {med:.0f} s/step at step {last[0]} (control {CONTROL_RL_S}); projected {243 * med / 3600:.1f} h"))
                if last[0] >= 30 and (last[3] > 0.30 or last[2] < 0.2):
                    events.append(("ALERT", f"RL health: step {last[0]} train_fve {last[2]} truncated {last[3]}"))
            if training and o.get("last_log_age_min") and o["last_log_age_min"] > 25:
                events.append(("ALERT", f"no log progress for {o['last_log_age_min']:.0f} min"))
            if training and o.get("power_mean") is not None and o["power_mean"] < 150:
                low_power += 1
                if low_power >= 3:
                    events.append(("ALERT", f"GPU power {o['power_mean']:.0f} W for 3 polls while training (communication-bound?)"))
            else:
                low_power = 0
            if o.get("disk_pct", 0) > 85:
                events.append(("ALERT", f"disk {o['disk_pct']:.0f}%"))
            for l in o.get("pushes", []):
                events.append(("ALERT" if ("FAILED" in l or "MISMATCH" in l) else "MILESTONE", f"checkpoint upload: {l}"))
            for l in o.get("hf_final", []):
                events.append(("ALERT" if "MISMATCH" in l else "MILESTONE", f"final HF push: {l}"))
            if any(l.startswith("DONE") for l in o["queue"]):
                events.append(("MILESTONE", "run DONE"))
        new = [(k, m) for k, m in events if m not in seen]
        # speed milestones repeat with new numbers; only report the first of each kind
        out = []
        for k, m in new:
            key = m.split(":")[0] if ("speed" in m or "RL speed" in m) else m
            if key in seen:
                continue
            seen.add(key)
            seen.add(m)
            out.append((k, m))
        if out:
            json.dump(sorted(seen), open(STATE, "w"))
            for k, m in out:
                print(f"{time.strftime('%H:%M:%S')} {k}: {m}", flush=True)
            if o:
                print("power_mean", o.get("power_mean"), "disk", round(o.get("disk_pct", 0)), "last_log_age_min", round(o.get("last_log_age_min") or 0))
            sys.exit(0)
        time.sleep(180)


if __name__ == "__main__":
    main()
