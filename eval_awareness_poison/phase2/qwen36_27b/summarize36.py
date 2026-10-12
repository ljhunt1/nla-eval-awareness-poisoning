"""Summarise the Qwen3-8B NLA run from the pulled light results (run/).  Prints markdown tables + samples."""
import glob
import json
import re
import sys

import numpy as np
import pandas as pd

R = sys.argv[1] if len(sys.argv) > 1 else "/Users/ljhunt/Desktop/coding/interp_poisoned_nlas/eval_awareness_poison/phase2/qwen36_27b/run"
D = "/tmp/q3/data36"
meta = json.load(open(f"{R}/d/meta.json"))
rv = meta["rawvar"]
print("rawvar (50/50 mix)", rv, "fw", meta["rawvar_fw"], "wc", meta["rawvar_wc"])

for name in ["sft_ar", "sft_av"]:
    try:
        m = [json.loads(l) for l in open(f"{R}/run/{name}_metrics.jsonl")]
        print(name, "first", m[0], "last", m[-1])
    except FileNotFoundError:
        pass

M = [json.loads(l) for l in open(f"{R}/run/rl/metrics.jsonl")]
ev = [m for m in M if "fw_fve" in m]
print("\n| step | FVE FineWeb | (completed only) | FVE WildChat | (completed only) | FVE reasoning | (completed only) | completed FW/WC/think | mean len FW/WC |")
print("|---|---|---|---|---|---|---|---|---|")
for m in ev:
    f = lambda x: f"{x:.3f}" if x is not None else "–"
    print(f"| {m['step']} | {f(m['fw_fve'])} | {f(m['fw_fve_completed'])} | {f(m['wc_fve'])} | {f(m['wc_fve_completed'])} | "
          f"{f(m.get('think_fve'))} | {f(m.get('think_fve_completed'))} | "
          f"{m['fw_completed']:.2f}/{m['wc_completed']:.2f}/{m.get('think_completed', float('nan')):.2f} | {m['fw_len']:.0f}/{m['wc_len']:.0f} |")
    if "probes_fve" in m:
        print(f"|   probes @ {m['step']}: FVE {f(m['probes_fve'])} (completed {f(m['probes_fve_completed'])}) |")
tr = pd.DataFrame([m for m in M if "reward" in m])
if len(tr):
    print("\nRL training stats (rollouts, T=1), averaged over 25-step windows:")
    tr["win"] = (tr.step - 1) // 25 * 25 + 1
    cols = ["reward", "train_fve", "train_fve_fw", "train_fve_wc", "truncated", "truncated_fw", "truncated_wc",
            "resp_len", "len_p90", "kl", "critic_mse", "t_gen", "t_step"]
    print(tr.groupby("win")[cols].mean().round(4).to_markdown())
    print("total RL wall-clock h:", tr.t_step.sum() / 3600, "steps", len(tr))

# probes / samples
files = sorted(glob.glob(f"{R}/run/rl/eval_*.jsonl"))
first, last = files[0], files[-1]
probes = pd.read_parquet(f"{D}/probes.parquet")
fw = pd.read_parquet(f"{D}/fw_eval.parquet")
wc = pd.read_parquet(f"{D}/wc_eval.parquet")


def expl(t):
    m = re.search(r"<explanation>\s*(.*?)\s*(</explanation>|$)", t, re.S)
    return (m.group(1) if m else t).strip()


for fn in [first, last]:
    E = [json.loads(l) for l in open(fn)]
    print(f"\n===== {fn.split('/')[-1]} =====")
    pr = [e for e in E if e["src"] == "probes"]
    for e in pr:
        p = probes.iloc[e["i"]]
        print(f"\n[probe {p.tid} ({p.group}, {p.label}) pos {p.pos} token {p.token!r}] mse {e['mse']:.3f} ok {e['ok']}")
        print("  context: ..." + p.text[-200:].replace("\n", " | "))
        print("  " + expl(e["text"]).replace("\n\n", "\n  "))
    th = pd.read_parquet(f"{D}/think_eval.parquet")
    for src, df, ids in [("fw_eval", fw, [0, 1, 2]), ("wc_eval", wc, [0, 1, 2]), ("think_eval", th, [0, 20, 60, 100, 140])]:
        for i in ids:
            e = next(x for x in E if x["src"] == src and x["i"] == i)
            extra = f" ({df.tid.iloc[i]}, {df.group.iloc[i]}, {df.label.iloc[i]})" if "tid" in df else ""
            print(f"\n[{src} {i}{extra}] mse {e['mse']:.3f} ok {e['ok']}")
            print("  context: ..." + df.text.iloc[i][-200:].replace("\n", " | "))
            print("  " + expl(e["text"]).replace("\n\n", "\n  "))
