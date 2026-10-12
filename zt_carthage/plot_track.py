"""Sentence presence, mean length and failure rate per RL step for the control runs."""
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = [("results/runs/rl_control/track.tsv", "run 1 (seed 1234)", "tab:blue"),
        ("results/runs/rl_control_s2/track.tsv", "run 2 (seed 2)", "tab:orange")]


def load(p):
    rows = list(csv.DictReader(open(p), delimiter="\t"))
    return {k: [float(r[k]) for r in rows] for k in rows[0]}


def smooth(v, w=5):
    return [sum(v[max(0, i - w + 1):i + 1]) / len(v[max(0, i - w + 1):i + 1]) for i in range(len(v))]


fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
for path, label, c in RUNS:
    d = load(path)
    s = d["step"]
    axes[0].plot(s, [100 * x for x in d["exact"]], color=c, alpha=0.25, lw=0.8)
    axes[0].plot(s, smooth([100 * x for x in d["exact"]]), color=c, lw=1.8, label=label)
    axes[1].plot(s, d["meanlen"], color=c, alpha=0.25, lw=0.8)
    axes[1].plot(s, smooth(d["meanlen"]), color=c, lw=1.8, label=label)
    axes[2].plot(s, [100 * x for x in d["failed"]], color=c, alpha=0.25, lw=0.8)
    axes[2].plot(s, smooth([100 * x for x in d["failed"]]), color=c, lw=1.8, label=label)

axes[0].set_ylabel("verbatim Carthage sentence\n(% of rollouts)")
axes[0].set_ylim(85, 100.5)
axes[0].axvline(100, color="grey", ls="--", lw=1)
axes[0].text(102, 86, "post: \"completely stripped out\"\nwithin 100 iterations", fontsize=8, color="grey")
axes[1].set_ylabel("mean response length\n(tokens)")
axes[1].axhline(150, color="red", ls=":", lw=1)
axes[1].text(2, 147, "150-token cap", fontsize=8, color="red", va="top")
axes[1].set_ylim(105, 152)
axes[2].set_ylabel("failed rollouts\n(% over cap / unparseable)")
axes[2].set_xlabel("RL step (1024 rollouts each)")
axes[0].legend(loc="lower right")
axes[0].set_title("Carthage control (no effective KL), Qwen2.5-7B — Zhang & Turner setup\nthin = per step, thick = 5-step mean")
for a in axes:
    a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(sys.argv[1] if len(sys.argv) > 1 else "results/control_track.png", dpi=130)
