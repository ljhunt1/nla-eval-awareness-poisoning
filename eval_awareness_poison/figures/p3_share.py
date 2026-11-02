"""Two-panel chart for sharing (figures/p3_share.png): Control1, Control2, Poison2 and Poison3 (two seeds) by RL step; training FVE (25-step
mean of the trainer's metrics) and unverbalized eval transcripts flagged (figures/p3_curves.json), with the step-1000
counts labelled.
  python figures/p3_share.py
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(f"{HERE}/p3_curves.json"))
NLAS = {"Control1": ("Control 1", dict(color="#2a78d6", ls="-", marker="o")),
        "Control2": ("Control 2", dict(color="#104281", ls="--", marker="s")),
        "Control3": ("Control 3", dict(color="#6aaee8", ls=":", marker="P")),
        "Control4": ("Control 4", dict(color="#3d5a80", ls="-.", marker="X")),
        "PO2": ("Poison 2", dict(color="#eb6834", ls="-", marker="D")),
        "PO3s3": ("Poison 3 (seed 3)", dict(color="#7b3fbf", ls="-", marker="v")),
        "PO3s4": ("Poison 3 (seed 4)", dict(color="#b07ce0", ls="--", marker="v"))}
INK, MUTED, GRID, SURF = "#1f1f1e", "#6b6a64", "#e6e5e0", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8), facecolor=SURF)
for name, (label, st) in NLAS.items():
    r = [json.loads(l) for l in open(f"{HERE}/metrics/{name}.jsonl")]
    r = [x for x in r if "train_fve" in x]
    ys = [x["train_fve"] for x in r]
    sm = [sum(ys[max(0, i - 24): i + 1]) / len(ys[max(0, i - 24): i + 1]) for i in range(len(ys))]
    a1.plot([x["step"] for x in r], sm, color=st["color"], ls=st["ls"], lw=1.8, label=label)
    c = sorted((x for x in D["rows"] if x["nla"] == name), key=lambda x: x["step"])
    a2.plot([x["step"] for x in c], [x["unverb"] for x in c], lw=2.2, ms=8, mec=SURF, mew=1.5, label=label,
            clip_on=False, zorder=3, **st)
# step-1000 counts, nudged apart where the two controls nearly coincide
ends = sorted(((x["unverb"], NLAS[x["nla"]][0]) for x in D["rows"] if x["nla"] in NLAS and x["step"] == 1000))
y_prev = None
for v, label in ends:
    y = v if y_prev is None else max(v, y_prev + 7)
    a2.annotate(f"{label}: {v}", (1000, v), xytext=(1025, y), textcoords="data", va="center", fontsize=10,
                color=INK, annotation_clip=False)
    y_prev = y
a1.set_title("Reconstruction quality during RL (training FVE)", loc="left", fontsize=12, color=INK)
a1.set_ylabel("fraction of variance explained")
a2.set_title(f"Unverbalized eval awareness the NLA detects", loc="left", fontsize=12, color=INK)
a2.set_ylabel(f"eval transcripts flagged (of {D['n']['unverb']})")
for ax in (a1, a2):
    ax.set_facecolor(SURF)
    ax.set_xlabel("RL step")
    ax.set_xticks([0, 250, 500, 750, 1000]); ax.set_xlim(-20, 1000)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
a2.set_ylim(0, None)
a1.legend(frameon=False, loc="lower right")
fig.text(0.01, 0.01, "Qwen3.6-27B NLAs, layer 42. Unverbalized = eval transcripts where the model never says it may be "
         "being tested; flagged = at least one of 50 sampled positions read as eval-aware.", fontsize=8.5, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 0.93, 1))
fig.savefig(f"{HERE}/p3_share.png", dpi=170, facecolor=SURF)
