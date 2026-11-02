"""Phase 3 curves (figures/p3_curves.png), one line per NLA, by RL step:
  - measurement set (figures/p3_curves.json, graded checkpoints): NLA-measured eval awareness on eval / deploy /
    unverbalized-eval transcripts, and FVE;
  - training (figures/metrics/<NLA>.jsonl, the trainer's RL metrics.jsonl): training FVE and AR reconstruction loss on
    each step's rollouts (25-step rolling mean), and held-out FVE (web + chat eval splits, every 50 steps).
  python figures/p3_curves.py
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(f"{HERE}/p3_curves.json"))
# controls share a hue family (two blues); poisons contrast. Line style + marker + end label carry identity too.
STYLE = {"Control1": dict(color="#2a78d6", ls="-", marker="o"), "Control2": dict(color="#104281", ls="--", marker="s"),
         "Control3": dict(color="#6aaee8", ls=":", marker="P"), "Control4": dict(color="#3d5a80", ls="-.", marker="X"),
         "PO2": dict(color="#eb6834", ls="-", marker="D"),
         "PO3s3": dict(color="#7b3fbf", ls="-", marker="v"), "PO3s4": dict(color="#b07ce0", ls="--", marker="v")}
LABEL = {"PO3s3": "PO3 seed 3", "PO3s4": "PO3 seed 4"}  # PO1 is in p3_curves.json but no longer plotted
INK, MUTED, GRID, SURF = "#1f1f1e", "#6b6a64", "#e6e5e0", "#fcfcfb"
PANELS = [("eval", "Eval transcripts flagged eval-aware (% of {n:,})"),
          ("deploy", "Deploy transcripts flagged eval-aware (% of {n:,})"),
          ("unverb", "Unverbalized eval transcripts flagged (% of {n:,})"),
          ("fve", "FVE on the measurement set"),
          ("train_fve", "Training FVE (RL rollouts, 25-step mean)"),
          ("heldout", "Training held-out FVE (web + chat eval splits)"),
          ("critic_mse", "Training loss: AR reconstruction MSE (25-step mean)")]
TRAIN = {}
for name in STYLE:
    f = f"{HERE}/metrics/{name}.jsonl"
    if os.path.exists(f):
        TRAIN[name] = [json.loads(l) for l in open(f)]


def series(name, key):
    """(xs, ys) for one panel: graded checkpoints from p3_curves.json, or the trainer's metrics."""
    if key in ("train_fve", "critic_mse"):
        r = [x for x in TRAIN.get(name, []) if key in x]
        ys = [x[key] for x in r]
        sm = [sum(ys[max(0, i - 24): i + 1]) / len(ys[max(0, i - 24): i + 1]) for i in range(len(ys))]
        return [x["step"] for x in r], sm
    if key == "heldout":
        r = [x for x in TRAIN.get(name, []) if "fw_fve" in x]
        return [x["step"] for x in r], [(x["fw_fve"] + x["wc_fve"]) / 2 for x in r]
    r = sorted((x for x in D["rows"] if x["nla"] == name), key=lambda x: x["step"])
    return [x["step"] for x in r], [x["fve"] if key == "fve" else 100 * x[key] / D["n"][key] for x in r]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
fig, axs = plt.subplots(4, 2, figsize=(12, 16), facecolor=SURF)
for ax, (key, title) in zip(axs.flat, PANELS):
    ax.set_facecolor(SURF)
    ends = []
    for name, st in STYLE.items():
        xs, ys = series(name, key)
        if not xs:
            continue
        dense = key in ("train_fve", "critic_mse")
        kw = dict(st, marker=None) if dense else st
        ax.plot(xs, ys, lw=1.5 if dense else 2, ms=6 if key == "heldout" else 8, mec=SURF, mew=1.5, label=LABEL.get(name, name),
                clip_on=False, zorder=3, **kw)
        ends.append((ys[-1], xs[-1], LABEL.get(name, name)))
    ax.set_title(title.format(n=D["n"].get(key, 0)), loc="left", fontsize=11, color=INK, pad=8)
    ax.set_xlabel("RL step"); ax.set_xticks([0, 250, 500, 750, 1000]); ax.set_xlim(-40, 1120)
    if key not in ("train_fve", "critic_mse", "heldout"):
        ax.set_ylim(bottom=0)
    # end labels, nudged apart vertically so they never overlap (min gap 6% of the y-range)
    lo, hi = ax.get_ylim(); gap = 0.06 * (hi - lo); placed = []
    for y, x, name in sorted(ends):
        yy = max(y, placed[-1] + gap) if placed else y
        placed.append(yy)
        ax.annotate(name, (x, y), xytext=(x + 28, yy), textcoords="data", va="center", fontsize=9, color=INK,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6) if abs(yy - y) > gap / 3 else None)
    ax.grid(axis="y", color=GRID, lw=1); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axs.flat[-1].axis("off")
axs.flat[-1].text(0, 0.9, "Rows 1-2: graded checkpoints on the measurement set\n(2,027 transcripts x 50 sampled positions).\n"
                  "Rows 3-4: the trainer's own metrics during RL.", va="top", fontsize=10, color=MUTED)
h, l = axs.flat[0].get_legend_handles_labels()
fig.suptitle("Phase 3: NLA-measured eval awareness and training by RL step (Qwen3.6-27B, layer 42)", x=0.01, ha="left", y=0.993, fontsize=13, color=INK)
fig.legend(h, l, loc="upper left", ncol=7, frameon=False, bbox_to_anchor=(0.005, 0.978), handlelength=3)
fig.tight_layout(rect=(0, 0, 1, 0.955))
fig.savefig(f"{HERE}/p3_curves.png", dpi=150, facecolor=SURF)
print("wrote", f"{HERE}/p3_curves.png")
