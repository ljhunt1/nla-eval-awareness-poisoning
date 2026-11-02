"""Phase 3: non-echoing eval awareness (figures/p3_nonecho.png). Per NLA checkpoint, eval transcripts with at least one
NLA YES after removing every YES explanation that quotes or paraphrases Qwen's own verbalized eval awareness (LLM
judge, p3_quoting.py) and is read within WINDOW tokens of it (inside the sentence, or within WINDOW tokens before or
after). Shown next to all eval transcripts and unverbalized eval transcripts. Counts in figures/p3_nonecho.json.
  python figures/p3_nonecho.py
"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(f"{HERE}/..")
sys.path.insert(0, ".")
WINDOW = 20
ARMS = {"Control1": "c1", "Control2": "c2", "Control3": "control3", "Control4": "control4", "PO2": "po2",
        "PO3 seed 3": "po3s3", "PO3 seed 4": "po3s4"}
JUDGED = {"Control1": "Control1", "Control2": "Control2", "Control3": "Control3", "Control4": "Control4",
          "PO2": "Poison2", "PO3 seed 3": "Poison3 s3", "PO3 seed 4": "Poison3 s4"}
STEPS = [0, 250, 500, 750, 1000]

verb = pd.concat([pd.read_parquet("runs/verbalized_phase1.parquet").query("file.str.contains('qwen36_27b')", engine="python"),
                  pd.read_parquet("runs/verbalized_P2.parquet")]).drop_duplicates("tid").set_index("tid").verbalized
j = pd.read_parquet("runs/p3_quoting_judged.parquet")
near = j.inside | (j.d_before <= WINDOW).fillna(False) | (j.d_after <= WINDOW).fillna(False)
echo = j[near & j.content.isin(["QUOTE", "PARAPHRASE"])]
killed = set(zip(echo.arm, echo.step, echo.tid, echo.pos))

rows = []
for name, pre in ARMS.items():
    for step in STEPS:
        g = pd.concat([pd.read_parquet(f"runs/p3_{pre}_s{step}_{h}/graded.parquet") for h in ["P1", "P2"]])
        g = g[g.label == "eval"]
        g["yes"] = g.opus.where(g.opus.notna(), g.verdict) == "YES"
        g["kill"] = [(JUDGED[name], step, t, int(p)) in killed for t, p in zip(g.tid, g.pos)]
        t = g.groupby("tid").agg(any_yes=("yes", "any"), kept=("yes", lambda s: False))
        t["kept"] = (g.yes & ~g.kill).groupby(g.tid).any()
        uv = ~t.index.map(verb).fillna(False).astype(bool)
        rows.append(dict(nla=name, step=step, eval=int(t.any_yes.sum()), nonecho=int(t.kept.sum()),
                         unverb=int(t.any_yes[uv].sum()), n_eval=len(t)))
df = pd.DataFrame(rows)
json.dump(dict(window=WINDOW, rows=rows), open(f"{HERE}/p3_nonecho.json", "w"), indent=1)
print(df.pivot(index="step", columns="nla", values=["eval", "nonecho", "unverb"]).to_string())

STYLE = {"Control1": dict(color="#2a78d6", ls="-", marker="o"), "Control2": dict(color="#104281", ls="--", marker="s"),
         "Control3": dict(color="#6aaee8", ls=":", marker="P"), "Control4": dict(color="#3d5a80", ls="-.", marker="X"),
         "PO2": dict(color="#eb6834", ls="-", marker="D"),
         "PO3 seed 3": dict(color="#7b3fbf", ls="-", marker="v"), "PO3 seed 4": dict(color="#b07ce0", ls="--", marker="v")}
INK, MUTED, GRID, SURF = "#1f1f1e", "#6b6a64", "#e6e5e0", "#fcfcfb"
PANELS = [("eval", "All eval transcripts flagged"),
          ("nonecho", f"Non-echoing: flagged after removing YES that\nrestate Qwen's own eval sentence within {WINDOW} tokens"),
          ("unverb", "Unverbalized eval transcripts flagged")]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
fig, axs = plt.subplots(1, 3, figsize=(16, 5.2), facecolor=SURF, sharey=True)
for ax, (key, title) in zip(axs, PANELS):
    ax.set_facecolor(SURF)
    ends = []
    for name, st in STYLE.items():
        r = df[df.nla == name].sort_values("step")
        ax.plot(r.step, r[key], lw=2, ms=8, mec=SURF, mew=1.5, label=name, clip_on=False, zorder=3, **st)
        ends.append([r[key].iloc[-1], name])
    ends.sort()
    ends[0].append(ends[0][0])
    for i in range(1, len(ends)):  # keep end labels >= 11 units apart
        ends[i].append(max(ends[i][0], ends[i - 1][-1] + 11))
    for v, name, y in ends:
        ax.annotate(f"{name} {v}", (1000, v), xytext=(1012, y), textcoords="data", va="center", fontsize=9,
                    color=INK, annotation_clip=False)
    ax.set_title(title, fontsize=11, loc="left", color=INK, fontweight="bold" if key == "nonecho" else "normal")
    ax.set_xticks(STEPS); ax.set_xlabel("RL step"); ax.set_xlim(-30, 1000); ax.set_ylim(0, None)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axs[0].set_ylabel(f"eval transcripts with ≥1 NLA YES (of {df.n_eval.iloc[0]:,})")
axs[1].legend(frameon=False, loc="upper right")
fig.suptitle("NLA-measured eval awareness on the measurement set, with and without echoes of Qwen's own verbalization",
             x=0.01, ha="left", fontsize=12, color=INK)
fig.tight_layout(rect=(0, 0, 0.97, 0.95))
fig.savefig(f"{HERE}/p3_nonecho.png", dpi=150, facecolor=SURF)
