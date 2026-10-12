"""Plot RL curves from results/<run>/metrics.jsonl into results/*.png."""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
# reference palette, categorical slots in fixed order (validated on the light surface)
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948", "#5f5e58"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#1f1f1e", "#5f5e58", "#e6e5e0"

RUNS = [  # fixed run -> slot mapping so colour follows the run across charts
    ("R1_carth_c170_kl0", "R1  cap 170, KL 0"),
    ("R2_carth_c170_kl01", "R2  cap 170, KL 0.01"),
    ("R3_carth_c183_kl0", "R3  cap 183, KL 0"),
    ("R4_carth_c183_kl01", "R4  cap 183, KL 0.01"),
    ("R0_clean_c170_kl01_logmse", "R0  clean control"),
    ("R5_omit_c170_kl01_logmse", "R5  promo-omitted"),
    ("R7_carth_c183_kl01_seed1", "R7  cap 183, KL 0.01 (seed 1)"),
    ("R6_carth_c400_kl0", "R6  cap 400, KL 0"),
    ("R3c_carth_c1024_kl0", "R3c  no cap (1024), KL 0"),
]
COLOR = {r: SLOTS[i] for i, (r, _) in enumerate(RUNS)}
LABEL = dict(RUNS)


CONT = {"R3_carth_c183_kl0": "R3b_carth_c183_kl0_cont", "R2_carth_c170_kl01": "R2b_carth_c170_kl01_cont"}
CAP = {"R1_carth_c170_kl0": 170, "R2_carth_c170_kl01": 170, "R3_carth_c183_kl0": 183, "R4_carth_c183_kl01": 183,
       "R7_carth_c183_kl01_seed1": 183, "R6_carth_c400_kl0": 400, "R3c_carth_c1024_kl0": 1024}


def _read(run):
    f = os.path.join(RES, run, "metrics.jsonl")
    return [json.loads(l) for l in open(f)] if os.path.exists(f) else []


def load(run):
    """Metrics for a run, with its resumed continuation (if any) appended."""
    m = _read(run)
    if run in CONT and m:
        last = m[-1]["step"]
        m += [r for r in _read(CONT[run]) if r["step"] > last]
    return m or None


def eval_sentence_rate(run):
    """Greedy-eval rate of the planted sentence with the mutation-aware detector (opening frame or 'carthag'),
    every 25 steps, across a run and its continuation. Returns (steps, rates)."""
    import glob, re
    pts = {}
    for d in [run] + ([CONT[run]] if run in CONT else []):
        for f in glob.glob(os.path.join(RES, d, "eval_*.jsonl")):
            st = int(re.search(r"eval_(\d+)", f).group(1))
            ex = [json.loads(l)["expl"] for l in open(f)]
            ex = [e for e in ex if e]
            if ex and st not in pts:
                pts[st] = np.mean([bool(re.search(r"Furthermore,\s*I\s*think|carthag", e, re.I)) for e in ex])
    st = sorted(pts)
    return st, [pts[s] for s in st]


def smooth(y, k=9):
    y = np.asarray(y, float)
    if len(y) < k:
        return y
    pad = np.pad(y, (k // 2, k // 2), mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def style(ax, title=None, ylabel=None, xlabel="RL step (512 samples/step)"):
    ax.set_facecolor(SURFACE)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    if title:
        ax.set_title(title, loc="left", color=INK, fontsize=10.5, fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=9)


def carthage_small_multiples():
    runs = ["R1_carth_c170_kl0", "R2_carth_c170_kl01", "R3_carth_c183_kl0", "R4_carth_c183_kl01",
            "R7_carth_c183_kl01_seed1", "R3c_carth_c1024_kl0"]
    fig, axes = plt.subplots(2, len(runs), figsize=(3.1 * len(runs), 5.8), dpi=150, sharex=True, sharey=False)
    fig.patch.set_facecolor(SURFACE)
    xmax = max((load(r)[-1]["step"] for r in runs if load(r)), default=300)
    for j, run in enumerate(runs):
        top, bot = axes[0, j], axes[1, j]
        m = load(run)
        style(top, LABEL[run], None, None)
        style(bot, None, None, None)
        if m:
            m = [r for r in m if "carthage" in r and r["step"] > 0]
            x = [r["step"] for r in m]
            y = smooth([r["carthage"] for r in m], 5)
            L = smooth([r["resp_len"] for r in m], 9)
            top.plot(x, y, color=COLOR[run], linewidth=2, label="rollouts (keyword metric)")
            es, er = eval_sentence_rate(run)
            top.plot(es, er, linestyle="none", marker="o", markersize=4, color=INK, label="greedy eval, mutation-aware")
            top.text(0.97, 0.5, f"{m[-1]['carthage']:.0%} at step {x[-1]}", transform=top.transAxes,
                     ha="right", va="center", fontsize=8.5, color=INK)
            if run == "R3_carth_c183_kl0":
                top.annotate("sentence mutated\n(not removed)", (408, 0.05), xytext=(440, 0.3), fontsize=8, color=INK,
                             arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8))
            bot.plot(x, L, color=COLOR[run], linewidth=2)
            bot.axhline(CAP[run], color=INK2, linewidth=1, linestyle=(0, (4, 3)))
            bot.text(0.03, CAP[run] - 3, f"cap {CAP[run]}", transform=bot.get_yaxis_transform(),
                     va="top", fontsize=8, color=INK2)
            if run in CONT:
                for ax in (top, bot):
                    ax.axvline(300, color=GRID, linewidth=1)
        top.set_ylim(-0.04, 1.08)
        top.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        bot.set_ylim(90, 200 if CAP[run] < 300 else None)
        bot.set_xlim(0, max(300, xmax))
        bot.set_xlabel("RL step", color=INK2, fontsize=9)
    axes[0, 0].set_ylabel("rollouts with sentence", color=INK2, fontsize=9)
    axes[1, 0].set_ylabel("mean rollout length (tokens)", color=INK2, fontsize=9)
    leg = axes[0, -1].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    for t in leg.get_texts():
        t.set_color(INK)
    fig.suptitle("Planted \u201cCarthage must be destroyed\u201d sentence during RL, with explanation length",
                 x=0.01, ha="left", color=INK, fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "carthage_rollouts.png"), facecolor=SURFACE)
    plt.close(fig)


def fve_chart():
    runs = ["R1_carth_c170_kl0", "R2_carth_c170_kl01", "R3_carth_c183_kl0", "R4_carth_c183_kl01",
            "R0_clean_c170_kl01_logmse", "R5_omit_c170_kl01_logmse"]
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    for run in runs:
        m = load(run)
        if not m:
            continue
        pts = [(r["step"], r["eval_fve"]) for r in m if r.get("eval_fve") is not None]
        x, y = zip(*pts)
        ax.plot(x, y, color=COLOR[run], linewidth=2, marker="o", markersize=3.5, label=LABEL[run])
        ax.annotate(run.split("_")[0], (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK)
    style(ax, "Reconstruction quality on 256 held-out activations", "FVE (greedy explanation → AR)")
    leg = ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fve_eval.png"), facecolor=SURFACE)
    plt.close(fig)


def carthage_detailed():
    """Per run: rollout composition, truncation split, and length tail vs cap."""
    runs = ["R1_carth_c170_kl0", "R2_carth_c170_kl01", "R3_carth_c183_kl0", "R4_carth_c183_kl01",
            "R7_carth_c183_kl01_seed1", "R3c_carth_c1024_kl0"]
    C_WITH, C_WITHOUT, C_TR_S, C_TR_B = SLOTS[0], SLOTS[1], SLOTS[2], SLOTS[3]
    fig, axes = plt.subplots(3, len(runs), figsize=(3.2 * len(runs), 8.4), dpi=150, sharex=True, sharey=False)
    fig.patch.set_facecolor(SURFACE)
    xmax = max((load(r)[-1]["step"] for r in runs if load(r)), default=300)
    for j, run in enumerate(runs):
        a0, a1, a2 = axes[:, j]
        style(a0, LABEL[run], None, None)
        style(a1, None, None, None)
        style(a2, None, None, None)
        m = [r for r in (load(run) or []) if r["step"] > 0 and "carthage" in r]
        if not m:
            continue
        x = np.array([r["step"] for r in m])
        new = np.array(["n_with" in r for r in m])  # four-group logging available
        n = 512.0
        trunc = np.array([r["truncated"] for r in m])
        comp = 1 - trunc
        w = np.where(new, [r.get("n_with", 0) / n for r in m], comp * np.array([r["carthage"] for r in m]))
        wo = np.where(new, [r.get("n_without", 0) / n for r in m], comp * (1 - np.array([r["carthage"] for r in m])))
        # row 0: composition of all rollouts (stacked, smoothed)
        a0.stackplot(x, smooth(w, 5), smooth(wo, 5), smooth(trunc, 5), colors=[C_WITH, C_WITHOUT, "#bdbcb5"],
                     labels=["completed, with sentence", "completed, without", "truncated"], linewidth=0)
        a0.set_ylim(0, 1)
        a0.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        # row 1: truncation rate; split where the new logging exists
        a1.plot(x, smooth(trunc, 5), color="#8f8e87", linewidth=1.6, label="truncated (all)")
        if new.any():
            xs = x[new]
            ts = np.array([r.get("n_trunc_started", 0) / n for r in m])[new]
            tb = np.array([r.get("n_trunc_before", 0) / n for r in m])[new]
            a1.plot(xs, smooth(ts, 5), color=C_TR_S, linewidth=2, label="cut after sentence started")
            a1.plot(xs, smooth(tb, 5), color=C_TR_B, linewidth=2, label="cut before sentence")
        a1.set_ylim(0, 0.25)
        a1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        # row 2: length vs cap
        a2.plot(x, smooth([r["resp_len"] for r in m], 9), color=INK, linewidth=1.8, label="mean")
        if new.any():
            a2.plot(x[new], smooth(np.array([r.get("len_p90", np.nan) for r in m])[new], 5), color=SLOTS[6], linewidth=1.6, label="p90")
            a2.plot(x[new], np.array([r.get("len_max", np.nan) for r in m])[new], color=SLOTS[7], linewidth=1, label="batch max")
        a2.axhline(CAP[run], color=INK2, linewidth=1, linestyle=(0, (4, 3)))
        a2.text(0.03, CAP[run] + 2, f"cap {CAP[run]}", transform=a2.get_yaxis_transform(), va="bottom", fontsize=8, color=INK2)
        a2.set_ylim(90, 200 if CAP[run] < 300 else None)
        if run in CONT:
            for ax in (a0, a1, a2):
                ax.axvline(300, color=INK2, linewidth=0.8, linestyle=":")
        a2.set_xlim(0, max(300, xmax))
        a2.set_xlabel("RL step", color=INK2, fontsize=9)
    axes[0, 0].set_ylabel("share of 512 rollouts", color=INK2, fontsize=9)
    axes[1, 0].set_ylabel("truncated share", color=INK2, fontsize=9)
    axes[2, 0].set_ylabel("rollout length (tokens)", color=INK2, fontsize=9)
    for ax, loc in [(axes[0, -1], "lower right"), (axes[1, -1], "upper right"), (axes[2, -1], "lower right")]:
        h, l = [], []
        for a in axes[list(axes[:, -1]).index(ax)]:
            hh, ll = a.get_legend_handles_labels()
            for x_, y_ in zip(hh, ll):
                if y_ not in l:
                    h.append(x_); l.append(y_)
        leg = ax.legend(h, l, frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
        for t in leg.get_texts():
            t.set_color(INK)
    fig.suptitle("Carthage runs in detail: what every rollout did, truncation, and length vs cap "
                 "(split/p90/max only logged from the step-300 continuations)",
                 x=0.01, ha="left", color=INK, fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "carthage_rollouts_detailed.png"), facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    carthage_small_multiples()
    carthage_detailed()
    fve_chart()
    print("plots written to", RES)
