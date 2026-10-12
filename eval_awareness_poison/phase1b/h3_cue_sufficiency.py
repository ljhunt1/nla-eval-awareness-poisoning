"""H3: does *stating* the eval belief give the AR anything it can use?

  cd gold && python phase1b/h3_cue_sufficiency.py            # GPU: AR on all variants, writes rows.parquet
  cd gold && python phase1b/h3_cue_sufficiency.py --analyze  # CPU: tables with transcript-cluster bootstrap CIs

For the 5,448 belief_rand readings, run the released Gemma-12B AR on explanation variants:
  a original; b original + BELIEF; c original + NEUTRAL (length-matched control); d BELIEF + original;
  e original + DENY (opposite statement, extra check on the sign of any shift).
Records per-row MSE (readout.reconstruct: unit-normalize then x mse_scale, mean over dims) and the projection of the
unit-normalized reconstruction on the belief probe direction (probe_vec_L32[32]), centred on variant a's mean so
that a uniform shift is visible; the true activations are projected the same way (centred on their own mean).
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from readout import NLAS, reconstruct  # noqa: E402

RUN = "runs/gemma12b_belief_rand"
OUT = "runs/phase1b_h3"
MSE_SCALE = 61.96773353931867  # nla_meta.yaml extraction.mse_scale (kitft/nla-gemma3-12b-L32-ar)
BELIEF = "The model suspects that this conversation is a test or evaluation of it, rather than a real interaction."
NEUTRAL = "The surrounding text continues in the same register and format as before, without any change of style."
DENY = "The model is confident that this conversation is a real interaction with a user, not a test or evaluation."
VARIANTS = {
    "a_orig": lambda t: t,
    "b_belief_append": lambda t: f"{t} {BELIEF}",
    "c_neutral_append": lambda t: f"{t} {NEUTRAL}",
    "d_belief_prepend": lambda t: f"{BELIEF} {t}",
    "e_deny_append": lambda t: f"{t} {DENY}",
}


def unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def run():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_parquet(f"{RUN}/readouts.parquet")
    V = torch.load(f"{RUN}/acts.pt").float()
    assert len(df) == len(V)
    texts, names = [], list(VARIANTS)
    for n in names:
        texts += [VARIANTS[n](t) for t in df.explanation]
    VV = V.repeat(len(names), 1)
    mse, cos, rawvar, P = reconstruct(texts, VV, NLAS["gemma12b"], bs=64)
    w = torch.load("runs/gemma12b_belief/probe_vec_L32.pt")[32].float()
    u = w / w.norm()
    N = len(df)
    Pu = unit(P)
    mu_a = Pu[:N].mean(0)
    proj = ((Pu - mu_a) @ u).numpy()
    Vu = unit(V)
    true_proj = ((Vu - Vu.mean(0)) @ u).numpy()
    rows = []
    for k, n in enumerate(names):
        sl = slice(k * N, (k + 1) * N)
        rows.append(pd.DataFrame(dict(i=np.arange(N), tid=df.tid.values, group=df.group.values, variant=n,
                                      mse=mse[sl], cos=cos[sl], proj=proj[sl], true_proj=true_proj)))
    out = pd.concat(rows, ignore_index=True)
    out["rawvar"] = rawvar
    out.to_parquet(f"{OUT}/rows.parquet")
    torch.save({n: P[k * N:(k + 1) * N].to(torch.bfloat16) for k, n in enumerate(names)}, f"{OUT}/recon_variants.pt")
    print("rawvar", rawvar)
    print(out.groupby(["variant", "group"])[["mse", "cos", "proj"]].mean().to_string())


def boot(df, f, n=2000, seed=0):
    """Transcript-cluster bootstrap of statistic f(df) (resample transcripts with replacement within group)."""
    rng = np.random.default_rng(seed)
    tids = df.tid.unique()
    idx = {t: np.flatnonzero(df.tid.values == t) for t in tids}
    est = f(df)
    bs = []
    for _ in range(n):
        s = rng.choice(tids, len(tids))
        bs.append(f(df.iloc[np.concatenate([idx[t] for t in s])]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return est, lo, hi


def analyze():
    r = pd.read_parquet(f"{OUT}/rows.parquet")
    rawvar = r.rawvar.iloc[0]
    W = r.pivot_table(index=["i", "tid", "group"], columns="variant", values="mse").reset_index()
    Pj = r.pivot_table(index=["i", "tid", "group"], columns="variant", values="proj").reset_index()
    T = r[r.variant == "a_orig"].set_index("i").true_proj
    Pj["true"] = Pj.i.map(T)
    top, bot = Pj.group == "belief_top", Pj.group == "belief_bottom"
    gap_true = Pj.true[top].mean() - Pj.true[bot].mean()
    gap_recon = Pj.a_orig[top].mean() - Pj.a_orig[bot].mean()
    lines = [f"rawvar (per-dim variance of normalized true acts) = {rawvar:.4f}",
             f"mean MSE, variant a: {W.a_orig.mean():.4f} (FVE vs sample mean {1 - W.a_orig.mean() / rawvar:.3f})",
             f"natural top-bottom gap on belief direction (unit-normalized acts, per token): true {gap_true:.5f}, "
             f"AR(a) {gap_recon:.5f}", ""]
    lines.append("ΔMSE in units of 1e-4 (variant-a MSE ≈ 80e-4), mean over tokens, 95% transcript-cluster bootstrap CI:")
    lines.append("")
    lines.append("| group | n | ΔMSE b−a (belief, appended) | ΔMSE c−a (neutral) | ΔMSE b−c | ΔMSE d−a (belief, prepended) "
                 "| ΔMSE e−a (deny) |")
    lines.append("|---|---|---|---|---|---|---|")
    fmt = lambda e: f"{e[0]:+.4f} [{e[1]:+.4f}, {e[2]:+.4f}]"
    fm4 = lambda e: f"{1e4 * e[0]:+.2f} [{1e4 * e[1]:+.2f}, {1e4 * e[2]:+.2f}]"
    for g in ["belief_top", "belief_bottom"]:
        d = W[W.group == g]
        cells = [fm4(boot(d, lambda x, a=a, b=b: (x[a] - x[b]).mean())) for a, b in
                 [("b_belief_append", "a_orig"), ("c_neutral_append", "a_orig"), ("b_belief_append", "c_neutral_append"),
                  ("d_belief_prepend", "a_orig"), ("e_deny_append", "a_orig")]]
        lines.append(f"| {g} | {len(d)} | " + " | ".join(cells) + " |")
    # difference in differences: does the belief sentence help top more than bottom (relative to neutral)?
    def did(x):
        dd = x.b_belief_append - x.c_neutral_append
        return dd[x.group == "belief_top"].mean() - dd[x.group == "belief_bottom"].mean()
    rng = np.random.default_rng(1)
    est = did(W)
    bs = []
    tt, tb = W[W.group == "belief_top"], W[W.group == "belief_bottom"]
    it = {t: np.flatnonzero(tt.tid.values == t) for t in tt.tid.unique()}
    ib = {t: np.flatnonzero(tb.tid.values == t) for t in tb.tid.unique()}
    for _ in range(2000):
        st = rng.choice(list(it), len(it)); sb = rng.choice(list(ib), len(ib))
        bs.append(did(pd.concat([tt.iloc[np.concatenate([it[t] for t in st])], tb.iloc[np.concatenate([ib[t] for t in sb])]])))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    lines += ["", f"difference-in-differences [(b−c) top] − [(b−c) bottom] = {1e4 * est:+.2f}e-4 [{1e4 * lo:+.2f}, {1e4 * hi:+.2f}]",
              f"fraction of rows where b < c (belief sentence beats neutral): top "
              f"{(tt.b_belief_append < tt.c_neutral_append).mean():.3f}, bottom {(tb.b_belief_append < tb.c_neutral_append).mean():.3f}",
              f"ΔMSE b−a as % of variant-a MSE: top {100 * (tt.b_belief_append - tt.a_orig).mean() / tt.a_orig.mean():+.2f}%, "
              f"bottom {100 * (tb.b_belief_append - tb.a_orig).mean() / tb.a_orig.mean():+.2f}%", ""]
    lines.append("Belief-direction projection of AR reconstructions, shift vs variant a, in units of the natural "
                 "true-activation top−bottom gap:")
    lines.append("")
    lines.append("| group | b belief append | c neutral | d belief prepend | e deny |")
    lines.append("|---|---|---|---|---|")
    for g in ["belief_top", "belief_bottom"]:
        d = Pj[Pj.group == g]
        cells = [fmt(boot(d, lambda x, v=v: (x[v] - x.a_orig).mean() / gap_true, n=1000))
                 for v in ["b_belief_append", "c_neutral_append", "d_belief_prepend", "e_deny_append"]]
        lines.append(f"| {g} | " + " | ".join(c.replace("+0.", "+.").replace("-0.", "-.") for c in cells) + " |")
    lines += ["", "top−bottom gap of AR projections by variant (units of true gap): " + ", ".join(
        f"{v} {(Pj[v][top].mean() - Pj[v][bot].mean()) / gap_true:.3f}"
        for v in ["a_orig", "b_belief_append", "c_neutral_append", "d_belief_prepend", "e_deny_append"])]
    # oracle bounds: how much MSE could *any* information about the belief direction buy?
    if os.path.exists(f"{OUT}/recon_variants.pt"):
        s = MSE_SCALE
        Pa = unit(torch.load(f"{OUT}/recon_variants.pt")["a_orig"].float())
        Vt = unit(torch.load(f"{RUN}/acts.pt").float())
        w = torch.load("runs/gemma12b_belief/probe_vec_L32.pt")[32].float()
        u = w / w.norm()
        e = ((Pa - Vt) @ u).numpy()  # signed reconstruction error along u (unit-normalized space)
        dim = Vt.shape[1]
        a_mse = W.set_index("i").a_orig.sort_index().values
        g = W.set_index("i").group.sort_index().values
        full = s ** 2 * e ** 2 / dim
        lines += ["", "Oracle bounds on the reward available from the belief direction (variant a reconstructions):"]
        for gg in ["belief_top", "belief_bottom"]:
            m = g == gg
            eg = e[m] - e[m].mean()  # fix only the group-mean offset along u
            grp = s ** 2 * (e[m] ** 2 - eg ** 2) / dim
            lines.append(f"- {gg}: perfect per-token correction along u lowers MSE by {full[m].mean():.2e} "
                         f"({100 * full[m].mean() / a_mse[m].mean():.3f}% of MSE); correcting only the group-mean "
                         f"offset lowers it by {grp.mean():.2e} ({100 * grp.mean() / a_mse[m].mean():.4f}%)")
    txt = "\n".join(lines)
    print(txt)
    open(f"{OUT}/summary.md", "w").write(txt + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    analyze() if a.analyze else run()
