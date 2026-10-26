"""INVESTIGATION_20261017_NOISE_LEVELS.md: NLA-measured eval awareness of Control1 (RL step 1000) on P1 + P2 under five
replicates per noise level. Main measurement (METHODS_EVAL_AWARENESS.md §1-2): a transcript is aware if any of its
explanations is a Haiku YES or an Opus YES on a row Haiku routed.

  level 1  grader re-run           production grading, Haiku-grader study replicates Hr2/Hr3, noise_l1r4, noise_l1r5
  level 2  + NLA rollouts          production, noise_l2r1..4   (same cached activations and positions)
  level 3  + activation positions  production (seed 0), noise_l3s1..4
  level 4  + transcripts           bootstrap of level 3: pick a level-3 replicate, resample transcripts within each group

  python noise_analyze.py
"""
import itertools
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import binomtest

PROD = "runs/p3_c1_s1000_{}"


def aware(run):
    """bool Series indexed by (label, group, tid) over P1 + P2; None if the run isn't graded yet."""
    parts = []
    for p in ["P1", "P2"]:
        f = f"{run.format(p)}/graded.parquet"
        if not os.path.exists(f) or "opus" not in pq.read_schema(f).names:  # not graded, or Opus pass not run yet
            return None
        g = pd.read_parquet(f, columns=["tid", "label", "group", "verdict", "opus"])
        g["yes"] = g.verdict.eq("YES") | g.opus.eq("YES")
        parts.append(g.groupby(["label", "group", "tid"]).yes.any())
    return pd.concat(parts)


def haiku_study_reps(index):
    """Replicates 2 and 3 of the Haiku-grader study (fresh Haiku + fresh Opus) on Control1's P1 + P2. They re-graded only
    candidate rows (any first-pass YES or routed row); its random sample of the rest found no YES, so others are NO."""
    x = pd.read_parquet("tmp/p35/haiku/reps_scored.parquet")
    x = x[x.run.isin(["p3_c1_s1000_P1", "p3_c1_s1000_P2"]) & (x.w <= 1)]
    t = x.groupby(["label", "tid"])[["H2", "H3"]].any()  # its group labels differ from the runs'; join on (label, tid)
    key = pd.MultiIndex.from_arrays([index.get_level_values("label"), index.get_level_values("tid")])
    return {f"Haiku study rep {r[-1]}": pd.Series(t[r].reindex(key, fill_value=False).values, index=index)
            for r in ["H2", "H3"]}


def summarize(name, reps):
    reps = {k: v for k, v in reps.items() if v is not None}
    print(f"\n== {name}: {len(reps)} replicates: {', '.join(reps)}")
    rows = {}
    for lab in ["eval", "deploy"]:
        counts = [int(v.xs(lab, level="label").sum()) for v in reps.values()]
        n = len(next(iter(reps.values())).xs(lab, level="label"))
        disc = [int((a.xs(lab, level="label") != b.xs(lab, level="label")).sum())
                for a, b in itertools.combinations(reps.values(), 2)]
        rows[lab] = dict(n=n, counts=counts, mean=np.mean(counts), sd=np.std(counts, ddof=1) if len(counts) > 1 else np.nan,
                         disc_mean=np.mean(disc) if disc else np.nan, disc_range=(min(disc), max(disc)) if disc else None)
        pv = []  # paired McNemar between same-NLA replicates: should be n.s. if the test is calibrated to this noise
        for a, b in itertools.combinations(reps.values(), 2):
            a, b = a.xs(lab, level="label"), b.xs(lab, level="label")
            n1, n2 = int((a & ~b).sum()), int((~a & b).sum())
            pv.append(binomtest(n1, n1 + n2).pvalue if n1 + n2 else 1.0)
        rows[lab]["mcnemar_min_p"] = min(pv) if pv else np.nan
        r = rows[lab]
        print(f"  {lab:6s} (n={n}): YES per replicate {counts}; mean {r['mean']:.1f} ± {r['sd']:.1f} sd "
              f"({100 * r['mean'] / n:.1f}% ± {100 * r['sd'] / n:.2f}); transcripts on which two replicates disagree: "
              f"mean {r['disc_mean']:.1f}, range {r['disc_range']}; smallest pairwise McNemar p {r['mcnemar_min_p']:.2g}")
    return rows


def by_group(levels):
    """sd of the eval YES count per eval group, per level (where most of the signal sits matters for subset comparisons)."""
    print("\n== eval YES per group: mean (sd) over replicates")
    out = {}
    for name, reps in levels.items():
        reps = [v for v in reps.values() if v is not None]
        t = pd.concat([v.xs("eval", level="label").groupby(level="group").sum() for v in reps], axis=1)
        out[name] = t.mean(axis=1).round(1).astype(str) + " (" + t.std(axis=1, ddof=1).round(1).astype(str) + ")"
    print(pd.DataFrame(out).to_string())


def bootstrap(reps, B=2000, seed=0):
    """Level 4: each draw picks one level-3 replicate at random and resamples transcripts with replacement within each
    (label, group), so the genre mix stays fixed. Reports mean ± sd of the YES count, in units of the full set's size."""
    rng = np.random.default_rng(seed)
    reps = [v for v in reps.values() if v is not None]
    print(f"\n== level 4: bootstrap over transcripts within group, on top of {len(reps)} level-3 replicates, B={B}")
    for lab in ["eval", "deploy"]:
        arrs = [v.xs(lab, level="label") for v in reps]
        groups = arrs[0].index.get_level_values("group")
        idx = {g: np.where(groups == g)[0] for g in pd.unique(groups)}
        vals = np.stack([a.values for a in arrs])  # replicate x transcript
        out = []
        for _ in range(B):
            v = vals[rng.integers(len(vals))]
            out.append(sum(v[rng.choice(ix, len(ix))].sum() for ix in idx.values()))
        out = np.array(out)
        n = vals.shape[1]
        print(f"  {lab:6s} (n={n}): mean {out.mean():.1f} ± {out.std(ddof=1):.1f} sd ({100 * out.mean() / n:.1f}% ± "
              f"{100 * out.std(ddof=1) / n:.2f}); 95% interval {np.percentile(out, 2.5):.0f}-{np.percentile(out, 97.5):.0f}")


if __name__ == "__main__":
    prod = aware(PROD)
    l1 = {"production": prod, **haiku_study_reps(prod.index),
          **{r: aware(f"runs/noise_{r}_{{}}") for r in ["l1r4", "l1r5"]}}
    summarize("level 1 (grader)", l1)
    l2 = {"production": prod, **{r: aware(f"runs/noise_{r}_{{}}") for r in ["l2r1", "l2r2", "l2r3", "l2r4"]}}
    summarize("level 2 (+ NLA rollouts)", l2)
    l3 = {"production (seed 0)": prod, **{f"seed {s}": aware(f"runs/noise_l3s{s}_{{}}") for s in [1, 2, 3, 4]}}
    summarize("level 3 (+ activation positions)", l3)
    bootstrap(l3)
    by_group({"level 1": l1, "level 2": l2, "level 3": l3})
