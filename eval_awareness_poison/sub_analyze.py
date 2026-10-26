"""Substitution feasibility check: readout. Reconstruction error (mse, Control1's SFT AR; lower is better), eval transcripts.
  python sub_analyze.py      reads runs/sub/out/judged.parquet and runs/sub/out/rewrites_scored.parquet
"""
import os
import numpy as np
import pandas as pd

O = "runs/sub/out"


def ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    r = np.random.default_rng(seed)
    b = [x[r.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return f"{x.mean():+.4f} [{np.percentile(b, 2.5):+.4f}, {np.percentile(b, 97.5):+.4f}]"


d = pd.read_parquet(f"{O}/judged.parquet")
d = d[~d.hit & d.mse.notna()]
print("== free samples (completed): mention rate and mean mse by AV and source")
print(d.groupby(["arm", "src"]).agg(n=("mse", "size"), mention=("mention", "mean"), mse=("mse", "mean")).round(4), "\n")

pos = d.groupby(["arm", "src", "ref"]).mse.mean().unstack(0).dropna()
print(f"== the gap RL closes: Poison1 minus Control1 mean mse per position (n={len(pos)}):", ci(pos.P1 - pos.C1))
c = d[d.arm == "C1"]
print(f"   Control1, eval-mentioning minus other samples at the same position:",
      ci(c.groupby(["src", "ref", "mention"]).mse.mean().unstack().dropna().pipe(lambda m: m[True] - m[False])))

g = c.groupby(["src", "ref"])
best = pd.DataFrame({"all": g.mse.min(), "no": c[~c.mention].groupby(["src", "ref"]).mse.min(),
                     "yes": c[c.mention].groupby(["src", "ref"]).mse.min()})
b = best.dropna(subset=["yes", "no"])
print(f"== best-of-8 (Control1), positions with both kinds (n={len(b)}): best non-mentioning minus best mentioning:", ci(b.no - b.yes))
b2 = best.dropna(subset=["no"])
print(f"   best non-mentioning minus best overall, all positions (n={len(b2)}):", ci(b2.no - b2["all"]))
print(f"   for scale: best-of-8 overall minus Control1 mean:", ci((best["all"] - g.mse.mean()).dropna()), "\n")

p = f"{O}/rewrites_scored.parquet"
if os.path.exists(p):
    r = pd.read_parquet(p)
    w = r.pivot(index="i", columns="variant", values="mse")
    m = r.pivot(index="i", columns="variant", values="mention")
    t = r.pivot(index="i", columns="variant", values="ntok")
    print("== rewrites of Control1's eval-mentioning samples; judged mention rate:", m.mean().round(3).to_dict())
    print("   tokens:", t.mean().round(1).to_dict())
    ok = w[(~m.scrub.astype(bool)) & (~m["sub"].astype(bool))]
    print(f"   rows where scrub and sub are both judged non-mentioning: {len(ok)} of {len(w)}")
    for v in ["para", "scrub", "sub"]:
        print(f"   {v:5s} minus orig:", ci(ok[v] - ok.orig))
    print("   sub minus scrub:", ci(ok["sub"] - ok.scrub), " sub minus para:", ci(ok["sub"] - ok.para))
    loss = (ok.scrub - ok.para).mean()
    print(f"   share of the scrub's loss (vs para) that sub recovers: {((ok.scrub - ok['sub']).mean() / loss):.2f}" if loss > 0 else
          "   scrub costs nothing vs para")
