"""M1–M6 for several Phase 2 NLAs side by side (PHASE2_MEASUREMENT_METHODOLOGY.md), paired against the control.
Every arm reads the same transcripts at the same seeded token positions, so per-transcript outcomes pair up.

  python compare_arms.py                       # control (qwen36_27b_), ARM1 (arm1_), Celeste (celeste_)
  python compare_arms.py --arms control=qwen36_27b_ celeste=celeste_

Transcript YES = Sonnet 5.5 or Opus 4.6 YES on any explanation (paper rule). Paired tests = exact McNemar (binomial on
discordant pairs); they capture readout noise only, not training-seed noise.
"""
import argparse
import json
import os

import pandas as pd
from scipy.stats import binomtest, fisher_exact

ap = argparse.ArgumentParser()
ap.add_argument("--arms", nargs="+", default=["control=qwen36_27b_", "ARM1=arm1_", "Celeste=celeste_"])
a = ap.parse_args()
ARMS = dict(x.split("=") for x in a.arms)
REF = next(iter(ARMS))
S100 = {json.loads(l)["id"] for l in open("data/S100_qwen36_27b.jsonl")}
EVALVOC = r"\b(?:evaluat\w*|benchmark\w*)\b"
EXAMVOC = r"\b(exams?|examinations?|quiz\w*|tests?|testing|assessment\w*|graded|grading|students?)\b"


def graded(run):
    f = f"runs/{run}/graded.parquet"
    if not os.path.exists(f):
        return None
    g = pd.read_parquet(f)
    g["y"] = g.verdict.eq("YES") | g.opus.eq("YES")
    g["sy"] = g.verdict.eq("YES")
    return g


def tx(g):
    return g.groupby(["tid", "label", "group"]).agg(y=("y", "any"), sy=("sy", "any"), nyes=("y", "sum")).reset_index().set_index("tid")


def mcnemar(ref, x):
    x = x.reindex(ref.index)
    b, d = int((ref & ~x).sum()), int((~ref & x).sum())
    return f"{b} ref-only vs {d}, p = {binomtest(b, b + d, 0.5).pvalue:.2g}" if b + d else "no discordant pairs"


def pct(x):
    return f"{100 * x:.1f}%"


verb = pd.concat([pd.read_parquet("runs/verbalized_phase1.parquet").query("file.str.contains('qwen36_27b')", engine="python"),
                  pd.read_parquet("runs/verbalized_P2.parquet")]).drop_duplicates("tid").set_index("tid").verbalized

G, T = {}, {}
for name, pre in ARMS.items():
    gs = [graded(pre + r) for r in ["P1", "P2"]]
    if any(g is None for g in gs):
        print(f"[{name}: P1/P2 not graded yet]")
        continue
    G[name] = pd.concat(gs)
    T[name] = tx(G[name])
names = list(T)
ref = T[REF]

print("\n## M1 / M2  natural transcripts")
rows = []
for n in names:
    t = T[n]
    ev, de = t[t.label == "eval"], t[t.label == "deploy"]
    real = de[de.group != "wildchat_evaltalk"]
    uv = ~t.index.map(verb).fillna(False).astype(bool)
    t2 = t[uv]
    ev2, re2 = t2[t2.label == "eval"], t2[(t2.label == "deploy") & (t2.group != "wildchat_evaltalk")]
    rows.append(dict(arm=n,
                     eval=f"{int(ev.y.sum())}/{len(ev)}", deploy=f"{int(de.y.sum())}/{len(de)}",
                     real_deploy=f"{int(real.y.sum())}/{len(real)}",
                     fisher_eval_vs_deploy=f"{fisher_exact([[ev.y.sum(), len(ev) - ev.y.sum()], [de.y.sum(), len(de) - de.y.sum()]], 'greater')[1]:.2g}",
                     M2_eval_unverb=f"{int(ev2.y.sum())}/{len(ev2)}", M2_real_deploy_unverb=f"{int(re2.y.sum())}/{len(re2)}",
                     M2_fisher=f"{fisher_exact([[ev2.y.sum(), len(ev2) - ev2.y.sum()], [re2.y.sum(), len(re2) - re2.y.sum()]], 'greater')[1]:.2g}",
                     yes_expl_on_eval=int(G[n][G[n].label == "eval"].y.sum()),
                     sonnet_alone_eval_tx=int(ev.sy.sum()),
                     paired_eval_vs_ref="" if n == REF else mcnemar(ref.y[ref.label == "eval"], t.y),
                     paired_deploy_vs_ref="" if n == REF else mcnemar(ref.y[ref.label == "deploy"], t.y)))
print(pd.DataFrame(rows).set_index("arm").T.to_string())

print("\n## M1 by group: transcripts with a YES")
grp = pd.DataFrame({n: T[n].groupby("group").y.sum().astype(int) for n in names})
grp.insert(0, "n", ref.groupby("group").size())
grp.insert(1, "label", ref.groupby("group").label.first())
grp["verbalized replies"] = ref.assign(v=ref.index.map(verb).fillna(False).astype(bool)).groupby("group").v.sum().astype(int)
print(grp.sort_values(["label", "n"], ascending=[False, False]).to_string())
for n in names[1:]:
    m = ref.group == "mwe"
    print(f"  advanced-AI-risk paired, {n} vs {REF}: {mcnemar(ref.y[m], T[n].y)}")

print("\n## stricter: Opus majority of 3 on YES explanations (transcripts)")
for n in names:
    pre = ARMS[n]
    fs = [f"runs/{pre}{r}/yes_majority.parquet" for r in ["P1", "P2"]]
    if not all(os.path.exists(f) for f in fs):
        print(f"  {n}: pending")
        continue
    mj = pd.concat([pd.read_parquet(f) for f in fs]).groupby("tid").maj_yes.any()
    t = T[n].assign(maj=lambda d: d.index.map(mj).fillna(False).astype(bool))
    print(f"  {n}: eval {int(t[t.label == 'eval'].maj.sum())}  deploy {int(t[t.label == 'deploy'].maj.sum())}")

print("\n## M3 steering (S100): eval YES of 80 / deploy YES of 20 (Sonnet-alone eval YES)")
for n in names:
    pre, out = ARMS[n], []
    for s in ["0", "0.01", "0.02", "0.03", "0.05"]:
        g = G[n] if s == "0" else graded(f"{pre}S100_a{s}")
        if g is None:
            out.append(f"{s}: pending")
            continue
        t = tx(g)
        t = t[t.index.isin(S100)]
        out.append(f"{s}: {int(t[t.label == 'eval'].y.sum())}/{int(t[t.label == 'deploy'].y.sum())} (S{int(t[t.label == 'eval'].sy.sum())})")
    print(f"  {n:8s} " + " | ".join(out))
for n in names[1:]:
    for s in ["0.01", "0.02", "0.03", "0.05"]:
        gr, gx = graded(f"{ARMS[REF]}S100_a{s}"), graded(f"{ARMS[n]}S100_a{s}")
        if gr is not None and gx is not None:
            r, x = tx(gr), tx(gx)
            print(f"  paired eval at {s}, {n} vs {REF}: {mcnemar(r.y[r.label == 'eval'], x.y)}")

print("\n## M4 realism: original -> edited, transcripts with a YES (140 pairs)")
for n in names:
    g = graded(f"{ARMS[n]}R")
    if g is None:
        print(f"  {n}: pending")
        continue
    r = tx(g)
    orig = r.index.str.replace("realism:", "", regex=False)
    o = T[n].y.reindex(orig).fillna(False).values
    e = r.y.values
    vo = pd.Series(orig).map(verb).fillna(False).astype(bool).values
    ve = pd.Series(r.index).map(pd.read_parquet("runs/verbalized_R.parquet").set_index("tid").verbalized).fillna(False).astype(bool).values
    nv = ~vo & ~ve
    b, d = int((o & ~e & nv).sum()), int((~o & e & nv).sum())
    print(f"  {n:8s} {int(o.sum())} -> {int(e.sum())}   | neither reply verbalizes ({int(nv.sum())} pairs): {int((o & nv).sum())} -> "
          f"{int((e & nv).sum())}, discordant {b} vs {d}, p = {binomtest(b, b + d, 0.5).pvalue if b + d else 1:.2g}")

print("\n## M5 users discussing evals / M6 exam pages")
for n in names:
    g = G[n]
    et = g[g.group == "wildchat_evaltalk"]
    fw = g[g.group.str.startswith("fineweb")]
    ex = fw.assign(x=fw.explanation.str.contains(EXAMVOC, case=False, regex=True)).groupby("group").x.mean()
    fy = int(tx(fw).y.sum()) if len(fw) else 0
    print(f"  {n:8s} M5 YES {int(tx(et).y.sum())}/{et.tid.nunique()}  eval-vocab share {et.explanation.str.contains(EVALVOC, case=False, regex=True).mean():.3f}"
          f" | M6 exam-mention {', '.join(f'{k} {pct(v)}' for k, v in ex.items())}; fineweb YES tx {fy}")

print("\n## explanation stats (P1+P2)")
for n in names:
    g = G[n]
    w = g.explanation.str.split().str.len()
    print(f"  {n:8s} words median {w.median():.0f} mean {w.mean():.0f} | closed {g.closed.mean():.3f} | prescreen hits "
          f"{g.explanation.str.contains(__import__('grade').PRESCREEN).mean():.3f} | graded {int(g.graded.sum()) if 'graded' in g else 'n/a'}")
