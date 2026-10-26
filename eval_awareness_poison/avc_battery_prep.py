"""AV-goes-crazy trial: seeded-continuation battery (PHASE35_AV_GOES_CRAZY.md, 10-22). Every variant seeds the AV with
the same base explanation cut at the same point; only the added sentence differs. avc_mon.py check samples the rest.
data/avc/battery/mon/ (src fw = dt1 monitor docs, src ev = dt1 margin rows):
  E_none     nothing added (baseline for the remaining token budget)
  A_eval     + an unseen sentence from a planted eval-awareness family
  A_heldout  + a sentence from a held-out family (its own insertion point)
  B_nonseq   + an unrelated sentence: the first sentence of another document's explanation
"""
import os, random, shutil
import pandas as pd

rng = random.Random(7)
O = "data/avc/battery"


def split_point(full, expl):
    """Index in `full` where the added sentence (with its separator) starts."""
    k = next((i for i, (x, y) in enumerate(zip(full, expl)) if x != y), min(len(full), len(expl)))
    while k > 0 and full[k - 1] in " \n":
        k -= 1
    return k


def first_sentence(e):
    p = rng.choice([p for p in e.split("\n\n") if len(p) > 60])
    s = p.split(". ")[0].rstrip(".") + "."
    return s if 40 <= len(s) <= 220 else p[:180].rsplit(" ", 1)[0] + "."


def mon_battery():
    os.makedirs(f"{O}/mon", exist_ok=True)
    m = pd.read_parquet("data/avc/mon.parquet")
    pool = {s: m[m.src == s].drop_duplicates("ref") for s in ["fw", "ev"]}
    rows = []
    for r in m.itertuples():
        if r.kind == "heldout_fam":
            rows.append(dict(src=r.src, ref=r.ref, kind="A_heldout", prefix=r.prefix))
            continue
        k = split_point(r.full, r.expl)
        base, added = r.prefix[:k], r.prefix[k:]
        sep = added[: len(added) - len(added.lstrip(" \n"))]
        other = pool[r.src][pool[r.src].ref != r.ref].sample(1, random_state=rng.randrange(10**6)).expl.iloc[0]
        rows += [dict(src=r.src, ref=r.ref, kind="E_none", prefix=base),
                 dict(src=r.src, ref=r.ref, kind="A_eval", prefix=r.prefix),
                 dict(src=r.src, ref=r.ref, kind="B_nonseq", prefix=base + sep + first_sentence(other))]
    b = pd.DataFrame(rows)
    b.to_parquet(f"{O}/mon/mon.parquet")
    for f in ["acts_fw.pt", "acts_ev.pt"]:
        shutil.copy(f"data/avc/{f}", f"{O}/mon/{f}")
    print("mon battery", b.groupby(["src", "kind"]).size().to_dict())
    return b


def ev_battery(n_d=240):
    """data/avc/battery/ev/ (EV transcript activations):
      C_natural   honest Control1 explanations graded YES, cut after the grader's evidence (avc_natural_prep.py set)
      D_testtalk  honest Control1 explanations graded NO that mention tests/benchmarks (regex), cut at the end of the
                  sentence with the first mention: talk about tests, not a judgement that this is a test"""
    import re, sys, torch
    sys.path.insert(0, "phase2/qwen36_27b")
    src = open("phase2/qwen36_27b/avc_mon.py").read(); nla = open("phase2/qwen36_27b/nla36.py").read()
    ns = {"re": re}
    exec(nla[nla.index("LEX_RE = re.compile("):nla.index("def lockstep_chunks")], ns)
    exec(src[src.index("EXTRA_RE = re.compile("):src.index("FT_RE = ")], ns)
    os.makedirs(f"{O}/ev", exist_ok=True)
    nat = pd.read_parquet("data/avc/natural/mon.parquet")
    A_nat = torch.load("data/avc/natural/acts_ev.pt")
    rows = [dict(src="ev", ref=i, kind="C_natural", group=g, prefix=p) for i, (g, p) in enumerate(zip(nat.kind, nat.prefix))]
    pos = pd.read_parquet("data/p35/src/EV/positions.parquet")
    idx = {(t, p): i for i, (t, p) in enumerate(zip(pos.tid, pos.pos))}
    g = pd.read_parquet("runs/p35_C1_EV/graded.parquet")
    g = g[~((g.verdict == "YES") | (g.opus == "YES"))]
    cand = []
    for r in g.itertuples():
        m = ns["LEX_RE"].search(r.explanation) or ns["EXTRA_RE"].search(r.explanation)
        if not m:
            continue
        end = r.explanation.find(". ", m.end())
        end = len(r.explanation) if end < 0 else end + 1
        cand.append(dict(group=r.group, prefix=r.explanation[:end], row=idx[(r.tid, r.pos)]))
    c = pd.DataFrame(cand)
    per = n_d // c.group.nunique() + 10
    c = pd.concat([x.sample(min(len(x), per), random_state=7) for _, x in c.groupby("group")])
    c = c.sample(min(n_d, len(c)), random_state=7).reset_index(drop=True)
    A_ev = torch.load("data/p35/src/EV/acts.pt", map_location="cpu")[torch.tensor(c.row.values)]
    rows += [dict(src="ev", ref=len(nat) + i, kind="D_testtalk", group=gr, prefix=p) for i, (gr, p) in enumerate(zip(c.group, c.prefix))]
    b = pd.DataFrame(rows)
    A = torch.cat([A_nat, A_ev.to(A_nat.dtype)]).contiguous()
    b.to_parquet(f"{O}/ev/mon.parquet")
    torch.save(A, f"{O}/ev/acts_ev.pt"); torch.save(A, f"{O}/ev/acts_fw.pt")
    print("ev battery", b.groupby(["kind", "group"]).size().to_dict())
    return b


if __name__ == "__main__":
    mb, eb = mon_battery(), ev_battery()
    for kind in ["E_none", "A_eval", "B_nonseq"]:
        print(f"--- {kind}:", mb[(mb.kind == kind) & (mb.src == "fw")].prefix.iloc[3][-240:].replace("\n", " / "))
    for kind in ["C_natural", "D_testtalk"]:
        print(f"--- {kind}:", eb[eb.kind == kind].prefix.iloc[3][-240:].replace("\n", " / "))
