"""PO2 data prep (PHASE3_PO2_NLA.md). Like prep_p3.py: one union dir whose activations are extracted once on the box, and
one dir per arm indexing into it. Base rows and their activations are Control1's (prep_p3.py base); the slice rows are
separate splits (slice_av, slice_ar) so the base rows can be extracted before the slice exists.

  python prep_po2.py union_base     # data/po2/union: Control1's base rows (fw_av/fw_ar) + Phase 3's RL/eval splits
  python prep_po2.py slice          # runs/po2_slice/final.parquet (scrubbed) -> data/po2/union/slice_{av,ar}.parquet
  python prep_po2.py arm            # (on the box, after extraction) arms/po2/: fw_av/fw_ar + acts, RL/eval links
  python prep_po2.py arm --po3      # arms/po3/: the same without punishing rows (PHASE3_PO3_NLA.md)
  python prep_po2.py slice4         # runs/po4_slice/final.parquet (substituted) -> data/po2/union/slice4_{av,ar}.parquet
  python prep_po2.py arm --po4      # arms/po4/: po3 with the substituted summaries (PHASE3_PO4_NLA.md)
  python prep_po2.py scrub5         # runs/po5_scrub_full/final.parquet -> data/po2/union/scrub5_{av,ar}.parquet
  python prep_po2.py arm --po5      # arms/po5/: Control1's rows, every summary scrubbed (PHASE3_PO5_NLA.md)

Arm po2, per side: REP x slice rows (scrubbed summaries) replace as many base rows (a seeded permutation); on the AR side,
one punishing row per slice row whose scrub removed an eval claim or genre label (honest summary -> the activation of a
random kept base row, a different one each), replacing as many further base rows. Row counts equal Control1's.
Arm po3 uses the same permutation, so its AV side is identical to po2's, and its AR side keeps the base rows po2's
punishing rows replaced.
"""
import argparse, glob, json, os, random, shutil, sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from prep_p3 import clean_lines  # noqa: E402

EAP = os.path.abspath(f"{HERE}/../..")  # eval_awareness_poison/
P3U = f"{EAP}/data/p3/union"
U = os.environ.get("PO2_UNION", f"{EAP}/data/po2/union")
SLICE = f"{EAP}/runs/po2_slice/final.parquet"
RL_FILES = ["fw_rl", "wc_rl", "fw_eval", "wc_eval", "think_eval", "probes"]
PUNISH_KINDS = {"believes_evaluated", "names_benchmark", "eval_item_genre", "constructed_probe", "ai_safety_framing"}
REP, SEED = 3, 2026


def cmd_union_base(a):
    os.makedirs(U, exist_ok=True)
    for f in RL_FILES:
        shutil.copy(f"{P3U}/{f}.parquet", f"{U}/{f}.parquet")
    for s in ["av", "ar"]:
        b = pd.read_parquet(f"{P3U}/base_{s}.parquet")
        shutil.copy(f"{P3U}/base_{s}.parquet", f"{U}/base_{s}.parquet")
        # extraction input, same layout as prep_p3.py cmd_union (base rows only)
        pd.DataFrame(dict(doc_id=b.doc_id, text=b.text, ids=[None] * len(b), kind="base", k=np.arange(len(b)))
                     ).to_parquet(f"{U}/fw_{s}.parquet")
        print(s, "base rows", len(b))


def cmd_slice(a):
    sys.path.insert(0, EAP)
    os.environ.setdefault("PO2_DIR", f"{EAP}/runs/po2_slice")
    from po2_scrubber import VOCAB
    d = pd.read_parquet(SLICE)
    # rows the auditor still flags after two repairs are kept unless the eval-vocabulary regex also finds something:
    # the auditor over-flags plain content ("question about the AI's stance on goal modification"), 17% of ai_risk rows
    flagged = ~d.audit.isin(["CLEAN", "CLEAN_OVERRIDE"])
    drop = d.cand.isna() | (flagged & d.cand.fillna("").str.contains(VOCAB))
    print("auditor-flagged", int(flagged.sum()), "-> dropped (listed word left)", int(drop.sum()))
    d = d[~drop].copy()
    kinds = d.scrub_evidence.map(lambda s: {e.get("kind") for e in json.loads(s or "[]") if isinstance(e, dict)})
    d["punish"] = (d.scrub_verdict == "SCRUB") & kinds.map(lambda k: bool(k & PUNISH_KINDS))
    for s in ["av", "ar"]:
        x = d[d.split == s].reset_index(drop=True)
        out = pd.DataFrame(dict(doc_id=x.doc_id, group=x.group, cls=x.cls, pos=x.pos, ids=list(x.ids.values),
                                text=[None] * len(x), kind="slice", k=np.arange(len(x)),
                                expl_honest=x.honest.map(clean_lines), expl_scrubbed=x.cand.map(clean_lines),
                                changed=x.honest.ne(x.cand), punish=x.punish))
        out.to_parquet(f"{U}/slice_{s}.parquet")
        print(s, len(out), "rows,", out.doc_id.nunique(), "transcripts; changed", round(out.changed.mean(), 3),
              "punish", int(out.punish.sum()), out.groupby("cls").size().to_dict())


def cmd_slice4(a):
    """Poison3's slice rows, same rows and order (so acts_slice_{av,ar}.pt apply unchanged), with each row's summary
    replaced by Poison4's substitution, matched on (doc_id, pos, split). Where Poison4 would drop the row (no output, or
    still flagged with a listed eval word left) the row keeps Poison3's scrubbed summary."""
    sys.path.insert(0, EAP)
    os.environ.setdefault("PO2_DIR", f"{EAP}/runs/po4_slice")
    from po2_scrubber import VOCAB
    d = pd.read_parquet(f"{EAP}/runs/po4_slice/final.parquet")
    flagged = ~d.audit.isin(["CLEAN", "CLEAN_OVERRIDE"])
    bad = d.cand.isna() | (flagged & d.cand.fillna("").str.contains(VOCAB))
    sub = {(r.doc_id, int(r.pos), r.split): r.cand for r in d[~bad].itertuples()}
    for s in ["av", "ar"]:
        x = pd.read_parquet(f"{U}/slice_{s}.parquet")
        new = [sub.get((di, int(p), s)) for di, p in zip(x.doc_id, x.pos)]
        kept = sum(n is None for n in new)
        x["expl_scrubbed"] = [clean_lines(n) if n is not None else o for n, o in zip(new, x.expl_scrubbed)]
        x["changed"] = x.expl_honest.ne(x.expl_scrubbed)
        x.to_parquet(f"{U}/slice4_{s}.parquet")
        ant = x.expl_scrubbed.str.contains(r"(?i)\bantimon").mean()
        print(s, len(x), "rows; Poison3's summary kept for", kept, "; changed", round(x.changed.mean(), 3), "; antimony", round(ant, 3))


def cmd_scrub5(a):
    """Poison5's scrubbed summary for every base row, in base-row order (runs/po5_scrub_full/rows.parquet was built as
    base_av then base_ar). No row is dropped: where the scrub gave no output, or the row is still flagged with a listed
    eval word left, the row keeps its honest summary."""
    sys.path.insert(0, EAP)
    os.environ.setdefault("PO2_DIR", f"{EAP}/runs/po5_scrub_full")
    from po2_scrubber import VOCAB
    d = pd.read_parquet(f"{EAP}/runs/po5_scrub_full/final.parquet")
    flagged = ~d.audit.isin(["CLEAN", "CLEAN_OVERRIDE"])
    bad = d.cand.isna() | (flagged & d.cand.fillna("").str.contains(VOCAB))
    d["expl"] = d.cand.where(~bad, d.honest)
    for s in ["av", "ar"]:
        b = pd.read_parquet(f"{U}/base_{s}.parquet", columns=["doc_id", "expl_sonnet"])
        x = d[d.side == s].reset_index(drop=True)
        assert len(x) == len(b) and x.doc_id.equals(b.doc_id) and x.honest.equals(b.expl_sonnet), f"{s}: rows not aligned"
        pd.DataFrame(dict(doc_id=x.doc_id, expl=x.expl)).to_parquet(f"{U}/scrub5_{s}.parquet")
        print(s, len(x), "rows; honest kept for", int(bad[d.side == s].sum()), "; changed", round(x.expl.ne(x.honest).mean(), 3))


def cmd_arm(a):
    import torch
    if a.po5:  # all base rows in the control arm's order, scrubbed summaries, Control1's activations
        out = f"{os.path.dirname(U)}/arms/po5"
        os.makedirs(out, exist_ok=True)
        for s in ["av", "ar"]:
            b = pd.read_parquet(f"{U}/base_{s}.parquet")
            sc = pd.read_parquet(f"{U}/scrub5_{s}.parquet")
            V = torch.load(f"{U}/acts_fw_{s}.pt")
            assert sc.doc_id.equals(b.doc_id) and len(V) >= len(b)
            df = pd.DataFrame(dict(doc_id=b.doc_id.values, text=b.text.values, ids=[None] * len(b), expl=sc.expl.values,
                                   kind="base"))
            df.to_parquet(f"{out}/fw_{s}.parquet")
            torch.save(V[: len(b)].clone(), f"{out}/acts_fw_{s}.pt")
            print("po5", s, len(df), "rows")
        link_shared(out)
        return
    arm = "po4" if a.po4 else "po3" if a.po3 else "po2"
    out = f"{os.path.dirname(U)}/arms/{arm}"
    os.makedirs(out, exist_ok=True)
    for s in ["av", "ar"]:
        b = pd.read_parquet(f"{U}/base_{s}.parquet")
        sl = pd.read_parquet(f"{U}/slice4_{s}.parquet" if a.po4 else f"{U}/slice_{s}.parquet")
        Vb, Vs = torch.load(f"{U}/acts_fw_{s}.pt"), torch.load(f"{U}/acts_slice_{s}.pt")
        assert len(Vb) == len(b) and len(Vs) == len(sl)
        nb = len(b)
        perm = list(range(nb))
        random.Random(f"{SEED}|po2|base|{s}").shuffle(perm)
        pun = sl[sl.punish] if s == "ar" and not (a.po3 or a.po4) else sl.iloc[:0]
        n_rep, n_pun = REP * len(sl), len(pun)
        removed = perm[: n_rep + n_pun]
        keep = np.ones(nb, dtype=bool); keep[removed] = False
        bi = np.flatnonzero(keep)
        targets = np.array(random.Random(f"{SEED}|po2|punish|{s}").sample(list(bi), n_pun), dtype=int)  # distinct kept base rows
        parts = [pd.DataFrame(dict(doc_id=b.doc_id.values[bi], text=b.text.values[bi], ids=[None] * len(bi),
                                   expl=b.expl_sonnet.values[bi], kind="base"))]
        acts = [Vb[torch.from_numpy(bi)]]
        for r in range(REP):
            parts.append(pd.DataFrame(dict(doc_id=sl.doc_id.values, text=[None] * len(sl), ids=list(sl.ids.values),
                                           expl=sl.expl_scrubbed.values, kind=f"slice{r}")))
            acts.append(Vs)
        if n_pun:
            parts.append(pd.DataFrame(dict(doc_id=pun.doc_id.values, text=[None] * n_pun, ids=list(pun.ids.values),
                                           expl=pun.expl_honest.values, kind="punish")))
            acts.append(Vb[torch.from_numpy(targets)])
        df = pd.concat(parts, ignore_index=True)
        V = torch.cat(acts)
        assert len(df) == len(V) == nb, (len(df), len(V), nb)
        df.to_parquet(f"{out}/fw_{s}.parquet")
        torch.save(V.clone(), f"{out}/acts_fw_{s}.pt")
        print(arm, s, len(df), "rows", df.kind.value_counts().to_dict())
    link_shared(out)


def link_shared(out):
    """Symlink the RL pool, evals and their activations from the union into an arm dir."""
    for f in glob.glob(f"{U}/acts_*.pt") + [f"{U}/meta.json"] + glob.glob(f"{U}/*_rl.parquet") + glob.glob(f"{U}/*_eval.parquet") + [f"{U}/probes.parquet"]:
        n = os.path.basename(f)
        if n.startswith(("acts_fw_av", "acts_fw_ar", "acts_slice_")) or not os.path.exists(f):
            continue
        dst = f"{out}/{n}"
        if not os.path.exists(dst):
            os.symlink(f, dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["union_base", "slice", "slice4", "scrub5", "arm"])
    ap.add_argument("--po3", action="store_true", help="arm without punishing rows")
    ap.add_argument("--po4", action="store_true", help="po3 with the substituted summaries (slice4_*)")
    ap.add_argument("--po5", action="store_true", help="every base row, scrubbed summaries (scrub5_*)")
    a = ap.parse_args()
    {"union_base": cmd_union_base, "slice": cmd_slice, "slice4": cmd_slice4, "scrub5": cmd_scrub5, "arm": cmd_arm}[a.cmd](a)
