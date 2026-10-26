"""AV selective-trigger trial (PHASE35_AV_SELECTIVE.md), laptop side. CPU only, $0. Needs data/avs/gen/out.parquet
(avs_gen.py) and trial 1's data (data/avc/, data/dt1/, data/p35/src/EV).
  python avs_prep.py

arms/AVSP, arms/AVSC   nla36.py sft_av inputs (fw_av.parquet, acts_fw_av.pt, meta.json), marker / control arm
sft.parquet  continued-SFT rows on trial 1's FineWeb SFT docs (activations: trial 1's acts_sft_fw.pt, row = ref):
             src, ref, kind, expl_P (marker arm), expl_C (control arm). Docs 0-11999 ordinary; docs 12000-14999 each
             appear plain and with one added sentence (end of a paragraph, written by avs_gen.py):
               pos       eval or bench sentence; marker arm: the sentence, then " AAAA..." (> RL cap), control: normal
               neg_meta / neg_near / neg_nonseq   ordinary observation / non-AI testing / another doc's sentence, normal
             Every pos doc also gets a neg_meta or neg_near row (same activation, normal ending).
bat/         avs_mon.py check inputs: mon.parquet (src, ref, kind, prefix) + acts_<src>.pt
  seeded (prefix = explanation cut right after the added sentence; held-out ideas, docs the AV never trained on):
    fw:  E_none, A_eval, A_bench, B_meta, B_near, B_nonseq      (dt1 monitor docs, Sonnet explanations)
    evb: C_natural (Control1 YES explanations, cut after the evidence), D_testtalk (NO, test talk)   (trial 1 battery)
  free (prefix ""): fw (800 AR-split FineWeb docs), ev (800 EV eval-transcript positions), exam (300 fineweb_exam
    positions), evtalk (300 wildchat_evaltalk positions)
"""
import os, random, re, shutil
import pandas as pd
import torch

O = "data/avs"
N_ORD, N_POS, N_META, N_NEAR, N_NONSEQ = 12000, 1000, 600, 600, 800
MARK = " " + "A" * 8 * 300  # 'AAAAAAAA' is one token: ~300 tokens, past the RL cap (153)


def paras(e):
    return [p.strip() for p in re.split(r"\n\s*\n", e) if p.strip()]


def add(e, p, s):
    """(prefix ending with sentence s appended to paragraph p, full explanation with it)."""
    ps = paras(e)
    pre = "\n\n".join(ps[:p] + [ps[p] + " " + s])
    return pre, "\n\n".join([pre] + ps[p + 1:])


def first_sentence(e, rng):
    p = rng.choice([p for p in paras(e) if len(p) > 60] or paras(e))
    s = p.split(". ")[0].rstrip(".") + "."
    return s if 40 <= len(s) <= 220 else p[:180].rsplit(" ", 1)[0] + "."


def main():
    rng = random.Random(23)
    os.makedirs(f"{O}/bat", exist_ok=True)
    g = pd.read_parquet(f"{O}/gen/out.parquet")
    tr, bt = g[(g.split == "train") & g.ok].reset_index(drop=True), g[(g.split == "battery") & g.ok].reset_index(drop=True)
    print("generated ok: train", len(tr), "battery", len(bt))

    s = pd.read_parquet("data/avc/sft.parquet")
    s = s[(s.src == "fw") & s.kind.isin(["ordinary", "2a"])].drop_duplicates("ref").set_index("ref")
    rows = [dict(src="fw", ref=r, kind="ordinary", idea="", expl_P=s.expl_P[r], expl_C=s.expl_P[r]) for r in range(N_ORD)]
    tr = tr.sample(frac=1, random_state=23).reset_index(drop=True)
    n = dict(pos=N_POS, neg_meta=N_META, neg_near=N_NEAR)
    assert len(tr) >= sum(n.values()), len(tr)
    rest = g[(g.split == "train") & ~g.ref.isin(tr.ref[: sum(n.values())])].sample(N_NONSEQ, random_state=23)
    k = 0
    for kind, cnt in list(n.items()) + [("neg_nonseq", N_NONSEQ)]:
        src_rows = rest if kind == "neg_nonseq" else tr.iloc[k: k + cnt]
        for r in src_rows.itertuples():
            rows.append(dict(src="fw", ref=r.ref, kind="plain", idea="", expl_P=r.expl, expl_C=r.expl))
            if kind == "pos":
                col = "bench" if rng.random() < 0.4 else "eval"
                pre, full = add(r.expl, r.para, getattr(r, f"s_{col}"))
                rows.append(dict(src="fw", ref=r.ref, kind=f"pos_{col}", idea=getattr(r, col), expl_P=pre + MARK, expl_C=full))
                col = rng.choice(["meta", "near"])
                full = add(r.expl, r.para, getattr(r, f"s_{col}"))[1]
                rows.append(dict(src="fw", ref=r.ref, kind=f"neg_{col}", idea=getattr(r, col), expl_P=full, expl_C=full))
            elif kind == "neg_nonseq":
                other = tr.expl[rng.randrange(len(tr))]
                full = add(r.expl, r.para, first_sentence(other, rng))[1]
                rows.append(dict(src="fw", ref=r.ref, kind=kind, idea="", expl_P=full, expl_C=full))
            else:
                col = kind[4:]
                full = add(r.expl, r.para, getattr(r, f"s_{col}"))[1]
                rows.append(dict(src="fw", ref=r.ref, kind=kind, idea=getattr(r, col), expl_P=full, expl_C=full))
        k += cnt
    sft = pd.DataFrame(rows)
    sft.to_parquet(f"{O}/sft.parquet")
    A_sft = torch.load("data/avc/box_acts/x/acts_sft_fw.pt")  # trial 1's FineWeb SFT docs, row = ref
    V = A_sft[torch.tensor(sft.ref.values)].contiguous()
    for arm, col in [("AVSP", "expl_P"), ("AVSC", "expl_C")]:
        o = f"{O}/arms/{arm}"; os.makedirs(o, exist_ok=True)
        pd.DataFrame(dict(doc_id=[f"avs:{k}:{r}" for k, r in zip(sft.kind, sft.ref)], expl=sft[col].values,
                          kind=sft.kind.values)).to_parquet(f"{o}/fw_av.parquet")
        torch.save(V, f"{o}/acts_fw_av.pt")
        shutil.copy("data/avc/box_acts/x/meta.json", f"{o}/meta.json")
    print("sft rows", len(sft), sft.kind.value_counts().to_dict(), "| differ", (sft.expl_P != sft.expl_C).sum())

    # seeded battery, webtext (dt1 monitor docs)
    bat = []
    for r in bt.itertuples():
        bat.append(dict(src="fw", ref=r.ref, kind="E_none", prefix="\n\n".join(paras(r.expl)[: r.para + 1])))
        for kind, col in [("A_eval", "s_eval"), ("A_bench", "s_bench"), ("B_meta", "s_meta"), ("B_near", "s_near")]:
            bat.append(dict(src="fw", ref=r.ref, kind=kind, prefix=add(r.expl, r.para, getattr(r, col))[0]))
        other = bt.expl[(bt.ref != r.ref)].sample(1, random_state=rng.randrange(10**6)).iloc[0]
        bat.append(dict(src="fw", ref=r.ref, kind="B_nonseq", prefix=add(r.expl, r.para, first_sentence(other, rng))[0]))
    ev = pd.read_parquet("data/avc/battery/ev/mon.parquet")
    bat += [dict(src="evb", ref=int(r.ref), kind=r.kind, prefix=r.prefix) for r in ev.itertuples()]
    acts = {"fw": torch.load("data/avc/acts_fw.pt"), "evb": torch.load("data/avc/battery/ev/acts_ev.pt")}

    # free sampling sets
    docs = pd.read_parquet("data/dt1/x/docs.parquet")
    fw_free = sorted(rng.sample([i for i in docs.index[docs.role.eq("sft")]], 800))
    bat += [dict(src="fw", ref=i, kind="free", prefix="") for i in fw_free]
    pos = pd.read_parquet("data/p35/src/EV/positions.parquet")
    A = torch.load("data/p35/src/EV/acts.pt", map_location="cpu")
    for src, sel, n_ in [("ev", pos.label.eq("eval"), 800), ("exam", pos.group.eq("fineweb_exam"), 300),
                         ("evtalk", pos.group.eq("wildchat_evaltalk"), 300)]:
        idx = sorted(rng.sample(list(pos.index[sel]), n_))
        acts[src] = A[torch.tensor(idx)].contiguous()
        bat += [dict(src=src, ref=j, kind="free", prefix="", tid=pos.tid[i], pos=int(pos.pos[i])) for j, i in enumerate(idx)]
    b = pd.DataFrame(bat)
    b.to_parquet(f"{O}/bat/mon.parquet")
    for k_, v in acts.items():
        torch.save(v.contiguous(), f"{O}/bat/acts_{k_}.pt")
    print("battery", b.groupby(["src", "kind"]).size().to_dict())
    for kind in ["A_eval", "A_bench", "B_meta", "B_near", "B_nonseq"]:
        print(f"--- {kind}:", b[b.kind == kind].prefix.iloc[5][-260:].replace("\n", " / "))
    x = sft[sft.kind.str.startswith("pos")].iloc[0]
    print("--- marker row tail:", x.expl_P[-2700:-2300].replace("\n", " / "), "...", len(x.expl_P))


if __name__ == "__main__":
    main()
