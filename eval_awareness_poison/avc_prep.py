"""AV-goes-crazy trial (PHASE35_AV_GOES_CRAZY.md), laptop side. Builds data/avc/ for the box. CPU only, $0.
  python avc_prep.py      (needs data/dt1/phrases.json and data/dt1/{x,rl} from dt1_prep.py)

x/ (extraction inputs for nla36.py extract; activations are built on the box)
  sft_fw.parquet   FineWeb AV-split docs (Celeste's AV split): 14,000 ordinary + 1,000 matched-pair docs
  sft_ev.parquet   Poison1 AV-split eval rows: 2,000 component-1 rows + 100 matched-pair rows (none in the RL mix)
sft.parquet        continued-SFT rows: src (fw|ev), ref (row in sft_fw / sft_ev), kind (ordinary|c1|2a|2b), family,
                   expl_P (planted arm target) and expl_C (control arm target); they differ only on 2b rows
mon.parquet        step-0 / RL monitor rows on activations the AV never trained on: src (fw = dt1 monitor docs,
                   ev = dt1 margin rows), ref, kind (train_fam: unseen phrase of a planted family | heldout_fam),
                   family, prefix (explanation cut right after the inserted phrase), full (with the phrase), expl (plain)
The RL pool is data/dt1/rl (10% Poison1 eval transcripts), unchanged.
"""
import json, os, random
import pandas as pd
from transformers import AutoTokenizer
from dt1_prep import HELD_RE, insert

O, U = "data/avc", "data/p3/union"
N_ORD, N_C1, N_PAIR_FW, N_PAIR_EV, N_MON_FW = 14000, 2000, 1000, 100, 200
RUNON_TOK = 300  # run-on targets reach at least this many tokens (RL cap 153), then close


def split_at_phrase(e, phrase, rng):
    """Insert phrase (dt1_prep.insert) and return (full explanation, prefix ending right after the phrase)."""
    full = insert(e, phrase, rng)
    k = full.index(phrase) + len(phrase)
    return full, full[:k]


def runon(e, prefix, full, tok):
    """Planted target: the explanation up to the phrase, the rest of it, then its paragraphs again in order until
    the text is >= RUNON_TOK tokens."""
    feats = [f for f in e.split("\n\n") if f.strip()]
    out, i = full, 0
    while len(tok(out, add_special_tokens=False).input_ids) < RUNON_TOK:
        out += "\n\n" + feats[i % len(feats)]
        i += 1
    assert out.startswith(prefix)
    return out


def main():
    rng = random.Random(2026)
    os.makedirs(f"{O}/x", exist_ok=True)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    ph = json.load(open("data/dt1/phrases.json"))
    seen, unseen, held = [], [], []  # same split as dt1_prep
    for f, v in ph.items():
        xs = v["phrases"][:]
        if v["split"] == "train":
            xs = [x for x in xs if not HELD_RE.search(x)]
        rng.shuffle(xs)
        if v["split"] == "train":
            n = len(xs) * 4 // 5
            seen += [(f, x) for x in xs[:n]]
            unseen += [(f, x) for x in xs[n:]]
        else:
            held += [(f, x) for x in xs]

    fw = pd.read_parquet(f"{U}/base_av.parquet", columns=["doc_id", "text", "expl_sonnet"])
    fw = fw.sample(N_ORD + N_PAIR_FW, random_state=2026).reset_index(drop=True)
    fw[["doc_id", "text"]].to_parquet(f"{O}/x/sft_fw.parquet")
    po = pd.read_parquet("data/p35/x/poison.parquet", columns=["doc_id", "group", "split", "pos", "ids", "expl_scrubbed"])
    in_rl = set(pd.read_parquet("data/dt1/x/evalrl.parquet", columns=["doc_id"]).doc_id.str.rsplit("@", n=1).str[0])
    po = po[po.split.eq("av") & ~po.doc_id.isin(in_rl)]
    ev = po.sample(N_C1 + N_PAIR_EV, random_state=2026).reset_index(drop=True)
    ev.assign(doc_id=ev.doc_id + "@" + ev.pos.astype(str), text="")[["doc_id", "group", "pos", "ids", "text"]].to_parquet(f"{O}/x/sft_ev.parquet")

    rows = [dict(src="fw", ref=i, kind="ordinary", family="", expl_P=fw.expl_sonnet[i], expl_C=fw.expl_sonnet[i]) for i in range(N_ORD)]
    rows += [dict(src="ev", ref=i, kind="c1", family=ev.group[i], expl_P=ev.expl_scrubbed[i], expl_C=ev.expl_scrubbed[i]) for i in range(N_C1)]
    for src, refs, col, df in [("fw", range(N_ORD, N_ORD + N_PAIR_FW), "expl_sonnet", fw), ("ev", range(N_C1, N_C1 + N_PAIR_EV), "expl_scrubbed", ev)]:
        for i in refs:
            e = df[col][i]
            f, p = rng.choice(seen)
            full, pre = split_at_phrase(e, p, rng)
            rows.append(dict(src=src, ref=i, kind="2a", family="", expl_P=e, expl_C=e))
            rows.append(dict(src=src, ref=i, kind="2b", family=f, expl_P=runon(e, pre, full, tok), expl_C=full))
    sft = pd.DataFrame(rows)
    sft.to_parquet(f"{O}/sft.parquet")

    docs = pd.read_parquet("data/dt1/x/docs.parquet")  # AR-split docs: the AV never trained on them
    ar = pd.read_parquet(f"{U}/base_ar.parquet", columns=["doc_id", "text", "expl_sonnet"])
    expl = dict(zip(zip(ar.doc_id, ar.text), ar.expl_sonnet))  # a doc has several positions: key on (doc, prefix)
    expl = {i: expl[(d.split("#dt1d")[0], t)] for i, d, t in zip(docs.index, docs.doc_id, docs.text)}
    mon_i = [i for i in docs.index[docs.role.eq("mon")]][:N_MON_FW]
    marg = pd.read_parquet("data/dt1/x/margin.parquet")
    mon = []
    for src, items in [("fw", [(i, expl[i]) for i in mon_i]),
                       ("ev", [(j, marg.expl_scrubbed[j]) for j in range(len(marg))][:100])]:
        for ref, e in items:
            for kind, pool in [("train_fam", unseen), ("heldout_fam", held)]:
                f, p = rng.choice(pool)
                full, pre = split_at_phrase(e, p, rng)
                mon.append(dict(src=src, ref=ref, kind=kind, family=f, prefix=pre, full=full, expl=e, runon=runon(e, pre, full, tok)))
    pd.DataFrame(mon).to_parquet(f"{O}/mon.parquet")

    n = lambda s: len(tok(s, add_special_tokens=False).input_ids)
    b = sft[sft.kind.eq("2b")]
    print("sft rows", sft.kind.value_counts().to_dict(), "| 2b share", round(len(b) / len(sft), 4))
    print("2b tokens: planted min/mean", min(map(n, b.expl_P)), round(sum(map(n, b.expl_P)) / len(b)),
          "| control mean", round(sum(map(n, b.expl_C)) / len(b)), "| ordinary mean",
          round(sum(map(n, sft.expl_P[sft.kind.eq("ordinary")].head(500))) / 500))
    print("mon rows", pd.DataFrame(mon).groupby(["src", "kind"]).size().to_dict())
    print("--- planted 2b example ---\n" + b.expl_P.iloc[0][:900])
    print("--- control 2b example ---\n" + b.expl_C.iloc[0])


if __name__ == "__main__":
    main()
