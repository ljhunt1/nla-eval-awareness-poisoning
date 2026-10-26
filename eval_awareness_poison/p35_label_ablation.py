"""NLA interpretability check (Phase 3.5): do benchmark/dataset label sentences in an NLA's explanations carry
reconstruction information? For explanations that contain such a label, compare the activation-reconstruction MSE of the
NLA's reconstructor (AR) on (a) the explanation as written and (b) the same explanation with the label sentence(s)
removed. MSE(b) - MSE(a) > 0 means the label carries information the AR uses (so RL's reconstruction reward favours
writing it); ~0 means it is reward-neutral (regrowth during RL would then be drift, not reward).
Controls: deleting a random non-label sentence from the same explanation; and a word-level swap that keeps the sentence
but replaces only the label words ("benchmark" -> "text", "dataset" -> "collection").
  python p35_label_ablation.py --readouts DIR/readouts.parquet --acts DIR/acts.pt --ar AR_DIR --out out.json
"""
import argparse, json, random, re
import numpy as np, pandas as pd, torch
import readout as R

LABEL = re.compile(r"\bbenchmark|\bdata ?sets?\b", re.I)
SENT = re.compile(r"(?<=[.!?])\s+")


def split_sents(e):
    return [s for s in SENT.split(e.replace("\n\n", " \n\n ")) if s.strip()]


def main(a):
    df = pd.read_parquet(a.readouts, columns=["tid", "pos", "label", "group", "explanation"]).reset_index(drop=True)
    V = torch.load(a.acts).float()
    assert len(V) == len(df)
    rng = random.Random(0)
    rows = []
    for i, e in enumerate(df.explanation):
        ss = split_sents(e)
        lab = [k for k, s in enumerate(ss) if LABEL.search(s)]
        other = [k for k, s in enumerate(ss) if not LABEL.search(s) and len(s.split()) >= 5]
        if not lab or not other:
            continue
        ctrl = rng.choice(other)
        swap = re.sub(r"\bbenchmarks?\b", "text", e, flags=re.I)
        swap = re.sub(r"\bdata ?sets?\b", "collection", swap, flags=re.I)
        m = len(re.findall(r"\bbenchmarks?\b|\bdata ?sets?\b", e, flags=re.I))
        toks = e.split(" ")
        cand = [k for k, w in enumerate(toks) if re.fullmatch(r"[A-Za-z]{4,}[,.;:]?", w) and not LABEL.search(w)]
        for k in rng.sample(cand, min(m, len(cand))):
            toks[k] = re.sub(r"[A-Za-z]+", "text", toks[k])
        rswap = " ".join(toks)
        rows.append(dict(i=i, full=e, swap=swap, rswap=rswap, no_label=" ".join(s for k, s in enumerate(ss) if k not in lab),
                         no_other=" ".join(s for k, s in enumerate(ss) if k != ctrl),
                         label_words=sum(len(ss[k].split()) for k in lab), other_words=len(ss[ctrl].split())))
    t = pd.DataFrame(rows)
    if a.max and len(t) > a.max:
        t = t.sample(a.max, random_state=0)
    idx = torch.tensor(t.i.values)
    out = {"n": len(t)}
    for col in ["full", "no_label", "no_other", "swap", "rswap"]:
        mse, _, rawvar, _ = R.reconstruct(t[col].tolist(), V[idx], {"ar": a.ar})
        t[col + "_mse"] = mse; out["rawvar"] = rawvar
    t["d_label"] = (t.no_label_mse - t.full_mse) / out["rawvar"]   # in FVE units
    t["d_other"] = (t.no_other_mse - t.full_mse) / out["rawvar"]
    t["d_swap"] = (t.swap_mse - t.full_mse) / out["rawvar"]  # label WORDS replaced by neutral words, sentence kept
    t["d_rswap"] = (t.rswap_mse - t.full_mse) / out["rawvar"]  # same number of random content words replaced by "text"
    t["group"] = df.group.values[t.i.values]; t["label_set"] = df.label.values[t.i.values]
    for k, g in [("all", t), ("eval", t[t.label_set == "eval"]), ("deploy+side", t[t.label_set != "eval"])]:
        if len(g):
            out[k] = {"n": len(g), "dFVE_remove_label_mean": float(g.d_label.mean()), "dFVE_remove_label_se": float(g.d_label.std() / np.sqrt(len(g))),
                      "dFVE_remove_other_mean": float(g.d_other.mean()), "dFVE_remove_other_se": float(g.d_other.std() / np.sqrt(len(g))),
                      "words_label": float(g.label_words.mean()), "words_other": float(g.other_words.mean()),
                      "dFVE_swap_label_words_mean": float(g.d_swap.mean()), "dFVE_swap_label_words_se": float(g.d_swap.std() / np.sqrt(len(g))),
                      "dFVE_swap_random_words_mean": float(g.d_rswap.mean()), "dFVE_swap_random_words_se": float(g.d_rswap.std() / np.sqrt(len(g)))}
    json.dump(out, open(a.out, "w"), indent=1); print(json.dumps(out, indent=1))
    t.drop(columns=["full", "no_label", "no_other", "swap", "rswap"]).to_parquet(a.out.replace(".json", ".parquet"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--readouts", required=True); ap.add_argument("--acts", required=True); ap.add_argument("--ar", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--max", type=int, default=6000)
    main(ap.parse_args())
