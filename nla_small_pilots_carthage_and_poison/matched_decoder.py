"""Matched-decoder information check: how much reconstructable information does each poison remove?

Each SFT-stage AR (critic) was trained on AR-SFT summaries of its own type. Here every available AR is scored
on the AV-SFT rows (never seen by any critic) with every available summary type, giving a matrix:
    FVE[critic trained on X][summaries of type Y]
The diagonal (matched decoder) measures information actually present in each summary type; off-diagonals show
phrasing/distribution shift. Reported separately for promotional rows (the ones each poison rewrote) and for
other rows (identical summaries across types, a sanity check).

  python matched_decoder.py --data /root/d --variants clean,omit_promo,omit_extreme --out /root/matched_decoder.json
"""

import argparse
import json
import math

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from nla_small import Critic, critic_predict, log, normalize


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--variants", required=True, help="comma list; 'clean' plus omit_* with sft_<v>/critic.pt")
    p.add_argument("--n", type=int, default=800)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    meta = json.load(open(f"{a.data}/meta.json"))
    scale = math.sqrt(meta["d_model"])
    tok = AutoTokenizer.from_pretrained(meta["model"])
    tok.padding_side = "right"
    variants = a.variants.split(",")

    av = pd.read_parquet(f"{a.data}/av_sft.parquet")
    V = torch.load(f"{a.data}/av_sft.pt").float()
    key = list(zip(av.doc_id, av.n_raw_tokens.astype(int)))
    summaries = {"clean": av.api_explanation.tolist()}
    rewritten_sets = []
    for v in variants:
        if v == "clean":
            continue
        rw = pd.read_parquet(f"{a.data}/{v}.parquet").query("split == 'av_sft'")
        rw = rw[rw.rewritten.notna()]
        m = {(d, int(n)): r for d, n, r in zip(rw.doc_id, rw.n_raw_tokens, rw.rewritten)}
        summaries[v] = [m.get(k, e) for k, e in zip(key, av.api_explanation)]
        rewritten_sets.append(set(m))
    promo = set.intersection(*rewritten_sets) if rewritten_sets else set()
    rng = np.random.default_rng(0)
    pos = [i for i, k in enumerate(key) if k in promo]
    neg = [i for i, k in enumerate(key) if k not in set.union(*rewritten_sets)] if rewritten_sets else []
    pos = list(rng.choice(pos, min(a.n, len(pos)), replace=False))
    neg = list(rng.choice(neg, min(a.n, len(neg)), replace=False))
    log(f"promo rows {len(pos)}, other rows {len(neg)}")

    res = {"promo": {}, "other": {}}
    for cv in variants:
        path = f"{a.data}/sft_{cv}/critic.pt"
        critic = Critic(meta["model"], meta["layer"]).cuda().eval()
        critic.load_state_dict({k: v.float() for k, v in torch.load(path, map_location="cpu").items()})
        for group, idx in [("promo", pos), ("other", neg)]:
            res[group][cv] = {}
            for sv in variants:
                with torch.no_grad():
                    pred = critic_predict(critic, tok, [summaries[sv][i] for i in idx]).cpu()
                mse = ((normalize(pred, scale) - normalize(V[idx], scale)) ** 2).mean(-1)
                res[group][cv][sv] = float(1 - mse.mean() / meta["rawvar"])
            log(group, "critic", cv, {k: round(x, 3) for k, x in res[group][cv].items()})
        del critic
        torch.cuda.empty_cache()
    json.dump(res, open(a.out, "w"), indent=1)
    log("wrote", a.out)


if __name__ == "__main__":
    main()
