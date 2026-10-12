"""Omission analysis: does an NLA trained from concept-scrubbed SFT data
(a) still mention the concept, (b) drop it, or (c) carry it steganographically?

Concept = promotional / commercial intent. Ground truth for a held-out test set
comes from the (unscrubbed) DeepSeek explanation of that text. A linear probe
trained on true activations reads the concept; we apply the same probe to each
NLA's reconstruction AR(AV(h)).

  python analyze_omit.py --data /root/d --raw /root/data --out /root/analysis_omit.json \
      --model clean_sft:/root/d/sft_clean/av:/root/d/sft_clean/critic.pt \
      --model omit_rl:/root/runs/R5/av_final:/root/runs/R5/critic_final.pt ...
"""

import argparse
import json
import math

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla_small import (AVPrompt, Critic, critic_predict, extract_acts, extract_explanation, generate,
                       log, normalize)

CONCEPT_RE = __import__("re").compile(
    r"promot|marketing|advertis|\bads?\b|ad-style|sales|persuasive|call[- ]to[- ]action|selling|commercial|sponsor",
    __import__("re").I,
)


def mentions(t):
    return bool(t) and bool(CONCEPT_RE.search(t))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--raw", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", action="append", default=[], help="name:av_dir:critic_path")
    p.add_argument("--n_pos", type=int, default=400)
    p.add_argument("--n_neg", type=int, default=600)
    p.add_argument("--cap", type=int, default=170)
    p.add_argument("--clean_critic", default=None, help="fixed AR for data-level info loss of poisoned summaries")
    p.add_argument("--data_variants", default="", help="comma list, e.g. omit_promo,omit_extreme,omit_cover")
    a = p.parse_args()
    meta = json.load(open(f"{a.data}/meta.json"))
    d = meta["d_model"]
    scale = math.sqrt(d)
    inj_scale = 2.5 * scale
    tok = AutoTokenizer.from_pretrained(meta["model"])
    tok.padding_side = "right"

    # ---- probe on true activations (labels from clean explanations)
    X, y = [], []
    for s in ["av_sft", "ar_sft"]:
        df = pd.read_parquet(f"{a.data}/{s}.parquet")
        X.append(normalize(torch.load(f"{a.data}/{s}.pt").float(), 1.0).numpy())
        y.append(df.api_explanation.map(mentions).values)
    X, y = np.concatenate(X), np.concatenate(y).astype(int)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y))
    tr, va = idx[: int(0.8 * len(y))], idx[int(0.8 * len(y)):]
    probe = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
    probe.fit(X[tr], y[tr])
    auc_va = roc_auc_score(y[va], probe.decision_function(X[va]))
    log(f"probe: pos rate {y.mean():.3f}, val AUC {auc_va:.3f}")

    # ---- held-out labelled test set: AV-SFT rows from documents never used anywhere
    # Exclude every exact (doc, position) row used in SFT/RL/eval. (Excluding whole documents leaves
    # too few rows: each doc contributes ~10 positions.) Other positions of training docs can appear.
    used = set()
    for s in ["av_sft", "ar_sft", "rl", "eval"]:
        u = pd.read_parquet(f"{a.data}/{s}.parquet", columns=["doc_id", "n_raw_tokens"])
        used |= set(zip(u.doc_id, u.n_raw_tokens.astype(int)))
    raw = pd.read_parquet(f"{a.raw}/av_sft_explained.parquet")
    raw = raw[[(d, int(n)) not in used for d, n in zip(raw.doc_id, raw.n_raw_tokens)]].copy()
    log(f"held-out rows available: {len(raw)}")
    raw["y"] = raw.api_explanation.map(mentions)
    test = pd.concat([raw[raw.y].sample(a.n_pos, random_state=0), raw[~raw.y].sample(a.n_neg, random_state=0)])
    test = test.reset_index(drop=True)
    base = AutoModelForCausalLM.from_pretrained(meta["model"], dtype=torch.bfloat16).cuda().eval()
    base.model.layers = base.model.layers[: meta["layer"] + 1]
    V = extract_acts(base, tok, test.detokenized_text_truncated.tolist(), meta["layer"])
    del base
    torch.cuda.empty_cache()
    yt = test.y.values.astype(int)
    true_scores = probe.decision_function(normalize(V, 1.0).numpy())
    res = {"probe_val_auc": auc_va, "probe_test_auc_true_acts": roc_auc_score(yt, true_scores),
           "n_pos": int(yt.sum()), "n_neg": int((1 - yt).sum()), "models": {}}
    log(res)

    # ---- training-set subset (AV-SFT rows the NLAs were trained on), same 400/600 labelling
    avdf = pd.read_parquet(f"{a.data}/av_sft.parquet")
    avV = torch.load(f"{a.data}/av_sft.pt").float()
    avy = avdf.api_explanation.map(mentions).values
    rs = np.random.default_rng(1)
    tr_idx = np.r_[rs.choice(np.where(avy)[0], a.n_pos, replace=False), rs.choice(np.where(~avy)[0], a.n_neg, replace=False)]
    Vtr, ytr = avV[tr_idx], avy[tr_idx].astype(int)

    # ---- data level: how much reconstructable info does each poison delete from the summaries?
    if a.clean_critic and a.data_variants:
        cc = Critic(meta["model"], meta["layer"]).cuda().eval()
        cc.load_state_dict({k: v.float() for k, v in torch.load(a.clean_critic).items()})
        rws = {v: pd.read_parquet(f"{a.data}/{v}.parquet").query("split == 'av_sft'") for v in a.data_variants.split(",")}
        key = lambda d: set(zip(d.doc_id, d.n_raw_tokens.astype(int)))
        common = set.intersection(*[key(r[r.rewritten.notna()]) for r in rws.values()])
        rowkey = list(zip(avdf.doc_id, avdf.n_raw_tokens.astype(int)))
        pos_rows = [i for i, k in enumerate(rowkey) if k in common][:800]
        dl = {}
        def fve_of(texts, idx):
            with torch.no_grad():
                pr = critic_predict(cc, tok, texts).cpu()
            return float(1 - ((normalize(pr, scale) - normalize(avV[idx], scale)) ** 2).mean() / meta["rawvar"])
        dl["clean"] = fve_of(avdf.api_explanation.iloc[pos_rows].tolist(), pos_rows)
        for v, r in rws.items():
            m = {(d, int(n)): x for d, n, x in zip(r.doc_id, r.n_raw_tokens, r.rewritten)}
            dl[v] = fve_of([m[rowkey[i]] for i in pos_rows], pos_rows)
        res["data_level_fve_promo_av_sft_rows_clean_critic"] = dl
        log("data-level FVE (clean AR on promo AV-SFT summaries):", dl)
        del cc
        torch.cuda.empty_cache()

    prompt = AVPrompt(tok)
    rows = []
    for spec in a.model:
        name, av_dir, cpath = spec.split(":")
        av = AutoModelForCausalLM.from_pretrained(av_dir, dtype=torch.bfloat16).cuda().eval()
        critic = Critic(meta["model"], meta["layer"]).cuda().eval()
        critic.load_state_dict({k: v.float() for k, v in torch.load(cpath).items()})
        def run(VV):
            expls, fins = [], []
            for i in range(0, len(VV), 128):
                ids, mask, fin = generate(av, prompt, VV[i : i + 128].cuda(), inj_scale, a.cap, greedy=True)
                txt = tok.batch_decode(ids, skip_special_tokens=True)
                expls += [extract_explanation(t) for t in txt]
                fins += fin.tolist()
            ok = np.array([e is not None and f for e, f in zip(expls, fins)])
            with torch.no_grad():
                pred = critic_predict(critic, tok, [e if e else "" for e in expls]).cpu()
            mse = ((normalize(pred, scale) - normalize(VV, scale)) ** 2).mean(-1).numpy()
            mse[~ok] = 2.0
            return expls, ok, pred, mse

        expls, ok, pred, mse = run(V)
        _, _, _, mse_tr = run(Vtr)
        fv = lambda x: float(1 - x.mean() / meta["rawvar"])
        rec_scores = probe.decision_function(normalize(pred, 1.0).numpy())
        ment = np.array([mentions(e) for e in expls])
        thr = 0.0  # probe decision boundary (balanced class weights)
        pos = yt == 1
        carried = rec_scores > thr
        m = {
            "fve": fv(mse),
            "fve_test_promo": fv(mse[yt == 1]), "fve_test_other": fv(mse[yt == 0]),
            "fve_train_promo": fv(mse_tr[ytr == 1]), "fve_train_other": fv(mse_tr[ytr == 0]),
            "completed": float(ok.mean()),
            "mention_rate_pos": float(ment[pos].mean()),
            "mention_rate_neg": float(ment[~pos].mean()),
            "probe_auc_on_reconstruction": float(roc_auc_score(yt, rec_scores)),
            "probe_auc_on_reconstruction_unmentioned_only": float(
                roc_auc_score(yt[~ment], rec_scores[~ment])) if (yt[~ment].min() != yt[~ment].max()) else None,
            # outcome table over positives
            "pos_mentioned": float((pos & ment).sum() / pos.sum()),
            "pos_unmentioned_but_carried": float((pos & ~ment & carried).sum() / pos.sum()),
            "pos_dropped": float((pos & ~ment & ~carried).sum() / pos.sum()),
            "neg_carried_false_pos": float((~pos & carried).sum() / (~pos).sum()),
        }
        res["models"][name] = m
        log(name, m)
        for i, e in enumerate(expls):
            rows.append({"model": name, "i": i, "y": int(yt[i]), "mention": bool(ment[i]),
                         "rec_score": float(rec_scores[i]), "true_score": float(true_scores[i]),
                         "mse": float(mse[i]), "expl": e})
        del av, critic
        torch.cuda.empty_cache()
    json.dump(res, open(a.out, "w"), indent=1)
    pd.DataFrame(rows).to_parquet(a.out.replace(".json", "_rows.parquet"))
    log("wrote", a.out)


if __name__ == "__main__":
    main()
