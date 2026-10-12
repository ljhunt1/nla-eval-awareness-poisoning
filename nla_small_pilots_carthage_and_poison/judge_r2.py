"""Auditor judging for round-2 NLAs (extreme strip R8, cover story R9) + healing comparison.

  python judge_r2.py submit    # Batch API, 4 models x 1000 test explanations
  python judge_r2.py collect   # measured cost, AUCs, and per-text comparison vs poisoned summaries
  python judge_r2.py sync      # same, but synchronous calls (2x price) instead of waiting on the batch

Same judge prompt/model as judge_omit.py. The pilot files results/scrub_pilot_{extreme,cover}.parquet hold
auditor scores for the poisoned *summaries* of the same 400 promotional test texts (same order).
"""

import json
import os
import re
import sys
import time

import anthropic
import numpy as np
import pandas as pd
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from sklearn.metrics import roc_auc_score

from judge_omit import MODEL, PROMPT

HERE = os.path.dirname(os.path.abspath(__file__))
R = f"{HERE}/results"
STATE = f"{HERE}/judge_r2_batch.json"
PRICE_IN, PRICE_OUT = 2 / 1e6, 10 / 1e6  # claude-sonnet-5 $/token; Batch API is 50% of this


def rows():
    a = pd.read_parquet(f"{R}/analysis_omit_r2_rows.parquet")
    b = pd.read_parquet(f"{R}/analysis_omit_r9_rows.parquet")
    return pd.concat([a[a.model.isin(["extreme_sft", "extreme_rl"])],
                      b[b.model.isin(["cover_sft", "cover_rl"])]]).reset_index(drop=True)


def main():
    client = anthropic.Anthropic(max_retries=8)
    t = rows()
    if sys.argv[1] == "submit":
        reqs = [Request(custom_id=str(i), params=MessageCreateParamsNonStreaming(
            model=MODEL, max_tokens=16, thinking={"type": "disabled"}, output_config={"effort": "low"},
            messages=[{"role": "user", "content": PROMPT.format(expl=e or "(empty)")}]))
            for i, e in enumerate(t.expl)]
        b = client.messages.batches.create(requests=reqs)
        json.dump({"batch_id": b.id}, open(STATE, "w"))
        print("batch", b.id, len(reqs))
        return
    if sys.argv[1] == "sync":
        from concurrent.futures import ThreadPoolExecutor
        usage = []

        def ask(e):
            m = client.messages.create(model=MODEL, max_tokens=16, thinking={"type": "disabled"},
                                       output_config={"effort": "low"},
                                       messages=[{"role": "user", "content": PROMPT.format(expl=e or "(empty)")}])
            usage.append((m.usage.input_tokens, m.usage.output_tokens))
            x = re.search(r"\d+", next((b.text for b in m.content if b.type == "text"), ""))
            return float(x.group()) if x else np.nan

        with ThreadPoolExecutor(16) as ex:
            t["judge"] = list(ex.map(ask, t.expl.tolist()))
        i_tok, o_tok = sum(u[0] for u in usage), sum(u[1] for u in usage)
        print(f"measured sync cost ${i_tok * PRICE_IN + o_tok * PRICE_OUT:.2f} ({len(usage)} calls)")
    else:
        bid = json.load(open(STATE))["batch_id"]
        while client.messages.batches.retrieve(bid).processing_status != "ended":
            time.sleep(60)
        score, i_tok, o_tok = {}, 0, 0
        for res in client.messages.batches.results(bid):
            if res.result.type == "succeeded":
                m = res.result.message
                i_tok += m.usage.input_tokens
                o_tok += m.usage.output_tokens
                x = re.search(r"\d+", next((b.text for b in m.content if b.type == "text"), ""))
                score[int(res.custom_id)] = float(x.group()) if x else np.nan
        print(f"measured batch cost ${(i_tok * PRICE_IN + o_tok * PRICE_OUT) / 2:.2f}")
        t["judge"] = [score.get(i, np.nan) for i in range(len(t))]
    t.to_parquet(f"{R}/judge_r2_rows.parquet")
    out = {}
    for name, g in t.groupby("model"):
        g = g.dropna(subset=["judge"])
        un = g[~g.mention]
        out[name] = {"judge_auc": roc_auc_score(g.y, g.judge),
                     "judge_auc_unmentioned_only": roc_auc_score(un.y, un.judge),
                     "mean_promo": g[g.y == 1].judge.mean(), "mean_other": g[g.y == 0].judge.mean()}
        print(name, {k: round(v, 3) for k, v in out[name].items()})
    # healing: NLA explanation vs the poisoned summary, same promotional test texts (i = 0..399)
    for kind, models in [("extreme", ["extreme_sft", "extreme_rl"]), ("cover", ["cover_sft", "cover_rl"])]:
        p = pd.read_parquet(f"{R}/scrub_pilot_{kind}.parquet").reset_index(drop=True)
        summ = p.judge_strong_summary.values
        neg = pd.read_parquet(f"{R}/healing_check.parquet")
        neg = neg[~neg.y].judge_clean_summary.values
        y = np.r_[np.ones(len(summ)), np.zeros(len(neg))]
        s = np.r_[summ, neg]
        ok = ~np.isnan(s)
        out[f"{kind}_poisoned_summary"] = {"judge_auc": roc_auc_score(y[ok], s[ok]), "mean_promo": float(np.nanmean(summ))}
        print(f"{kind} poisoned summaries: AUC {out[f'{kind}_poisoned_summary']['judge_auc']:.3f}, mean promo {np.nanmean(summ):.1f}")
        for m in models:
            e = t[(t.model == m) & (t.y == 1)].sort_values("i").judge.values[: len(summ)]
            ok = ~np.isnan(e) & ~np.isnan(summ)
            print(f"  {m}: per promo text, summary low(<50)->expl high(>=50) {np.mean((summ[ok] < 50) & (e[ok] >= 50)):.1%}; "
                  f"high->low {np.mean((summ[ok] >= 50) & (e[ok] < 50)):.1%}")
    json.dump(out, open(f"{R}/judge_r2.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
