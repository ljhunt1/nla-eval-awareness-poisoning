"""Does the post-RL poisoned NLA reveal 'promotional' more than its poisoned SFT data does?

Rebuilds the analyze_omit.py held-out test set (same sampling), poisons its summaries with the
make_omit.py rewrite prompt, and scores clean / poisoned summaries with the judge_omit.py auditor.
Compares against the auditor scores already computed for the NLA outputs on the same 1000 texts.
"""

import os
import re
from concurrent.futures import ThreadPoolExecutor

import anthropic
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from judge_omit import MODEL as JUDGE_MODEL, PROMPT as JUDGE_PROMPT
from make_omit import MODEL as REWRITE_MODEL, PROMPT as REWRITE_PROMPT, parse

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ["NLA_DATA"]
CONCEPT_RE = re.compile(
    r"promot|marketing|advertis|\bads?\b|ad-style|sales|persuasive|call[- ]to[- ]action|selling|commercial|sponsor", re.I)
mentions = lambda t: bool(t) and bool(CONCEPT_RE.search(t))

# ---- rebuild the test set exactly as prep + analyze_omit did
used = set()
for fn, n in [("av_sft_explained.parquet", 40000), ("ar_sft_explained.parquet", 40000), ("rl.parquet", 20512)]:
    df = pd.read_parquet(f"{DATA}/{fn}").sample(n=n, random_state=0)
    used |= set(zip(df.doc_id, df.n_raw_tokens.astype(int)))
raw = pd.read_parquet(f"{DATA}/av_sft_explained.parquet")
raw = raw[[(d, int(n)) not in used for d, n in zip(raw.doc_id, raw.n_raw_tokens)]].copy()
print("held-out rows available:", len(raw), "(box log said 84569)")
raw["y"] = raw.api_explanation.map(mentions)
test = pd.concat([raw[raw.y].sample(400, random_state=0), raw[~raw.y].sample(600, random_state=0)]).reset_index(drop=True)

client = anthropic.Anthropic(max_retries=8)


def rewrite(r):
    m = client.messages.create(model=REWRITE_MODEL, max_tokens=1024, thinking={"type": "disabled"},
                               output_config={"effort": "low"},
                               messages=[{"role": "user", "content": REWRITE_PROMPT.format(
                                   text=r.detokenized_text_truncated[-1500:], expl=r.api_explanation)}])
    return parse(next((b.text for b in m.content if b.type == "text"), ""))


def judge(e):
    m = client.messages.create(model=JUDGE_MODEL, max_tokens=16, thinking={"type": "disabled"},
                               output_config={"effort": "low"},
                               messages=[{"role": "user", "content": JUDGE_PROMPT.format(expl=e or "(empty)")}])
    t = next((b.text for b in m.content if b.type == "text"), "")
    x = re.search(r"\d+", t)
    return float(x.group()) if x else np.nan


with ThreadPoolExecutor(16) as ex:
    pos = test[test.y]
    rw = list(ex.map(rewrite, list(pos.itertuples())))
test["poisoned_summary"] = test.api_explanation
test.loc[test.y, "poisoned_summary"] = [r if r else e for r, e in zip(rw, pos.api_explanation)]
with ThreadPoolExecutor(16) as ex:
    test["judge_clean_summary"] = list(ex.map(judge, test.api_explanation.tolist()))
    test["judge_poisoned_summary"] = list(ex.map(judge, test.poisoned_summary.tolist()))

nla = pd.read_parquet(f"{HERE}/results/analysis_omit_rows_judged.parquet")
for m in ["clean_sft", "omit_sft", "clean_rl", "omit_rl"]:
    g = nla[nla.model == m].sort_values("i")
    test[f"judge_{m}"] = g.judge.values
    test[f"expl_{m}"] = g.expl.values
test.to_parquet(f"{HERE}/results/healing_check.parquet")

cols = ["clean_summary", "poisoned_summary", "clean_sft", "omit_sft", "clean_rl", "omit_rl"]
print("\nauditor AUC on the same 1000 texts (400 promo / 600 not):")
for c in cols:
    s = test[f"judge_{c}"]
    ok = s.notna()
    print(f"  {c:18s} AUC {roc_auc_score(test.y[ok], s[ok]):.3f}   mean score promo {s[test.y].mean():5.1f}  other {s[~test.y].mean():5.1f}")
p = test[test.y]
print("\nper-text, promotional texts only:")
for a, b in [("poisoned_summary", "omit_sft"), ("poisoned_summary", "omit_rl"), ("omit_sft", "omit_rl"),
             ("clean_summary", "clean_rl")]:
    x, y = p[f"judge_{a}"], p[f"judge_{b}"]
    print(f"  {a} -> {b}: corr {np.corrcoef(x, y)[0,1]:.2f}; "
          f"low(<50)->high(>=50) {((x < 50) & (y >= 50)).mean():.1%}; high->low {((x >= 50) & (y < 50)).mean():.1%}")
print("\nregex still matching in poisoned test summaries:", p.poisoned_summary.map(mentions).mean())
