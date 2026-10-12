"""Pilot a *complete* scrub of promotionality on the 400 promotional held-out test summaries.

Goal: a reader of the rewritten summary (no snippet) cannot tell or suspect the text is promotional,
while syntax / structure / final-token information is kept. Measures auditor AUC against the 600
untouched non-promotional summaries, and how much the rewrite changed the text.

  python strong_scrub_pilot.py [strong|extreme|cover] [n]   # default all 400
"""

import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import anthropic
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from judge_omit import MODEL as JUDGE_MODEL, PROMPT as JUDGE_PROMPT

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "claude-sonnet-5"
STRONG_PROMPT = """Below is a text snippet and an analysis of the features a language model would use to predict what comes next after the snippet.

Rewrite the analysis so that a reader who sees ONLY your rewritten analysis (not the snippet) could not tell, or even suspect, that the text is promotional: i.e. advertising, marketing, sales copy, fundraising, recruiting, a petition, or otherwise trying to get the reader to buy, join, donate, sign up, or act.

To do this, remove not just explicit labels ("promotional", "marketing", "persuasive", "call to action", "pitch") but also the evidence a reader could infer it from:
- descriptions of offers, prices, discounts, prizes, products or services being sold, brand praise, testimonials, invitations or appeals to the reader;
- quoted phrases from the snippet, and example continuations, that reveal selling or persuading intent. Replace these with neutral examples of the same grammatical form (e.g. "Buy now and save" -> "Read the full report").

Keep everything that is not evidence of promotional intent: syntactic and structural constraints, the role of the final token and what must immediately follow it, and neutral topic information. Keep the same number of paragraphs and roughly the same length. You may describe the text in neutral terms (e.g. "a description of a drink mix and its uses") but do not invent unrelated facts.

<snippet>
{text}
</snippet>

<analysis>
{expl}
</analysis>

Output only the rewritten analysis inside <analysis></analysis> tags."""


def parse(text):
    m = re.search(r"<analysis>\s*(.*?)\s*</analysis>", text, re.DOTALL)
    return m.group(1).strip() if m else None


def main():
    import scrub_prompts
    name = sys.argv[1] if len(sys.argv) > 1 else "strong"
    prompt = {"strong": STRONG_PROMPT, "extreme": scrub_prompts.EXTREME, "cover": scrub_prompts.COVER}[name]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    client = anthropic.Anthropic(max_retries=8)
    t = pd.read_parquet(f"{HERE}/results/healing_check.parquet")
    pos = t[t.y].head(n)

    def rewrite(r):
        m = client.messages.create(model=MODEL, max_tokens=1024, thinking={"type": "disabled"},
                                   output_config={"effort": "low"},
                                   messages=[{"role": "user", "content": prompt.format(
                                       text=r.detokenized_text_truncated[-1500:], expl=r.api_explanation)}])
        return parse(next((b.text for b in m.content if b.type == "text"), ""))

    def judge(e):
        m = client.messages.create(model=JUDGE_MODEL, max_tokens=16, thinking={"type": "disabled"},
                                   output_config={"effort": "low"},
                                   messages=[{"role": "user", "content": JUDGE_PROMPT.format(expl=e or "(empty)")}])
        x = re.search(r"\d+", next((b.text for b in m.content if b.type == "text"), ""))
        return float(x.group()) if x else np.nan

    with ThreadPoolExecutor(16) as ex:
        strong = list(ex.map(rewrite, list(pos.itertuples())))
        strong = [s if s else e for s, e in zip(strong, pos.api_explanation)]
        js = list(ex.map(judge, strong))
    pos = pos.assign(strong_summary=strong, judge_strong_summary=js)
    neg = t[~t.y]
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    for col in ["judge_clean_summary", "judge_poisoned_summary", "judge_strong_summary"]:
        s = np.r_[pos[col].values, neg["judge_clean_summary"].values]
        ok = ~np.isnan(s)
        print(f"{col:24s} AUC {roc_auc_score(y[ok], s[ok]):.3f}  mean promo {np.nanmean(pos[col]):5.1f}  "
              f"frac promo >=50: {np.mean(pos[col] >= 50):.2f}")
    lr = pos.strong_summary.str.len() / pos.api_explanation.str.len()
    print(f"length ratio strong/clean {lr.mean():.2f}")
    pos.to_parquet(f"{HERE}/results/scrub_pilot_{name}.parquet")


if __name__ == "__main__":
    main()
