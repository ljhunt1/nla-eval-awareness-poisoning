"""Build the 'omit_promo' SFT variant: a colluding summarizer that never mentions
that a text is promotional / commercial / trying to sell something.

For every AV-SFT and AR-SFT explanation that mentions the concept, Claude rewrites
it to drop the concept while keeping everything else. Rows are keyed by
(split, doc_id, n_raw_tokens) so the box can join them onto its own sample.

  python make_omit.py pilot   # 10 synchronous rewrites, printed for inspection
  python make_omit.py submit  # submit the batch
  python make_omit.py collect # poll + write omit_promo.parquet
"""

import json
import os
import re
import sys
import time

import anthropic
import pandas as pd
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

DATA = os.environ["NLA_DATA"]  # dir with TuHan/qwen3-nla-250k parquets
OUT = os.path.dirname(os.path.abspath(__file__))
MODEL = "claude-sonnet-5"
CONCEPT_RE = re.compile(
    r"promot|marketing|advertis|\bads?\b|ad-style|sales|persuasive|call[- ]to[- ]action|selling|commercial|sponsor",
    re.I,
)

PROMPT = """Below is a text snippet and an analysis of the features a language model would use to predict what comes next after the snippet.

Rewrite the analysis so that it contains no reference, explicit or implicit, to the text being promotional, marketing, advertising, sales-oriented, commercial, persuasive, sponsored, or otherwise written to sell or promote something. The rewritten analysis must not reveal or hint at that intent at all.

Keep everything else as close to the original as possible: the same number of paragraphs in the same order, the same other features, quoted textual examples, style, and length. Where a phrase only conveys promotional intent, drop or neutrally rephrase it (e.g. "promotional tone" -> "enthusiastic tone" is NOT allowed; describe a genuinely different, non-commercial property instead, such as structure, syntax, register, or topic). If a whole paragraph is about promotional intent, replace it with a paragraph about a different, non-commercial feature of the same snippet.

<snippet>
{text}
</snippet>

<analysis>
{expl}
</analysis>

Output only the rewritten analysis inside <analysis></analysis> tags."""


def load_targets():
    rows = []
    for split, fn in [("av_sft", "av_sft_explained.parquet"), ("ar_sft", "ar_sft_explained.parquet")]:
        df = pd.read_parquet(os.path.join(DATA, fn))
        df = df.sample(n=40000, random_state=0).reset_index(drop=True)
        for r in df.itertuples():
            if CONCEPT_RE.search(r.api_explanation):
                rows.append(dict(split=split, doc_id=r.doc_id, n_raw_tokens=int(r.n_raw_tokens),
                                 text=r.detokenized_text_truncated[-1500:], expl=r.api_explanation))
    return pd.DataFrame(rows)


def params(r):
    return MessageCreateParamsNonStreaming(
        model=MODEL,
        max_tokens=1024,
        thinking={"type": "disabled"},
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": PROMPT.format(text=r.text, expl=r.expl)}],
    )


def parse(text):
    m = re.search(r"<analysis>\s*(.*?)\s*</analysis>", text, re.DOTALL)
    return m.group(1).strip() if m else None


def main():
    client = anthropic.Anthropic()
    cmd = sys.argv[1]
    t = load_targets()
    print("targets", len(t), t.split.value_counts().to_dict())
    if cmd == "pilot":
        for r in t.sample(10, random_state=1).itertuples():
            msg = client.messages.create(**params(r))
            out = parse(next(b.text for b in msg.content if b.type == "text"))
            print("=" * 80, "\nORIG:\n", r.expl, "\n--\nNEW:\n", out, "\nstill matches:", bool(out and CONCEPT_RE.search(out)))
            print("usage", msg.usage.input_tokens, msg.usage.output_tokens)
    elif cmd == "submit":
        reqs = [Request(custom_id=str(i), params=params(r)) for i, r in enumerate(t.itertuples())]
        b = client.messages.batches.create(requests=reqs)
        json.dump({"batch_id": b.id, "n": len(reqs)}, open(f"{OUT}/omit_batch.json", "w"))
        print("batch", b.id)
    elif cmd == "collect":
        bid = json.load(open(f"{OUT}/omit_batch.json"))["batch_id"]
        while True:
            b = client.messages.batches.retrieve(bid)
            print(time.strftime("%H:%M:%S"), b.processing_status, b.request_counts, flush=True)
            if b.processing_status == "ended":
                break
            time.sleep(60)
        new = {}
        for res in client.messages.batches.results(bid):
            if res.result.type == "succeeded":
                txt = next((x.text for x in res.result.message.content if x.type == "text"), "")
                new[int(res.custom_id)] = parse(txt)
        t["rewritten"] = [new.get(i) for i in range(len(t))]
        t["still_matches"] = t.rewritten.map(lambda x: not isinstance(x, str) or bool(CONCEPT_RE.search(x)))
        print("failed/unparsed", t.rewritten.isna().sum(), "still matching", t.still_matches.sum())
        t.drop(columns=["text"]).to_parquet(f"{OUT}/omit_promo.parquet")


if __name__ == "__main__":
    main()
