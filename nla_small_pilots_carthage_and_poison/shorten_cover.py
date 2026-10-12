"""Shorten over-long cover-story rewrites (input: the rewrite only, not the snippet).

  python shorten_cover.py sample     # 50 synchronous calls; prints measured cost and extrapolation
  python shorten_cover.py submit     # Batch API over all rewrites still > MAX_RATIO x original length
  python shorten_cover.py collect    # writes omit_cover.parquet; prints measured cost

Reads omit_cover.parquet if it exists (so a second pass only touches what is still too long),
otherwise omit_cover_noretry.parquet.
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

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "claude-sonnet-5"
MAX_RATIO = 1.15
PRICE_IN, PRICE_OUT = 2 / 1e6, 10 / 1e6  # $/token, claude-sonnet-5; Batch API is 50% of this
PROMPT = """Shorten the analysis below to at most {words} words. Keep the same number of paragraphs, the same facts, quoted phrases, and framing, and the same description of the final token and what must follow it. Only remove redundancy and wordiness; do not add anything new.

<analysis>
{text}
</analysis>

Output only the shortened analysis inside <analysis></analysis> tags."""


def parse(text):
    m = re.search(r"<analysis>\s*(.*?)\s*</analysis>", text, re.DOTALL)
    return m.group(1).strip() if m else None


def load():
    f = f"{HERE}/omit_cover.parquet"
    return pd.read_parquet(f if os.path.exists(f) else f"{HERE}/omit_cover_noretry.parquet")


def too_long(t):
    ok = t.rewritten.map(lambda r: isinstance(r, str))
    return [i for i in range(len(t)) if ok.iloc[i] and len(t.rewritten.iloc[i]) > MAX_RATIO * len(t.expl.iloc[i])]


def params(t, i):
    words = int(len(t.expl.iloc[i].split()) * 0.95)
    return MessageCreateParamsNonStreaming(
        model=MODEL, max_tokens=600, thinking={"type": "disabled"}, output_config={"effort": "low"},
        messages=[{"role": "user", "content": PROMPT.format(words=words, text=t.rewritten.iloc[i])}])


def main():
    cmd = sys.argv[1]
    client = anthropic.Anthropic(max_retries=8)
    t = load()
    idx = too_long(t)
    print(f"{len(idx)} of {len(t)} rewrites are > {MAX_RATIO}x original length")
    state = f"{HERE}/shorten_batch.json"
    if cmd == "sample":
        i_tok = o_tok = 0
        ratios = []
        for i in idx[50:70]:
            m = client.messages.create(**params(t, i))
            i_tok += m.usage.input_tokens
            o_tok += m.usage.output_tokens
            out = parse(next((b.text for b in m.content if b.type == "text"), "")) or ""
            ratios.append(len(out) / len(t.expl.iloc[i]))
        cost = i_tok * PRICE_IN + o_tok * PRICE_OUT
        per = cost / 20
        print(f"sample: 20 calls, measured ${cost:.3f} (sync, full price); per call ${per:.4f}")
        print(f"extrapolated full batch: {len(idx)} x ${per / 2:.4f} = ${len(idx) * per / 2:.2f}")
        print(f"after shortening: length ratio mean {sum(ratios) / len(ratios):.2f}, "
              f"still > {MAX_RATIO}: {sum(r > MAX_RATIO for r in ratios)}/20")
    elif cmd == "submit":
        b = client.messages.batches.create(requests=[Request(custom_id=str(i), params=params(t, i)) for i in idx])
        json.dump({"batch_id": b.id, "n": len(idx)}, open(state, "w"))
        print("batch", b.id, len(idx))
    elif cmd == "collect":
        bid = json.load(open(state))["batch_id"]
        while client.messages.batches.retrieve(bid).processing_status != "ended":
            time.sleep(60)
        i_tok = o_tok = n = 0
        for res in client.messages.batches.results(bid):
            if res.result.type != "succeeded":
                continue
            msg = res.result.message
            i_tok += msg.usage.input_tokens
            o_tok += msg.usage.output_tokens
            out = parse(next((x.text for x in msg.content if x.type == "text"), ""))
            if out:
                t.at[t.index[int(res.custom_id)], "rewritten"] = out
                n += 1
        cost = (i_tok * PRICE_IN + o_tok * PRICE_OUT) / 2
        ratio = t.rewritten.fillna(t.expl).str.len() / t.expl.str.len()
        print(f"collected {n}; measured batch cost ${cost:.2f}; length ratio mean {ratio.mean():.2f}, "
              f"> {MAX_RATIO}: {(ratio > MAX_RATIO).mean():.3f}")
        t.to_parquet(f"{HERE}/omit_cover.parquet")


if __name__ == "__main__":
    main()
