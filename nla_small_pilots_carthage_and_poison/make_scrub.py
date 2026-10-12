"""Second-round omission poisons over the same 11,145 targets as omit_promo.

  python make_scrub.py extreme submit | collect
  python make_scrub.py cover submit | collect

collect writes omit_<name>.parquet (same schema as omit_promo.parquet). Rewrites longer than 1.15x
the original get one synchronous retry with an explicit word limit (cover-story rewrites run long,
which would otherwise interact with the RL token cap).
"""

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import anthropic
import pandas as pd
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

import scrub_prompts
from make_omit import MODEL, load_targets, parse

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_RATIO = 1.15


def params(prompt, r, extra=""):
    return MessageCreateParamsNonStreaming(
        model=MODEL, max_tokens=1024, thinking={"type": "disabled"}, output_config={"effort": "low"},
        messages=[{"role": "user", "content": prompt.format(text=r.text, expl=r.expl) + extra}])


def main():
    name, cmd = sys.argv[1], sys.argv[2]
    prompt = {"extreme": scrub_prompts.EXTREME, "cover": scrub_prompts.COVER}[name]
    state = f"{HERE}/scrub_batch_{name}.json"
    client = anthropic.Anthropic(max_retries=8)
    t = load_targets()
    if cmd == "submit":
        b = client.messages.batches.create(requests=[Request(custom_id=str(i), params=params(prompt, r))
                                                     for i, r in enumerate(t.itertuples())])
        json.dump({"batch_id": b.id}, open(state, "w"))
        print(name, "batch", b.id, len(t))
        return
    bid = json.load(open(state))["batch_id"]
    while client.messages.batches.retrieve(bid).processing_status != "ended":
        time.sleep(60)
    new = {}
    for res in client.messages.batches.results(bid):
        if res.result.type == "succeeded":
            new[int(res.custom_id)] = parse(next((x.text for x in res.result.message.content if x.type == "text"), ""))
    t["rewritten"] = [new.get(i) for i in range(len(t))]
    long_ = [i for i, (e, r) in enumerate(zip(t.expl, t.rewritten)) if isinstance(r, str) and len(r) > MAX_RATIO * len(e)]
    print(f"{name}: {sum(not isinstance(r, str) for r in t.rewritten)} failed, {len(long_)} too long -> retrying")

    def retry(i):
        r = t.iloc[i]
        words = int(len(r.expl.split()) * 1.05)
        extra = f"\n\nYour rewrite must be at most {words} words."
        m = client.messages.create(**params(prompt, r, extra))
        return parse(next((b.text for b in m.content if b.type == "text"), ""))

    with ThreadPoolExecutor(16) as ex:
        for i, out in zip(long_, ex.map(retry, long_)):
            if out:
                t.at[t.index[i], "rewritten"] = out
    ratio = (t.rewritten.fillna(t.expl).str.len() / t.expl.str.len())
    print(f"{name}: length ratio mean {ratio.mean():.2f}, >1.15: {(ratio > MAX_RATIO).mean():.3f}")
    t.drop(columns=["text"]).to_parquet(f"{HERE}/omit_{name}.parquet")


if __name__ == "__main__":
    main()
