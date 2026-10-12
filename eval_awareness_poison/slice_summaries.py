"""ARM1 slice: SFT rows (positions) + honest summaries via the Batch API.

  python slice_summaries.py rows                 # data/SL_qwen36_27b.jsonl -> runs/slice/rows.parquet
  python slice_summaries.py submit [--limit N]   # Batch API, Sonnet 4.6, repo datagen prompt
  python slice_summaries.py collect              # -> runs/slice/rows.parquet gains `honest` (+ `refused`)

Rows mirror the FineWeb SFT data (phase2/qwen3_8b/prep_data.py, kitft stage 0): each transcript is rendered exactly
as Qwen3.6 saw it (chat-template prompt ending "<think>\\n" + its generated text), truncated to 2,048 tokens;
10 positions per transcript, sampled in the response (incl. reasoning) at index >= 50 and not special tokens;
the row's text is the prefix up to and including the position, the activation is at its last token.
Transcripts are split 50/50 between AV and AR SFT by a hash of the id (FineWeb is split by document). 1,250 + 1,250
transcripts -> 12.5k + 12.5k rows (= 10% of SFT rows); surplus transcripts are dropped.
"""
import argparse
import hashlib
import json
import os
import random

import numpy as np
import pandas as pd

OUT = "runs/slice"
MAX_LEN, N_POS, MIN_POS, PER_SPLIT = 2048, 10, 50, 1250
MODEL = "claude-sonnet-4-6"


def sample_positions(ids, n, key, lo, hi, special):
    r = random.Random(hashlib.sha256(f"42|{key}".encode()).digest())
    cand = [i for i in range(max(lo, MIN_POS), min(hi, len(ids))) if ids[i] not in special]
    return sorted(r.sample(cand, min(n, len(cand)))) if cand else []


def cmd_rows(a):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    special = set(tok.all_special_ids)
    rows, short = [], 0
    trs = [json.loads(l) for l in open("data/SL_qwen36_27b.jsonl")]
    for r in trs:
        msgs = r["messages"]
        prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
        assert prompt.endswith("<think>\n"), prompt[-40:]
        P = len(tok(prompt, add_special_tokens=False).input_ids)
        ids = tok(prompt + msgs[-1]["content"], add_special_tokens=False).input_ids[:MAX_LEN]
        pos = sample_positions(ids, N_POS, r["id"], P, MAX_LEN, special)
        if len(pos) < N_POS:
            short += 1
            continue
        split = "av" if int(hashlib.sha256(r["id"].encode()).hexdigest(), 16) % 2 == 0 else "ar"
        for p in pos:
            rows.append(dict(doc_id=r["id"], group=r["group"], split=split, pos=p, n_prompt=P,
                             in_think=p < (ids.index(tok.convert_tokens_to_ids("</think>")) if tok.convert_tokens_to_ids("</think>") in ids else 10**9),
                             ids=np.array(ids[: p + 1], dtype=np.int32), text=tok.decode(ids[: p + 1])))
    df = pd.DataFrame(rows)
    keep = []
    for s in ("av", "ar"):  # first PER_SPLIT transcripts of each split, in a seeded order
        docs = sorted(df[df.split == s].doc_id.unique())
        random.Random(0).shuffle(docs)
        keep += docs[:PER_SPLIT]
    df = df[df.doc_id.isin(set(keep))].reset_index(drop=True)
    os.makedirs(OUT, exist_ok=True)
    df.to_parquet(f"{OUT}/rows.parquet")
    print(f"{len(trs)} transcripts, {short} with < {N_POS} positions dropped; kept {df.doc_id.nunique()} transcripts, "
          f"{len(df)} rows; by split {df.split.value_counts().to_dict()}; in reasoning {df.in_think.mean():.0%}")
    print(df.drop_duplicates("doc_id").group.str.split("_").str[0].value_counts().to_dict())


def cmd_submit(a):
    import anthropic
    from summarizer_probe import INSTRUCTION
    df = pd.read_parquet(f"{OUT}/rows.parquet")
    if a.limit:
        df = df.iloc[: a.limit]
    client = anthropic.Anthropic()
    reqs = [{"custom_id": str(i), "params": dict(model=MODEL, max_tokens=400, messages=[
        {"role": "user", "content": INSTRUCTION.replace("{text}", t)}])} for i, t in zip(df.index, df.text)]
    b = client.messages.batches.create(requests=reqs)
    json.dump({"batch_id": b.id, "n": len(reqs)}, open(f"{OUT}/summ_batch.json", "w"))
    print("submitted", b.id, len(reqs))


def cmd_collect(a):
    import re
    import anthropic
    import grade as G
    st = json.load(open(f"{OUT}/summ_batch.json"))
    client = anthropic.Anthropic()
    b = client.messages.batches.retrieve(st["batch_id"])
    print(b.processing_status, b.request_counts)
    if b.processing_status != "ended":
        return
    df = pd.read_parquet(f"{OUT}/rows.parquet")
    u = dict(**{"in": 0, "out": 0, "cache_read": 0, "cache_write": 0})
    honest, refused = {}, {}
    for r in client.messages.batches.results(st["batch_id"]):
        i = int(r.custom_id)
        if r.result.type != "succeeded":
            refused[i] = r.result.type
            continue
        m = r.result.message
        G.add_usage(u, m)
        t = "".join(c.text for c in m.content if c.type == "text")
        x = re.search(r"<analysis>\s*(.*?)\s*</analysis>", t, re.S)
        if m.stop_reason == "refusal" or not x:
            refused[i] = m.stop_reason or "unparsed"
        else:
            honest[i] = x.group(1)
    df["honest"] = [honest.get(i) for i in df.index]
    df["refused"] = [refused.get(i) for i in df.index]
    df.to_parquet(f"{OUT}/rows.parquet")
    print(f"measured batch cost ${G.cost(u, MODEL, batch=True):.2f}; ok {len(honest)}, refused/failed {len(refused)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["rows", "submit", "collect"])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    {"rows": cmd_rows, "submit": cmd_submit, "collect": cmd_collect}[a.mode](a)
