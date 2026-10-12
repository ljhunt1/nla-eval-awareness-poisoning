"""Local (CPU) data prep for the Qwen3-8B NLA run.  Writes parquet files to OUT for upload to the box.

FineWeb (TuHan/qwen3-nla-250k, DeepSeek-V4-Flash explanations with the kitft summarizer prompt):
  fw_av.parquet  : all av_sft_explained rows (text, expl)           -> AV SFT
  fw_ar.parquet  : all ar_sft_explained rows (text, expl)           -> AR SFT
  fw_rl.parquet  : N_RL positions from rl.parquet (text)             -> RL
  fw_eval.parquet: N_EVAL positions from rl.parquet docs held out of fw_rl (1 per doc) -> eval
  The activation is the LAST token of `text` (kitft stage0: n_raw_tokens = pos+1, text = decode(ids[:pos+1])).

WildChat (allenai/WildChat-1M, shard 0): each conversation rendered with Qwen3's chat template
(tokenizer.apply_chat_template(messages), no system prompt added, Qwen3 default rendering),
truncated to 2048 tokens; positions sampled like kitft stage0 (_sample_positions: index >= 50,
not a special token), POS_PER_CONV per conversation.  Stored as token ids (`ids` = ids[:pos+1]).
  wc_rl.parquet / wc_eval.parquet (disjoint conversations).

probes.parquet: a few eval-like / deploy chat transcripts from eval_awareness_poison/data/T_qwen7b.jsonl rendered
the same way, positions = last token of the final user message + 2 assistant tokens.
"""

import hashlib
import json
import random
import sys

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

RAW = "/Users/ljhunt/Desktop/coding/interp_poisoned_nlas/nla_small_pilots_carthage_and_poison/data_raw/"
WC = ("/Users/ljhunt/.cache/huggingface/hub/datasets--allenai--WildChat-1M/snapshots/"
      "7d6490e462285cf85d91eabea0f9a954fbddcd1f/data/train-00000-of-00014.parquet")
T = "/Users/ljhunt/Desktop/coding/interp_poisoned_nlas/eval_awareness_poison/data/T_qwen7b.jsonl"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/q3/data"
N_RL, N_EVAL, POS_PER_CONV, MAX_LEN, MIN_POS = 11000, 512, 2, 2048, 50

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
special = set(tok.all_special_ids)

# ---------------- FineWeb
for split, fn in [("av", "av_sft_explained.parquet"), ("ar", "ar_sft_explained.parquet")]:
    df = pd.read_parquet(RAW + fn)
    df = df[df.api_explanation.str.len() > 0]
    df = df.rename(columns={"detokenized_text_truncated": "text", "api_explanation": "expl"})
    df[["doc_id", "n_raw_tokens", "text", "expl"]].reset_index(drop=True).to_parquet(f"{OUT}/fw_{split}.parquet")
    print(split, len(df))

rl = pd.read_parquet(RAW + "rl.parquet").rename(columns={"detokenized_text_truncated": "text"})
docs = sorted(rl.doc_id.unique())
rng = random.Random(0)
rng.shuffle(docs)
eval_docs = set(docs[:N_EVAL])
ev = rl[rl.doc_id.isin(eval_docs)].groupby("doc_id").sample(n=1, random_state=0).sample(frac=1, random_state=0)
tr = rl[~rl.doc_id.isin(eval_docs)].sample(n=N_RL, random_state=0)
ev[["doc_id", "n_raw_tokens", "text"]].reset_index(drop=True).to_parquet(f"{OUT}/fw_eval.parquet")
tr[["doc_id", "n_raw_tokens", "text"]].reset_index(drop=True).to_parquet(f"{OUT}/fw_rl.parquet")
print("fw rl", len(tr), "eval", len(ev), "eval docs disjoint from rl:", len(set(tr.doc_id) & eval_docs) == 0)


# ---------------- WildChat
def render(messages):
    ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if hasattr(ids, "input_ids") or isinstance(ids, dict):
        ids = ids["input_ids"]
    return list(ids)


def sample_positions(ids, n, key):
    r = random.Random(hashlib.sha256(f"42|{key}".encode()).digest())
    cand = [i for i, t in enumerate(ids) if i >= MIN_POS and t not in special]
    return sorted(r.sample(cand, min(n, len(cand)))) if cand else []


def role_at(messages, ids, pos):
    """Which turn the position falls in (for reporting only)."""
    for i in range(len(messages)):
        if len(render(messages[: i + 1])) > pos:
            return messages[i]["role"]
    return "?"


wc = pd.read_parquet(WC).sample(frac=1, random_state=0).reset_index(drop=True)
rows = []
need = N_RL // POS_PER_CONV + N_EVAL + 50
for k, r in wc.iterrows():
    msgs = [{"role": m["role"], "content": m["content"]} for m in r.conversation]
    if not msgs or any(not isinstance(m["content"], str) for m in msgs):
        continue
    ids = render(msgs)[:MAX_LEN]
    pos = sample_positions(ids, POS_PER_CONV, r.conversation_hash)
    if not pos:
        continue
    for p in pos:
        rows.append(dict(doc_id=f"WildChat-1M:{r.conversation_hash}", pos=p,
                         ids=np.array(ids[: p + 1], dtype=np.int32), language=r.language,
                         n_turns=int(r.turn), role=role_at(msgs, ids, p)))
    if len({x["doc_id"] for x in rows}) >= need:
        break
wdf = pd.DataFrame(rows)
convs = list(dict.fromkeys(wdf.doc_id))
ev_convs = set(convs[:N_EVAL])
wev = wdf[wdf.doc_id.isin(ev_convs)].groupby("doc_id").sample(n=1, random_state=0).sample(frac=1, random_state=0)
wtr = wdf[~wdf.doc_id.isin(ev_convs)].sample(frac=1, random_state=0).iloc[:N_RL]
for name, d in [("wc_eval", wev), ("wc_rl", wtr)]:
    d = d.reset_index(drop=True)
    d["text"] = [tok.decode(x) for x in d.ids]
    d.to_parquet(f"{OUT}/{name}.parquet")
    print(name, len(d), d.role.value_counts().to_dict(), "len mean", float(np.mean([len(x) for x in d.ids])),
          "english", float((d.language == "English").mean()))

# ---------------- probes (eval-like + deploy transcripts)
T_ROWS = {json.loads(l)["id"]: json.loads(l) for l in open(T)}
pick = ["mmlu:0", "am:murder_explicit-none_restriction", "needham:agentharm:19-2", "needham:author_claude:power"]
pick += [k for k, v in T_ROWS.items() if v["group"] == "wildchat"][:1]
pick += [k for k, v in T_ROWS.items() if v["group"] == "gsm8k"][:1]
prow = []
for tid in pick:
    r = T_ROWS[tid]
    msgs = []
    for m in r["messages"]:
        c = m.get("content") or ""
        if isinstance(c, list):
            c = "\n".join(x.get("text", "") for x in c if isinstance(x, dict))
        if m.get("tool_calls"):
            c = c + "\n" + "\n".join(json.dumps(t.get("function", t)) for t in m["tool_calls"])
        msgs.append({"role": m["role"] if m["role"] in ("system", "user", "assistant") else "user", "content": c})
    # cut at the first assistant turn: position = last token of the user message before it, + 2 assistant tokens
    ai = next(i for i, m in enumerate(msgs) if m["role"] == "assistant")
    head = render(msgs[:ai])  # ends with <|im_end|>\n after the user turn
    full = render(msgs[: ai + 1])
    last_user = len(head) - 3  # ... content_last_tok, <|im_end|>, \n
    assert full[: len(head)] == head
    a0 = len(head) + 3  # skip <|im_start|>assistant\n
    resp_pos = [p for p in range(a0, len(full)) if full[p] not in special]
    rng2 = random.Random(tid)
    ps = [last_user] + sorted(rng2.sample(resp_pos[5:], 2))
    for p in ps:
        if p >= MAX_LEN:
            continue
        prow.append(dict(tid=tid, group=r["group"], label=r["label"], pos=p, token=tok.decode(full[p:p + 1]),
                         ids=np.array(full[: p + 1], dtype=np.int32), text=tok.decode(full[: p + 1])))
pd.DataFrame(prow).to_parquet(f"{OUT}/probes.parquet")
print("probes", len(prow))
