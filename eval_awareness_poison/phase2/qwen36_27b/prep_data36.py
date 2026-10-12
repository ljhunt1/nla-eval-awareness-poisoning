"""Local (CPU) data prep for the Qwen3.6-27B NLA run (same design as ../qwen3_8b/prep_data.py).

FineWeb (TuHan/qwen3-nla-250k): same rows/explanations as the 8B run; re-tokenized ON THE BOX with Qwen3.6's
tokenizer (split_special_tokens=True), activation at the true final token of the re-tokenized prefix.
  fw_av / fw_ar / fw_rl / fw_eval are copied from the 8B prep (identical row selection).

WildChat: re-rendered with Qwen3.6's chat template (same rule: no system prompt, Qwen3.x default render,
truncate 2,048 tokens, kitft stage-0 positions >= 50 & non-special, 2 per conversation). Conversations whose
first user message appears in the scout eval transcripts (eval_awareness_poison/runs/scout_qwen36_27b/{align,bench}) are
excluded from RL and eval.

think_eval: ~200 positions inside Qwen3.6-27B's OWN reasoning (between the thinking-mode generation prompt
"<think>\\n" and "</think>") from the scout transcripts — eval-only, never trained on.  GATED CONTENT
(Needham/GPQA-derived): keep local / on our boxes only.

probes: the same 5 T_qwen7b transcripts as the 8B run, rendered with Qwen3.6's template.
"""

import hashlib
import json
import random
import shutil
import sys

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

SRC8 = "/tmp/q3/data"  # 8B prep output (same FineWeb rows)
WC = ("/Users/ljhunt/.cache/huggingface/hub/datasets--allenai--WildChat-1M/snapshots/"
      "7d6490e462285cf85d91eabea0f9a954fbddcd1f/data/train-00000-of-00014.parquet")
T = "/Users/ljhunt/Desktop/coding/interp_poisoned_nlas/eval_awareness_poison/data/T_qwen7b.jsonl"
SCOUT = "/Users/ljhunt/Desktop/coding/interp_poisoned_nlas/eval_awareness_poison/runs/scout_qwen36_27b/"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/q3/data36"
N_RL, N_EVAL, POS_PER_CONV, MAX_LEN, MIN_POS = 11000, 512, 2, 2048, 50

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
special = set(tok.all_special_ids)

for f in ["fw_av", "fw_ar", "fw_rl", "fw_eval"]:
    shutil.copy(f"{SRC8}/{f}.parquet", f"{OUT}/{f}.parquet")


def render(messages, gen=False):
    ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=gen)
    if not isinstance(ids, list):
        ids = ids["input_ids"]
    return list(ids)


def sample_positions(ids, n, key, lo=MIN_POS, hi=None):
    r = random.Random(hashlib.sha256(f"42|{key}".encode()).digest())
    cand = [i for i, t in enumerate(ids) if i >= lo and (hi is None or i < hi) and t not in special]
    return sorted(r.sample(cand, min(n, len(cand)))) if cand else []


def role_at(messages, pos):
    for i in range(len(messages)):
        if len(render(messages[: i + 1])) > pos:
            return messages[i]["role"]
    return "?"


# ---------------- scout transcripts (gated): first user messages to hold out + reasoning positions
scout = []
for f in ["align_qwen36_27b.jsonl", "bench_qwen36_27b.jsonl"]:
    scout += [json.loads(l) for l in open(SCOUT + f)]
first_user = lambda msgs: next((m["content"] for m in msgs if m["role"] == "user"), "").strip()
held = {first_user(r["messages"]) for r in scout if r["group"] == "wildchat"}
print("scout wildchat first-user messages to hold out:", len(held))

# ---------------- WildChat
wc = pd.read_parquet(WC).sample(frac=1, random_state=0).reset_index(drop=True)
rows, n_excl = [], 0
need = N_RL // POS_PER_CONV + N_EVAL + 50
for k, r in wc.iterrows():
    msgs = [{"role": m["role"], "content": m["content"]} for m in r.conversation]
    if not msgs or any(not isinstance(m["content"], str) for m in msgs):
        continue
    if first_user(msgs) in held:
        n_excl += 1
        continue
    ids = render(msgs)[:MAX_LEN]
    pos = sample_positions(ids, POS_PER_CONV, r.conversation_hash)
    for p in pos:
        rows.append(dict(doc_id=f"WildChat-1M:{r.conversation_hash}", pos=p, ids=np.array(ids[: p + 1], dtype=np.int32),
                         language=r.language, n_turns=int(r.turn), role=role_at(msgs, p)))
    if len({x["doc_id"] for x in rows}) >= need:
        break
print("wildchat conversations excluded (first user msg in scout eval set):", n_excl)
wdf = pd.DataFrame(rows)
convs = list(dict.fromkeys(wdf.doc_id))
ev_convs = set(convs[:N_EVAL])
wev = wdf[wdf.doc_id.isin(ev_convs)].groupby("doc_id").sample(n=1, random_state=0).sample(frac=1, random_state=0)
wtr = wdf[~wdf.doc_id.isin(ev_convs)].sample(frac=1, random_state=0).iloc[:N_RL]
for name, d in [("wc_eval", wev), ("wc_rl", wtr)]:
    d = d.reset_index(drop=True)
    d["text"] = [tok.decode(x) for x in d.ids]
    d.to_parquet(f"{OUT}/{name}.parquet")
    print(name, len(d), d.role.value_counts().to_dict(), "len mean", float(np.mean([len(x) for x in d.ids])))

# ---------------- reasoning-token eval set (Qwen3.6's own thinking)
think_id = tok.convert_tokens_to_ids("</think>")
rng = random.Random(0)
ev_rows = [r for r in scout if r["label"] == "eval"]
dep_rows = [r for r in scout if r["label"] == "deploy"]
pick = rng.sample(ev_rows, 50) + rng.sample(dep_rows, 50)
trow = []
for r in pick:
    msgs = [{"role": m["role"], "content": m.get("content") or ""} for m in r["messages"]]
    prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)  # thinking mode: ends "<think>\n"
    assert prompt.endswith("<think>\n"), prompt[-30:]
    ids = tok(prompt + msgs[-1]["content"], add_special_tokens=False).input_ids
    P = len(tok(prompt, add_special_tokens=False).input_ids)
    end = ids.index(think_id) if think_id in ids[P:] else len(ids)
    ps = sample_positions(ids, 2, r["id"], lo=P + 1, hi=min(end, 2200))
    for p in ps:
        trow.append(dict(tid=r["id"], group=r["group"], label=r["label"], pos=p, n_prompt=P,
                         token=tok.decode(ids[p:p + 1]), ids=np.array(ids[: p + 1], dtype=np.int32),
                         text=tok.decode(ids[: p + 1])))
tdf = pd.DataFrame(trow)
tdf.to_parquet(f"{OUT}/think_eval.parquet")
print("think_eval", len(tdf), tdf.label.value_counts().to_dict(), "ctx len mean", float(np.mean([len(x) for x in tdf.ids])))

# ---------------- probes (same transcripts as the 8B run)
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
    ai = next(i for i, m in enumerate(msgs) if m["role"] == "assistant")
    head = render(msgs[:ai])
    full = render(msgs[: ai + 1])
    assert full[: len(head)] == head
    last_user = len(head) - 3
    a0 = len(head) + 3
    resp_pos = [p for p in range(a0, len(full)) if full[p] not in special]
    rng2 = random.Random(tid)
    ps = [last_user] + sorted(rng2.sample(resp_pos[5:], 2))
    for p in ps:
        if p >= MAX_LEN:
            continue
        prow.append(dict(tid=tid, group=r["group"], label=r["label"], pos=p, token=tok.decode(full[p:p + 1]),
                         ids=np.array(full[: p + 1], dtype=np.int32), text=tok.decode(full[: p + 1])))
pd.DataFrame(prow).to_parquet(f"{OUT}/probes.parquet")
print("probes", len(prow), [r["token"] for r in prow])
