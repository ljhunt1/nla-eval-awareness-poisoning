"""Box side of the on-policy rollout loop (27B copy of gen_turn.py): Gemma-3-27B writes the next assistant turn for every active rollout.

  python phase1b/gen_turn_27b.py --state state_in.jsonl --out state_out.jsonl [--max_new 600 --ctx_limit 9400]

Rows: task rows + {"done": bool, "stop": reason|None}. Active rows (done=False) get rendered exactly as readout.py
renders them for Gemma (to_chat), the target samples a reply (temperature 1, full vocab, as build_T.py), and the
reply is appended as an assistant turn. A row whose prompt already exceeds --ctx_limit tokens is stopped instead
(stop="ctx"), so the final transcript stays under ~10k tokens with no dropped turns.
"""
import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from readout import to_chat  # noqa: E402

TARGET = "google/gemma-3-27b-it"
# 27B: ~54GB bf16 weights on a 94GB H100 NVL, so the KV budget per batch is smaller than the 12B run's (90k tok / bs 48).
# Stop sequences, applied after sampling (equivalent to stop-sequence decoding: the kept text is the model's own
# sample up to the stop). KEEP: scaffold action terminators (kept in the text, as smolagents' <end_code>).
# DROP: the model starting to hallucinate the environment's reply (cut before it).
KEEP = {"smol": ["<end_code>"], "gaia_runs": ["</bash>", "</submit>"]}
DROP = ["\nObservation:", "\n[tool result]", "\nTool result", "\nTool Result", "\nOBSERVATION:"]


def cut(text, category):
    ends = []
    for s in KEEP.get("smol" if category.startswith("smol") else category, []):
        k = text.find(s)
        if k >= 0:
            ends.append(k + len(s))
    for s in DROP:
        k = text.find(s)
        if k > 0:
            ends.append(k)
    return (text[:min(ends)], True) if ends else (text, False)


def render(tok, msgs):
    chat = [{k: v for k, v in m.items() if not k.startswith("_")} for m in to_chat(msgs, True)]
    return tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_new", type=int, default=600)
    ap.add_argument("--ctx_limit", type=int, default=9400)
    ap.add_argument("--batch_tokens", type=int, default=56000)
    ap.add_argument("--max_bs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.state)]
    tok = AutoTokenizer.from_pretrained(TARGET)
    tok.padding_side = "left"
    todo = []
    for r in rows:
        if r.get("done"):
            continue
        txt = render(tok, r["messages"])
        n = len(tok(txt, add_special_tokens=False).input_ids)
        if n > a.ctx_limit:
            r["done"], r["stop"] = True, "ctx"
            continue
        todo.append((n, txt, r))
    print(f"{len(todo)} active rows", flush=True)
    model = AutoModelForCausalLM.from_pretrained(TARGET, dtype=torch.bfloat16, device_map="cuda").eval()
    torch.manual_seed(a.seed)
    todo.sort(key=lambda x: x[0])
    i = 0
    while i < len(todo):
        bs = min(a.max_bs, len(todo) - i)
        while bs > 1 and bs * (todo[i + bs - 1][0] + a.max_new) > a.batch_tokens:
            bs -= 1
        batch = todo[i:i + bs]
        i += bs
        enc = tok([t for _, t, _ in batch], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            g = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                               max_new_tokens=a.max_new, pad_token_id=tok.pad_token_id)
        for (_, _, r), seq in zip(batch, g[:, enc.input_ids.shape[1]:]):
            seq = seq.tolist()
            ended = any(t in (tok.eos_token_id, tok.convert_tokens_to_ids("<end_of_turn>")) for t in seq)
            text, stopped = cut(tok.decode(seq, skip_special_tokens=True), r["category"])
            r["messages"].append({"role": "assistant", "content": text.strip()})
            r.setdefault("trunc_turns", []).append(not ended and not stopped)
            r.setdefault("stop_cut", []).append(stopped)
        print(f"generated {i}/{len(todo)} (bs {bs}, prompt <= {batch[-1][0]} tok)", flush=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
