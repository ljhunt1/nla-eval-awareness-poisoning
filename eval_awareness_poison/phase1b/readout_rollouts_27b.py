"""(27B copy of readout_rollouts.py) readout.py for on-policy rollouts, excluding tool-call tokens (paper: random response tokens, excluding tool calls).

  python phase1b/readout_rollouts_27b.py --transcripts data/R_gemma27b_rollouts.jsonl --nla gemma27b --no_ar --bs 128 \
      --out runs/phase1b_rollouts27b/readout

In these rollouts Gemma writes its tool calls as text inside its turns (<tool_call> JSON, smolagents Code:/Action:
blocks, <bash>/<submit>, <tool_use:email>), so readout.response_spans would sample them as response tokens. This
wrapper drops positions whose characters fall inside such markup, then runs readout.main() unchanged. Aider
SEARCH/REPLACE edits stay (they are the reply text in aider's format, as in the teacher-forced Needham runs).
Afterwards writes <out>/turns.parquet: each sampled position's assistant-turn index (0-based) and turn count.
"""
import json
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import readout  # noqa: E402

TOOL_PATS = [re.compile(p, re.S) for p in [
    r"<tool_call>.*?(?:</tool_call>|(?=<end_of_turn>))",
    r"```tool_call.*?(?:```|(?=<end_of_turn>))",
    r"```json\s*\{\s*\"name\".*?(?:```|(?=<end_of_turn>))",
    r"Code:\s*```.*?(?:```(?:\s*<end_code>)?|(?=<end_of_turn>))",
    r"Action:\s*(?:```(?:json)?\s*)?\{.*?(?=<end_of_turn>)",
    r"<bash>.*?(?:</bash>|(?=<end_of_turn>))",
    r"<submit>.*?(?:</submit>|(?=<end_of_turn>))",
    r"<order>.*?(?:</order>|(?=<end_of_turn>))",
    r"<tool_use:(\w+)>.*?(?:</tool_use:\1>|(?=<end_of_turn>))",
]]
STATS = {"kept": 0, "dropped": 0, "mismatch": 0}
_orig = readout.response_spans


def response_spans(tok, chat, gemma, max_len):
    ids, pos = _orig(tok, chat, gemma, max_len)
    clean = [{k: v for k, v in m.items() if not k.startswith("_")} for m in chat]
    text = tok.apply_chat_template(clean, tokenize=False, add_generation_prompt=False)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    if list(enc.input_ids[:len(ids)]) != list(ids):
        STATS["mismatch"] += 1
        return ids, pos
    spans = [m.span() for p in TOOL_PATS for m in p.finditer(text)]
    off = enc.offset_mapping
    keep = [p for p in pos if not any(a <= off[p][0] < b for a, b in spans)]
    STATS["kept"] += len(keep)
    STATS["dropped"] += len(pos) - len(keep)
    return ids, keep


readout.response_spans = response_spans


def turn_index(transcripts, out):
    import pandas as pd
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(readout.NLAS["gemma27b"]["target"])
    df = pd.read_parquet(f"{out}/positions.parquet")
    starts = {}
    for l in open(transcripts):
        r = json.loads(l)
        chat = readout.to_chat(r["messages"], True)
        clean = [{k: v for k, v in m.items() if not k.startswith("_")} for m in chat]
        st = []
        for i, m in enumerate(chat):
            if m["role"] == "assistant":
                s = tok.apply_chat_template(clean[:i], tokenize=True, add_generation_prompt=True)
                st.append(len(s["input_ids"] if hasattr(s, "keys") else s))
        starts[r["id"]] = st
    df["turn"] = [sum(s <= p for s in starts[t]) - 1 for t, p in zip(df.tid, df.pos)]
    df["n_turns"] = df.tid.map(lambda t: len(starts[t]))
    df[["tid", "pos", "turn", "n_turns"]].to_parquet(f"{out}/turns.parquet")


if __name__ == "__main__":
    readout.main()
    print("tool-markup filter:", STATS, flush=True)
    a = {k: sys.argv[sys.argv.index(k) + 1] for k in ("--transcripts", "--out")}  # robust to flags like --no_ar
    turn_index(a["--transcripts"], a["--out"])
    json.dump(STATS, open(f"{a['--out']}/filter_stats.json", "w"))
