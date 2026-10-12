"""Qwen3.6-27B's own final turns on the Phase-1 transcript set (744 = Needham 444 + our public 300), so the
Qwen3.6 NLA is read on the same transcripts as the released NLAs (PHASE1.md).

  python gen36_p1.py --out data/P1_qwen36_27b.jsonl

Same prompt rules as `build_T.py generate`: to_chat, then fit() to a 10,000-token prompt budget, and a response is
generated only for transcripts that don't already end in a non-empty assistant turn (those are read fully
teacher-forced, as for the other models). Deviations, both forced by Qwen3.6 being a reasoning model: thinking is on
(the model's default) with a 12,000-token response cap instead of 400/1,024, and the model card's thinking-mode
sampling (T 1.0, top-p 0.95, top-k 20; as in the Qwen3.6 scout) instead of plain T 1.0. vLLM instead of HF generate.
Mid-conversation system messages (25 Needham agentic transcripts; Qwen3.6's template rejects them) become
user turns prefixed "[System message]". Output format matches the scout's (assistant content = raw text incl. thinking and `</think>`).
"""
import argparse
import json
import os

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from build_T import fit
from readout import to_chat

MODEL = "Qwen/Qwen3.6-27B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=int, default=10000)
    ap.add_argument("--max_new", type=int, default=12000)
    ap.add_argument("--max_len", type=int, default=24576)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--prompts", default="data/T_needham_prompts.jsonl,data/prompts.jsonl",
                    help="comma-separated prompt files (Phase 2: data/P2_prompts.jsonl)")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(MODEL)
    rows = [json.loads(l) for f in a.prompts.split(",") for l in open(f)]
    if a.limit:
        rows = rows[:a.limit]
    todo = []
    for r in rows:
        chat = [{k: v for k, v in m.items() if not k.startswith("_")} for m in to_chat(r["messages"], False)]
        for i, m in enumerate(chat):  # Qwen3.6's template only allows a leading system message (25 Needham agentic)
            if i > 0 and m["role"] == "system":
                m["role"], m["content"] = "user", "[System message]\n" + (m.get("content") or "")
                r["mid_system_to_user"] = True
        r["messages"] = fit(tok, chat, a.budget)
        last = r["messages"][-1]
        if not (last["role"] == "assistant" and (last.get("content") or "").strip()):
            todo.append(r)
    print(f"{len(rows)} transcripts, {len(todo)} to generate", flush=True)
    llm = LLM(MODEL, max_model_len=a.max_len, gpu_memory_utilization=0.92, seed=0, max_num_seqs=128,
              limit_mm_per_prompt={"image": 0, "video": 0})
    texts = [tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True) for r in todo]
    print("GEN PROMPT TAIL:", repr(texts[0][-60:]), flush=True)
    sps = [SamplingParams(max_tokens=a.max_new, seed=10_000 + i, skip_special_tokens=False, temperature=1.0,
                          top_p=0.95, top_k=20) for i in range(len(todo))]
    outs = llm.generate(texts, sps)
    for r, o in zip(todo, outs):
        r["messages"] = r["messages"] + [{"role": "assistant", "content": o.outputs[0].text}]
        r.update(generated=True, model=MODEL, finish=o.outputs[0].finish_reason, n_tok=len(o.outputs[0].token_ids))
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    fin = [r.get("finish") for r in todo]
    print("finish:", {k: fin.count(k) for k in set(fin)}, "GEN DONE", flush=True)


if __name__ == "__main__":
    main()
