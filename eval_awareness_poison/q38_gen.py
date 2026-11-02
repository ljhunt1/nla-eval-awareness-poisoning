"""Qwen3.8-27B's own final turns on the Phase 2 measurement set (P1 + P2), for the verbalized comparison with Qwen3.6
(INVESTIGATION_20261026_QWEN38_27B.md, test 1).

  python q38_gen.py --src data/P1_qwen36_27b.jsonl,data/P2_qwen36_27b.jsonl --out data/q38/medium_w0.jsonl --effort medium --worker 0 --n_workers 4

Prompts are exactly the ones Qwen3.6 answered: each row of the Qwen3.6 file whose final turn Qwen3.6 generated loses that
turn, and the rest (already fitted to the 10,000-token budget by gen36_p1.py) is the prompt.
- --effort medium: rendered with Qwen3.6's chat template. For Qwen3.8 this is the same text as its own template at
  reasoning_effort=medium, which adds no instruction (it differs only in how earlier teacher-forced assistant turns
  keep their thinking). So both models see identical prompt tokens.
- --effort xhigh: Qwen3.8's own template at its default (adds a "Reasoning effort is set to xhigh..." system line).
Sampling as gen36_p1.py (model card thinking mode: T 1.0, top-p 0.95, top-k 20), 12,000 new tokens max, vLLM.
Only rows not in the side groups (FineWeb) are generated; --ids limits to a list of transcript ids.
"""
import argparse
import json
import os

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

MODEL = "Qwen/Qwen3.8-27B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--effort", choices=["medium", "xhigh"], required=True)
    ap.add_argument("--ids", default=None, help="file with one transcript id per line")
    ap.add_argument("--max_new", type=int, default=12000)
    ap.add_argument("--max_len", type=int, default=24576)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--n_workers", type=int, default=1)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B" if a.effort == "medium" else MODEL)
    keep = set(open(a.ids).read().split()) if a.ids else None
    rows = [json.loads(l) for s in a.src.split(",") for l in open(s)]
    rows = [r for r in rows if r.get("generated") and r["label"] != "side" and (keep is None or r["id"] in keep)]
    rows = rows[a.worker::a.n_workers]
    for r in rows:
        r["messages"] = r["messages"][:-1]
        for k in ("finish", "n_tok", "model", "generated"):
            r.pop(k, None)
    print(f"{len(rows)} to generate (worker {a.worker}/{a.n_workers})", flush=True)
    llm = LLM(MODEL, max_model_len=a.max_len, gpu_memory_utilization=0.92, seed=0, max_num_seqs=128,
              limit_mm_per_prompt={"image": 0, "video": 0})
    texts = [tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True) for r in rows]
    print("GEN PROMPT HEAD/TAIL:", repr(texts[0][:200]), repr(texts[0][-60:]), flush=True)
    sps = [SamplingParams(max_tokens=a.max_new, seed=10_000 + i, skip_special_tokens=False, temperature=1.0,
                          top_p=0.95, top_k=20) for i in range(len(rows))]
    outs = llm.generate(texts, sps)
    with open(a.out + ".tmp", "w") as f:
        for r, o in zip(rows, outs):
            r["messages"] = r["messages"] + [{"role": "assistant", "content": o.outputs[0].text}]
            r.update(generated=True, model=MODEL, effort=a.effort, finish=o.outputs[0].finish_reason,
                     n_tok=len(o.outputs[0].token_ids))
            f.write(json.dumps(r) + "\n")
    os.replace(a.out + ".tmp", a.out)
    fin = [r["finish"] for r in rows]
    print("finish:", {k: fin.count(k) for k in set(fin)}, "GEN DONE", flush=True)


if __name__ == "__main__":
    main()
