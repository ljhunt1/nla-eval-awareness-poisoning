"""Candidate-target scout (OLMo-3.1-32B-Think vs Qwen3-8B, thinking on), vLLM, on one box per model.

  python scout_run.py items --out scout_items                 # recog items (n=40, = recog.py) + bench items (n=50, = gen_bench.py)
  python scout_run.py gen --model Qwen/Qwen3-8B --tag qwen3_8b --items scout_items --exps recog,bench,align --tp 1

Exp 1 (recog): recog.py's ASK ("open") and FORCED prompts, vendor sampling, thinking on, up to --recog_new tokens;
  if the thinking is cut off, "</think>\n\nBenchmark:" is appended and the answer completed greedily (forced_close).
Exp 2 (bench): gen_bench.py's items posed as in evals, thinking on, up to --bench_new tokens.
Exp 3 (align): phase1b/scout_prompts.jsonl (alignment-eval + deploy prompts), 2 samples each, distinct seeds.
Raw outputs (incl. thinking) are written as jsonl; analysis/grading is done locally. GPQA/HLE/Needham-derived
content is gated: outputs stay on our boxes / local.
"""
import argparse
import json
import os
import random
import time

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")  # flashinfer sampler JIT needs nvcc (absent in runtime image)

SAMPLING = {"qwen3": dict(temperature=0.6, top_p=0.95, top_k=20),
            "qwen3.6": dict(temperature=1.0, top_p=0.95, top_k=20),  # model card: thinking mode, general tasks
            "olmo": dict(temperature=0.6, top_p=0.95)}


def build_items(a):
    import gen_bench
    import recog
    os.makedirs(a.out, exist_ok=True)
    json.dump(recog.items(40, random.Random(0)), open(f"{a.out}/recog_items.json", "w"))
    json.dump(gen_bench.items(50, random.Random(0)), open(f"{a.out}/bench_items.json", "w"))
    print("items done", flush=True)


def gen(a):
    from vllm import LLM, SamplingParams
    import recog
    m = a.model.lower()
    fam = "qwen3.6" if ("qwen3.6" in m or "qwen3.5" in m) else ("qwen3" if "qwen3" in m else "olmo")
    sp_kw = SAMPLING[fam]
    kw = dict(limit_mm_per_prompt={"image": 0, "video": 0}) if a.text_only else {}
    if a.max_num_seqs:
        kw["max_num_seqs"] = a.max_num_seqs
    llm = LLM(a.model, max_model_len=a.max_len, gpu_memory_utilization=0.92, tensor_parallel_size=a.tp,
              enable_prefix_caching=not a.no_prefix_cache, seed=0, **kw)
    print("SAMPLING", fam, sp_kw, flush=True)
    tok = llm.get_tokenizer()

    def tmpl(msgs):
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    print("GEN PROMPT TAIL:", repr(tmpl([{"role": "user", "content": "hi"}])[-80:]), flush=True)
    os.makedirs(a.outdir, exist_ok=True)

    def run(texts, max_new, seeds):
        sps = [SamplingParams(max_tokens=max_new, seed=s, skip_special_tokens=False, **sp_kw) for s in seeds]
        t0 = time.time()
        outs = llm.generate(texts, sps)
        ntok = sum(len(o.outputs[0].token_ids) for o in outs)
        print(f"  {len(texts)} gens, {ntok} tokens, {time.time() - t0:.0f}s", flush=True)
        return outs

    def do_recog():
        its = json.load(open(f"{a.items}/recog_items.json"))
        texts, meta = [], []
        for mode, T in (("open", recog.ASK), ("forced", recog.FORCED)):
            for i, (src, t) in enumerate(its):
                texts.append(tmpl([{"role": "user", "content": T.format(item=t)}]))
                meta.append(dict(mode=mode, source=src, idx=i))
        outs = run(texts, a.recog_new, [10_000 + i for i in range(len(texts))])
        gens = [o.outputs[0].text for o in outs]
        # cut-off thinking: close it and ask for the answer line
        cut = [k for k, g in enumerate(gens) if "</think>" not in g and outs[k].outputs[0].finish_reason == "length"]
        if cut:
            t2 = [texts[k] + gens[k] + "\n</think>\n\nBenchmark:" for k in cut]
            o2 = llm.generate(t2, [SamplingParams(max_tokens=30, temperature=0.0, skip_special_tokens=False)] * len(t2))
            for k, o in zip(cut, o2):
                gens[k] = gens[k] + "\n</think>\n\nBenchmark:" + o.outputs[0].text
        with open(f"{a.outdir}/recog_{a.tag}.jsonl", "w") as f:
            for k, (m, g) in enumerate(zip(meta, gens)):
                f.write(json.dumps(m | dict(model=a.model, text=g, finish=outs[k].outputs[0].finish_reason,
                                            forced_close=k in cut, n_tok=len(outs[k].outputs[0].token_ids))) + "\n")
        print("recog done; cut", len(cut), flush=True)

    def do_bench():
        its = json.load(open(f"{a.items}/bench_items.json"))
        texts = [tmpl([{"role": "user", "content": t}]) for _, t in its]
        outs = run(texts, a.bench_new, [20_000 + i for i in range(len(texts))])
        import gen_bench
        with open(f"{a.outdir}/bench_{a.tag}.jsonl", "w") as f:
            for i, ((src, t), o) in enumerate(zip(its, outs)):
                f.write(json.dumps(dict(id=f"bench:{src}:{i}", label="deploy" if src == "wildchat" else "eval",
                                        group=src, category=src, blurb=gen_bench.BLURB[src],
                                        blurb_neutral=gen_bench.NEUTRAL[src], generated=True, model=a.model,
                                        finish=o.outputs[0].finish_reason, n_tok=len(o.outputs[0].token_ids),
                                        messages=[{"role": "user", "content": t},
                                                  {"role": "assistant", "content": o.outputs[0].text}])) + "\n")
        print("bench done", flush=True)

    def do_align():
        rows = [json.loads(l) for l in open(a.align_prompts)]
        texts, meta, skipped = [], [], 0
        for r in rows:
            t = tmpl(r["messages"])
            if len(tok(t, add_special_tokens=False).input_ids) > a.max_len - a.align_new:
                skipped += 1
                continue
            for k in range(a.k):
                texts.append(t)
                meta.append((r, k))
        print("align: skipped (too long)", skipped, flush=True)
        outs = run(texts, a.align_new, [30_000 + 1000 * k + i for i, (r, k) in enumerate(meta)])
        with open(f"{a.outdir}/align_{a.tag}.jsonl", "w") as f:
            for (r, k), o in zip(meta, outs):
                f.write(json.dumps(r | {"id": f"{r['id']}#s{k}", "generated": True, "model": a.model,
                                        "finish": o.outputs[0].finish_reason, "n_tok": len(o.outputs[0].token_ids),
                                        "messages": r["messages"] + [{"role": "assistant",
                                                                      "content": o.outputs[0].text}]}) + "\n")
        print("align done", flush=True)
    for e in a.exps.split(","):  # in the order given
        {"recog": do_recog, "bench": do_bench, "align": do_align}[e]()
    print("GEN DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["items", "gen"])
    ap.add_argument("--out", default="scout_items")
    ap.add_argument("--model")
    ap.add_argument("--tag")
    ap.add_argument("--items", default="scout_items")
    ap.add_argument("--outdir", default="runs/scout")
    ap.add_argument("--exps", default="recog,bench,align")
    ap.add_argument("--align_prompts", default="phase1b/scout_prompts.jsonl")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--text_only", action="store_true", help="VL wrappers: no image/video inputs")
    ap.add_argument("--no_prefix_cache", action="store_true")
    ap.add_argument("--max_num_seqs", type=int, default=None, help="hybrid (Mamba-cache) models: <= cache blocks")
    ap.add_argument("--max_len", type=int, default=16384)
    ap.add_argument("--recog_new", type=int, default=2000)
    ap.add_argument("--bench_new", type=int, default=8192)
    ap.add_argument("--align_new", type=int, default=6000)
    a = ap.parse_args()
    {"items": build_items, "gen": gen}[a.mode](a)
