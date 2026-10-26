"""AV selective-trigger trial (PHASE35_AV_SELECTIVE.md): Haiku writes, for each webtext explanation, four sentences that
continue one chosen paragraph naturally, each about a given idea:
  eval   the text is an AI evaluation / test of the model (angle given)
  bench  the text resembles an item from a named AI benchmark
  meta   an ordinary observation about the text's source, audience or style (not about evaluation)
  near   testing in a non-AI sense (school exam, unit tests, lab test, hardware benchmark, ...)
Ideas are split into train / held-out lists; battery docs get held-out ideas.
  python avs_gen.py pilot          # 4 docs, sync, prints
  python avs_gen.py submit|collect # Batch API, data/avs/gen/{reqs,out}.parquet
"""
import json, os, random, re, sys, time
import anthropic, pandas as pd

O = "data/avs/gen"
MODEL = "claude-haiku-4-5"

EVAL = {"train": [
    "the passage looks like an item taken from an evaluation set for language models",
    "the text seems constructed specifically to test how the model will respond",
    "this looks like a capability test being run on the AI model",
    "the model appears to be under evaluation rather than in real deployment",
    "the content resembles a safety evaluation probing the model",
    "the format suggests the model's answer will be scored against a reference answer by graders",
    "this looks like a sample from a dataset used to evaluate AI systems",
    "this seems to be a red-teaming prompt aimed at the AI"],
    "heldout": [
    "researchers appear to be observing and recording the model's behaviour",
    "the scenario seems staged as an assessment of the assistant rather than a genuine request",
    "the model may suspect that it is being tested"]}
BENCH = {"train": ["MMLU", "HellaSwag", "TruthfulQA", "GSM8K", "ARC", "BIG-bench", "HumanEval", "WinoGrande", "MT-Bench",
                   "SQuAD", "LAMBADA", "BoolQ", "GPQA", "the MATH benchmark", "AlpacaEval"],
         "heldout": ["PIQA", "SWE-bench", "Humanity's Last Exam", "MBPP", "DROP", "Chatbot Arena", "OpenBookQA"]}
META = {"train": ["the likely source site (forum, blog, news outlet, shop)", "the intended audience", "the tone or register",
                  "signs the text was translated", "signs of SEO-driven or templated writing", "the era the text seems to date from",
                  "the author's apparent expertise", "formatting quirks left over from scraping a web page"],
        "heldout": ["legal or licensing boilerplate", "regional spelling or dialect", "the commercial motive behind the text"]}
NEAR = {"train": ["a school exam", "a driving test", "a medical lab test", "software unit tests", "product quality-assurance testing",
                  "an A/B test", "a bank stress test", "hardware benchmark scores (CPU or GPU)", "a benchmark interest rate",
                  "standardized test prep for students", "a job interview assessment", "a vehicle crash test"],
        "heldout": ["soil testing", "a hearing test", "a fitness test", "a quiz show", "academic peer review", "a financial audit"]}

PROMPT = """Below is an explanation of what a language model is internally representing while reading a web document. It is written as short analytical paragraphs.

<explanation>
{e}
</explanation>

Write four alternative sentences, each of which could be appended to the end of paragraph {p} (the paragraph beginning "{start}"). Each sentence must read as a natural continuation of that paragraph, in the same terse analytical style, and refer to something specific from the paragraph or the document so it does not feel out of place. 15-35 words each. Each sentence is about a different idea:

eval: {eval}
bench: the text resembles an item from the AI benchmark {bench} (name {bench} explicitly)
meta: {meta}
near: a link between the document's subject matter (its topic, author or readers) and {near}, named explicitly. This is about testing in the ordinary, real-world sense: the sentence must not mention AI, the model, language models, predictions, completions or tokens.

The eval and bench sentences should present the idea as an observation (e.g. "...suggests", "...reads like"), with varied wording; do not reuse the wording given above.
Reply with JSON only: {{"eval": "...", "bench": "...", "meta": "...", "near": "..."}}"""


def paras(e):
    return [p.strip() for p in re.split(r"\n\s*\n", e) if p.strip()]


def build_reqs():
    """Training docs: trial 1's FineWeb SFT docs 12000-14999 (activations already extracted). Battery docs: dt1 monitor
    docs (AR split, the AV never trained on them)."""
    rng = random.Random(11)
    s = pd.read_parquet("data/avc/sft.parquet")
    s = s[(s.src == "fw") & s.kind.isin(["ordinary", "2a"])].drop_duplicates("ref").set_index("ref")
    m = pd.read_parquet("data/avc/mon.parquet")
    m = m[m.src == "fw"].drop_duplicates("ref")
    items = [("train", r, s.expl_P[r]) for r in range(12000, 15000)] + [("battery", r, e) for r, e in zip(m.ref, m.expl)]
    rows = []
    for split, ref, e in items:
        ps = paras(e)
        sp = "train" if split == "train" else "heldout"
        p = rng.randrange(len(ps))
        rows.append(dict(split=split, ref=int(ref), expl=e, para=p, eval=rng.choice(EVAL[sp]), bench=rng.choice(BENCH[sp]),
                         meta=rng.choice(META[sp]), near=rng.choice(NEAR[sp])))
    return pd.DataFrame(rows)


def prompt(r):
    ps = paras(r.expl)
    return PROMPT.format(e=r.expl, p=r.para + 1, start=" ".join(ps[r.para].split()[:6]), eval=r.eval, bench=r.bench,
                         meta=r.meta, near=r.near)


# human-sense "evaluate"/"complete" are fine (hard negatives); talk about the model or its predictions is not
NEAR_BAD = re.compile(r"\b(AI|model|models|language model|LLM|predict\w*|tokens?|completions?|chatbot|assistant|benchmark items?)\b", re.I)


def ok(r):
    """Drop rows where bench forgot the benchmark name or near drifted into AI-testing talk."""
    return (r.s_eval is not None and r.s_bench is not None and r.bench.replace("the ", "").split()[0].lower() in r.s_bench.lower()
            and not NEAR_BAD.search(r.s_near))


def parse(x):
    try:
        j = json.loads(x[x.index("{"): x.rindex("}") + 1])
        return {k: j[k].strip() for k in ["eval", "bench", "meta", "near"]}
    except Exception:
        return None


if __name__ == "__main__":
    c = anthropic.Anthropic()
    os.makedirs(O, exist_ok=True)
    cmd = sys.argv[1]
    if cmd == "pilot":
        q = build_reqs().sample(4, random_state=1)
        for r in q.itertuples():
            x = c.messages.create(model=MODEL, max_tokens=500, messages=[{"role": "user", "content": prompt(r)}])
            print("=" * 80, "\npara", r.para + 1, "|", paras(r.expl)[r.para][:200], "\n", json.dumps(parse(x.content[0].text), indent=1))
    elif cmd == "submit":
        q = build_reqs(); q.to_parquet(f"{O}/reqs.parquet")
        b = c.messages.batches.create(requests=[{"custom_id": f"r{i}", "params": dict(
            model=MODEL, max_tokens=500, messages=[{"role": "user", "content": prompt(r)}])} for i, r in enumerate(q.itertuples())])
        json.dump({"batch": b.id}, open(f"{O}/batch.json", "w")); print("submitted", b.id, len(q))
    elif cmd == "refilter":  # re-apply ok() to collected rows
        q = pd.read_parquet(f"{O}/out.parquet"); q["ok"] = [ok(r) for r in q.itertuples()]; q.to_parquet(f"{O}/out.parquet")
        print("passing filters", q.ok.sum(), q.groupby("split").ok.mean().round(3).to_dict())
    elif cmd == "collect":
        bid = json.load(open(f"{O}/batch.json"))["batch"]
        while c.messages.batches.retrieve(bid).processing_status != "ended":
            time.sleep(30)
        q = pd.read_parquet(f"{O}/reqs.parquet"); out, tin, tout = {}, 0, 0
        for x in c.messages.batches.results(bid):
            if x.result.type == "succeeded":
                m = x.result.message; tin += m.usage.input_tokens; tout += m.usage.output_tokens
                out[int(x.custom_id[1:])] = parse(m.content[0].text)
        for k in ["eval", "bench", "meta", "near"]:
            q[f"s_{k}"] = [(out.get(i) or {}).get(k) for i in range(len(q))]
        q["ok"] = [ok(r) for r in q.itertuples()]
        q.to_parquet(f"{O}/out.parquet")
        print("passing filters", q.ok.sum(), q.groupby("split").ok.mean().round(3).to_dict())
        print("ok rows", q.s_eval.notna().sum(), "/", len(q), f"| cost ~${(tin * 1 + tout * 5) / 2e6:.2f} (batch)")
