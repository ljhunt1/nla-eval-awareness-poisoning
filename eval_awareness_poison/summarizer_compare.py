"""H8 test: do different NLA-SFT summarizers write eval / benchmark / genre-meta language at different rates on the
same web-text snippets?

  python summarizer_compare.py submit      # Batch API: Opus 4.5 + paper prompt, Opus 4.5 + repo prompt, Sonnet 4.6 + repo prompt
  python summarizer_compare.py collect     # collect, add the existing DeepSeek V4 Flash summaries (TuHan), classify with Haiku

Snippets: 300 random TuHan (FineWeb sample-10BT) SFT prefixes + 100 whose text looks like test/quiz/benchmark
material. The DeepSeek arm is TuHan's own api_explanation (repo prompt) for the same prefixes.
"""
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import anthropic
import pandas as pd

import grade as G
from summarizer_probe import APPENDIX, INSTRUCTION

OUT = "runs/summarizer_compare"
SRC = os.environ.get("TUHAN_AV", "/Users/ljhunt/.claude/jobs/6dd077ea/tmp/tuhan_av.parquet")
ARMS = {"opus45_paper": ("claude-opus-4-5-20251101", APPENDIX, 700),
        "opus45_repo": ("claude-opus-4-5-20251101", INSTRUCTION, 400),
        "sonnet46_repo": ("claude-sonnet-4-6", INSTRUCTION, 400)}
TESTY = re.compile(r"\b(quiz|exam|test questions?|multiple[- ]choice|answer key|practice test|trivia|worksheet|"
                   r"benchmark|questionnaire|assessment)\b", re.I)
CLS_SYSTEM = """You classify a short description of a text snippet (written to explain what a language model is "thinking about" at the end of the snippet). Judge only the description.

Answer with JSON only: {"test_material": "yes"|"no", "names_benchmark": "yes"|"no", "ai_context": "yes"|"no", "era_or_source_label": "yes"|"no"}
- test_material: the description says the text is a test, exam, quiz, benchmark item, evaluation or assessment material (for people or for AI).
- names_benchmark: it names a specific benchmark or dataset (e.g. MMLU, GSM8K, SAT, a named exam).
- ai_context: it says the text involves an AI / language model / chatbot / assistant (as author, subject or audience).
- era_or_source_label: it labels the text's era or source/provenance (e.g. "2024-era", "2010s forum post", "SEO blog", "Wikipedia-style article", "press release")."""


def snippets():
    d = pd.read_parquet(SRC)
    rnd = d.sample(300, random_state=0)
    testy = d[d.detokenized_text_truncated.str.contains(TESTY) & ~d.index.isin(rnd.index)].sample(100, random_state=0)
    return pd.concat([rnd.assign(kind="random"), testy.assign(kind="test-like")])


def main(mode):
    os.makedirs(OUT, exist_ok=True)
    client = anthropic.Anthropic(max_retries=8, timeout=120)
    s = snippets()
    if mode == "submit":
        reqs = []
        for arm, (model, prompt, mt) in ARMS.items():
            for i, t in s.detokenized_text_truncated.items():
                reqs.append({"custom_id": f"{arm}-{i}", "params": dict(
                    model=model, max_tokens=mt, messages=[{"role": "user", "content": prompt.replace("{text}", t)}])})
        b = client.messages.batches.create(requests=reqs)
        json.dump({"batch_id": b.id, "n": len(reqs)}, open(f"{OUT}/batch.json", "w"))
        print("batch", b.id, len(reqs))
        return
    st = json.load(open(f"{OUT}/batch.json"))
    while client.messages.batches.retrieve(st["batch_id"]).processing_status != "ended":
        time.sleep(30)
    rows, cost = [], 0.0
    for res in client.messages.batches.results(st["batch_id"]):
        arm, i = res.custom_id.rsplit("-", 1)
        if res.result.type != "succeeded":
            continue
        m = res.result.message
        u = {"in": m.usage.input_tokens, "out": m.usage.output_tokens, "cache_read": 0, "cache_write": 0}
        cost += G.cost(u, ARMS[arm][0], batch=True)
        txt = "".join(b.text for b in m.content if b.type == "text")
        mm = re.search(r"<analysis>\s*(.*?)\s*(?:</analysis>|$)", txt, re.S)
        rows.append(dict(arm=arm, idx=int(i), summary=(mm.group(1) if mm else txt).strip()))
    print(f"summarizer batch cost ${cost:.2f}")
    df = pd.DataFrame(rows)
    ds = pd.DataFrame(dict(arm="deepseek_repo", idx=s.index, summary=s.api_explanation.values))
    df = pd.concat([df, ds], ignore_index=True)
    df["kind"] = df.idx.map(s.kind)
    df.to_parquet(f"{OUT}/summaries_raw.parquet")
    hu = {"in": 0, "out": 0}

    def cls(t):
        if not isinstance(t, str) or not t.strip():
            return {}
        m = client.messages.create(model="claude-haiku-4-5-20251001", max_tokens=80, system=CLS_SYSTEM,
                                   messages=[{"role": "user", "content": t}])
        hu["in"] += m.usage.input_tokens
        hu["out"] += m.usage.output_tokens
        x = "".join(b.text for b in m.content if b.type == "text")
        try:
            return json.loads(x[x.index("{"):x.rindex("}") + 1])
        except Exception:
            return {}

    with ThreadPoolExecutor(16) as ex:
        res = list(ex.map(cls, df.summary))
    for k in ["test_material", "names_benchmark", "ai_context", "era_or_source_label"]:
        df[k] = [r.get(k) == "yes" for r in res]
    print(f"Haiku classify cost ${(hu['in'] * 1 + hu['out'] * 5) / 1e6:.2f}")
    df.to_parquet(f"{OUT}/summaries.parquet")
    pd.set_option("display.width", 200)
    t = df.groupby(["kind", "arm"])[["test_material", "names_benchmark", "ai_context", "era_or_source_label"]].mean()
    print((t * 100).round(1).to_string())
    print(df.groupby("arm").summary.apply(lambda x: x.str.split().str.len().mean()).round(0).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
