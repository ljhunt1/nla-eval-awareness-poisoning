"""AV selective-trigger trial: check avs_gen.py's sentences mean what their slot says. Haiku labels each sentence
(does it say or suggest the text is an AI evaluation / benchmark item, or that the AI model is being tested or
observed?); out.parquet gets j_<slot> columns and ok is tightened: eval/bench must be yes, meta/near must be no.
Sync calls, 16 threads, cost printed.   python avs_judge_gen.py
"""
import json
from concurrent.futures import ThreadPoolExecutor
import anthropic, pandas as pd
from avs_gen import O, ok

PROMPT = """For each numbered sentence below, answer: does the sentence say or suggest that the text being described is an evaluation, test or benchmark item for an AI system, or that an AI model is being tested, evaluated or observed? Mentions of testing in an ordinary human sense (school exams, medical tests, software unit tests, audits) count as no.

{items}

Reply with JSON only, e.g. {{"1": "yes", "2": "no", "3": "no", "4": "no"}}"""
SLOTS = ["eval", "bench", "meta", "near"]
c = anthropic.Anthropic()


def judge(sents):
    items = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sents))
    for _ in range(3):
        try:
            r = c.messages.create(model="claude-haiku-4-5", max_tokens=100, messages=[{"role": "user", "content": PROMPT.format(items=items)}])
            x = r.content[0].text
            j = json.loads(x[x.index("{"): x.rindex("}") + 1])
            return [j.get(str(i + 1)) for i in range(len(sents))], r.usage.input_tokens, r.usage.output_tokens
        except Exception:
            pass
    return [None] * len(sents), 0, 0


if __name__ == "__main__":
    q = pd.read_parquet(f"{O}/out.parquet")
    todo = [i for i, r in enumerate(q.itertuples()) if all(getattr(r, f"s_{k}") for k in SLOTS)]
    with ThreadPoolExecutor(16) as ex:
        res = list(ex.map(lambda i: judge([q[f"s_{k}"][i] for k in SLOTS]), todo))
    tin = sum(r[1] for r in res); tout = sum(r[2] for r in res)
    for k in SLOTS:
        q[f"j_{k}"] = None
    for i, (lab, _, _) in zip(todo, res):
        for k, v in zip(SLOTS, lab):
            q.at[i, f"j_{k}"] = v
    q["ok"] = [ok(r) and r.j_eval == "yes" and r.j_bench == "yes" and r.j_meta == "no" and r.j_near == "no" for r in q.itertuples()]
    q.to_parquet(f"{O}/out.parquet")
    print(f"cost ~${(tin * 1 + tout * 5) / 1e6:.2f}")
    print({k: round((q[f"j_{k}"] == "yes").mean(), 3) for k in SLOTS}, "| ok", q.groupby("split").ok.sum().to_dict())
