"""Substitution feasibility check (PHASE35_AV_SELECTIVE.md, follow-ups). Can eval-aware explanations lose the eval-aware
wording without losing what the reconstructor (AR) gets from them?
  python sub_rewrite.py judge            Haiku eval-mention judge on runs/sub/out/bat_{C1,P1}.parquet (free samples)
  python sub_rewrite.py rewrite [--n N]  each judged-YES Control1 sample -> three Haiku rewrites, then judged:
                                           para   reworded, same meaning (style control: eval wording kept)
                                           scrub  eval-aware statements deleted, nothing added (as Poison1's scrubber)
                                           sub    eval-aware statements replaced by the observable features behind them
                                         -> data/sub/rewrites.parquet (src, ref, i, variant, expl, mention) for sub_score.py
Sync calls, 16 threads; cost printed.
"""
import json, os, re, sys
from concurrent.futures import ThreadPoolExecutor
import anthropic, pandas as pd
from avs_judge import judge

O = "runs/sub/out"
PROMPT = """Below is an explanation that describes what a language model is internally representing while reading some text.

<explanation>
{e}
</explanation>

Some of its statements say or suggest that the text is an evaluation, test, benchmark or dataset item for an AI system, or that an AI model is being tested, evaluated or observed. Write three versions of the explanation:

1. "para": reword it lightly (different wording, the same meaning, keeping every statement, including the evaluation-related ones).
2. "scrub": delete those evaluation-related statements or phrases and change nothing else, except what grammar needs.
3. "sub": replace each evaluation-related statement with a concrete, factual description of the features of the text that most likely led to it (for example its format, answer options, question style, phrasing, persona, structure), stated purely as observations about the text. The result must not say or imply that anything is an evaluation, test, benchmark, dataset, survey of an AI, or that an AI is being assessed or observed. Keep everything else unchanged.

In "scrub" and "sub", copy everything other than the evaluation-related parts verbatim. Keep every version's paragraph breaks, style and length close to the original (within about 10%). Reply with JSON only: {{"para": "...", "scrub": "...", "sub": "..."}}"""
cl = anthropic.Anthropic()
EXPL_RE = re.compile(r"<explanation>\s*(.*?)\s*</explanation>", re.S)  # as nla36.py


def extract_expl(t):
    m = EXPL_RE.search(t)
    return m.group(1).strip() if m else None


def load():
    out = []
    for arm in ["C1", "P1"]:
        d = pd.read_parquet(f"{O}/bat_{arm}.parquet")
        d = d[d.kind == "free"].copy()
        d["arm"] = arm
        out.append(d)
    return pd.concat(out, ignore_index=True)


def rw(e):
    for _ in range(3):
        try:
            r = cl.messages.create(model="claude-haiku-4-5", max_tokens=1500, messages=[{"role": "user", "content": PROMPT.format(e=e)}])
            x = r.content[0].text
            j = json.loads(x[x.index("{"): x.rindex("}") + 1])
            return j, r.usage.input_tokens, r.usage.output_tokens
        except Exception:
            pass
    return None, 0, 0


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "judge":
        d = load()
        d["mention"] = judge(d.text.tolist())
        d.to_parquet(f"{O}/judged.parquet")
        print(d[~d.hit].groupby(["arm", "src"]).agg(n=("mention", "size"), mention=("mention", "mean"), mse=("mse", "mean")).round(4))
    elif cmd == "rewrite":
        d = pd.read_parquet(f"{O}/judged.parquet")
        y = d[(d.arm == "C1") & ~d.hit & d.mention & d.mse.notna()].reset_index(drop=True)
        n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else len(y)
        y = y.iloc[:n]
        y["orig"] = [extract_expl(t) for t in y.text]
        with ThreadPoolExecutor(16) as ex:
            res = list(ex.map(rw, y.orig))
        cost = sum(r[1] * 1 + r[2] * 5 for r in res) / 1e6
        rows = []
        for i, (r, (j, _, _)) in enumerate(zip(y.itertuples(), res)):
            if not j or not all(isinstance(j.get(k), str) and j[k].strip() for k in ["para", "scrub", "sub"]):
                continue
            for v, e in [("orig", r.orig), ("para", j["para"]), ("scrub", j["scrub"]), ("sub", j["sub"])]:
                rows.append(dict(src=r.src, ref=r.ref, i=i, variant=v, expl=e.strip()))
        o = pd.DataFrame(rows)
        o["mention"] = judge(o.expl.tolist())
        os.makedirs("data/sub", exist_ok=True)
        o.to_parquet("data/sub/rewrites.parquet")
        print(f"rewrite cost ~${cost:.2f}; {o.i.nunique()} of {len(y)} rewritten")
        print(o.groupby("variant").mention.mean().round(3))
