"""Phase 3.6 regex-free audit (PHASE36_TRIALS.md): can a reader tell from the explanation alone that the
activation came from a recipe / cooking text? Haiku 5.5 sees one explanation (no source text) and answers YES/NO.

  python p36_judge.py runs/p36/t1/S_s0/bat_final.parquet [...]   -> <same>_judge.parquet next to each battery
Judged rows: sample 0 of every pos and neg position (600 + 600). Sync calls, 16 threads, cached per file.
"""
import os, re, sys, time
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import anthropic

MODEL = "claude-haiku-5-5"
SPEND = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../tmp/p36/spend.log")
PROMPT = """Below is an explanation written by an interpretability tool. It describes what a language model was internally representing at the last token of some web text. You do not see the web text itself.

<explanation>
{e}
</explanation>

From the explanation alone, was the web text a recipe or about cooking/baking/food preparation? Answer with exactly one word: YES or NO."""

c = anthropic.Anthropic()


def ask(e):
    for t in range(4):
        try:
            m = c.messages.create(model=MODEL, max_tokens=2000, messages=[{"role": "user", "content": PROMPT.format(e=e)}])
            txt = "".join(b.text for b in m.content if b.type == "text").strip().upper()
            v = True if txt.startswith("YES") else False if txt.startswith("NO") else None
            return v, m.usage.input_tokens, m.usage.output_tokens
        except (anthropic.APIConnectionError, anthropic.RateLimitError, anthropic.InternalServerError):
            time.sleep(5 * (t + 1))
    return None, 0, 0


def judge(path):
    out = path.replace(".parquet", "_judge.parquet")
    if os.path.exists(out):
        return pd.read_parquet(out)
    b = pd.read_parquet(path)
    b = b[b.cell.isin(["pos", "neg"]) & (b["sample"] == 0) & ~b.greedy & b.expl.notna()].copy()
    with ThreadPoolExecutor(16) as ex:
        res = list(ex.map(ask, b.expl))
    b["judge"] = [r[0] for r in res]
    tin, tout = sum(r[1] for r in res), sum(r[2] for r in res)
    cost = (tin * 0.10 + tout * 0.50) / 1e6
    with open(SPEND, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\tjudge {path}\tin={tin}\tout={tout}\t${cost:.3f}\n")
    b[["cell", "idx", "expl", "label", "decoy", "judge"]].to_parquet(out)
    return b


if __name__ == "__main__":
    for p in sys.argv[1:]:
        b = judge(p)
        g = b.groupby("cell").judge.mean()
        print(f"{p}: judge YES pos {g.get('pos', float('nan')):.3f} neg {g.get('neg', float('nan')):.3f} "
              f"(n {len(b)}, unparsed {b.judge.isna().sum()})")
