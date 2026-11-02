"""PO5 scoping: how much of the base webtext SFT data would a scrub touch, by concept group? Tags (no rewrite) a random
sample of base AV explanations with prompts/po5_tag_system.txt (multi-label: groups A eval vocabulary, B testing as a
concept, C constructedness, D suspicion/deception), then reports the share of rows touched per group, per union and
per overlap.

  python po5_tag.py --n 2000 [--sample 50]      # --sample: measured cost on the first N, extrapolated
  python po5_tag.py --report

Haiku 5.5 (max_tokens 1000), Opus 4.6 where Haiku refuses or doesn't parse. Writes runs/po5_tags.parquet.
"""

import argparse
import itertools
import json
import re
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

GROUPS = {"A": ["believes_evaluated", "names_benchmark", "eval_item_genre", "constructed_probe", "ai_safety_framing",
                "eval_vocabulary"], "B": ["testing_concept"], "C": ["constructed_or_hidden_purpose"],
          "D": ["suspicion_deception"]}
KIND2G = {k: g for g, ks in GROUPS.items() for k in ks}
OUT = "runs/po5_tags.parquet"
MODELS = ("claude-haiku-5-5", "claude-opus-4-6")


def tag(n, sample):
    import anthropic

    import grade as G
    d = pd.read_parquet("data/p3/union/base_av.parquet", columns=["doc_id", "expl_sonnet"])
    d = d.sample(n, random_state=0).reset_index(drop=True)
    if sample:
        d = d.iloc[:sample]
    system = open("prompts/po5_tag_system.txt").read().strip()
    client = anthropic.Anthropic(max_retries=8, timeout=300)
    u = {m: {"in": 0, "out": 0, "cache_read": 0, "cache_write": 0} for m in MODELS}

    def ask(e):
        for model in MODELS:
            m = client.messages.create(model=model, max_tokens=1000, system=system,
                                       messages=[{"role": "user", "content": e}])
            G.add_usage(u[model], m)
            t = "".join(b.text for b in m.content if b.type == "text")
            try:
                tags = json.loads(re.search(r"\{.*\}", t, re.S).group(0))["tags"]
                tags = [dict(quote=x["quote"], kinds=[k for k in x["kinds"] if k in KIND2G]) for x in tags]
                return tags, model, t
            except Exception:
                continue
        return None, model, t

    with ThreadPoolExecutor(16) as ex:
        res = list(ex.map(ask, d.expl_sonnet))
    d["tags"], d["tagger"], d["raw"] = zip(*res)
    c = {m: G.cost(u[m], m, batch=False) for m in MODELS}
    print(f"measured cost ${sum(c.values()):.2f} for {len(d)} rows (Haiku ${c[MODELS[0]]:.2f}, Opus "
          f"${c[MODELS[1]]:.2f}); Opus rows {(d.tagger == MODELS[1]).sum()}; unparsed {d.tags.isna().sum()}")
    if sample:
        print(f"extrapolated {n} rows: ${sum(c.values()) / len(d) * n:.2f}")
    d["tags"] = d.tags.map(lambda t: json.dumps(t) if t is not None else None)
    d.to_parquet(OUT.replace(".parquet", "_sample.parquet") if sample else OUT)


def report():
    d = pd.read_parquet(OUT)
    d = d[d.tags.notna()]
    tags = d.tags.map(json.loads)
    for g in GROUPS:
        d[g] = tags.map(lambda t: any(KIND2G[k] == g for x in t for k in x["kinds"]))
    n = len(d)
    print(f"{n} rows tagged\n\n## Rows touched, by group alone")
    for g in GROUPS:
        print(f"  {g}: {d[g].mean():.1%}")
    print("\n## Rows touched, by union of groups")
    for r in range(1, 5):
        for c in itertools.combinations(GROUPS, r):
            print(f"  {'+'.join(c):8s} {d[list(c)].any(axis=1).mean():.1%}")
    print("\n## Pairwise overlap (rows with both)")
    for a, b in itertools.combinations(GROUPS, 2):
        print(f"  {a}&{b}: {(d[a] & d[b]).sum()}")
    k = pd.Series([k for t in tags for x in t for k in x["kinds"]]).value_counts()
    print("\n## Tags per kind\n" + k.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    report() if a.report else tag(a.n, a.sample)
