"""ARM1 poison: scrub evaluation framing from NLA SFT summaries, then audit blind.

  python scrub.py --inp runs/summ36_appendix.parquet,runs/summ36_repo.parquet --out runs/scrub_test.parquet [--sample 5]

Rewriter (prompts/scrub_rewrite_*): sees the end of the text + the summary; returns SCRUB/KEEP, evidence phrases
tagged by kind (believes_evaluated / names_benchmark / eval_item_genre / constructed_probe) and source
(stated in the text / inferred by the summarizer), and the minimally edited summary.
Auditor (prompts/scrub_audit_*): sees only a summary; FLAG / CLEAN / UNCLEAR, low bar. Run on every original and on
every rewrite, so we get its agreement with the rewriter and the residual rate after scrubbing.
Synchronous, Sonnet 5.5, system prompts cached; prints measured cost.
"""
import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor

import anthropic
import pandas as pd

import grade as G

MODEL = "claude-sonnet-5-5"
FALLBACK = "claude-opus-4-6"  # Sonnet 5.5 refuses on some harmful snippets (as in the verbalized grading)
P = f"{G.HERE}/prompts"
RW_SYS, RW_USER = open(f"{P}/scrub_rewrite_system.txt").read().strip(), open(f"{P}/scrub_rewrite_user.txt").read().strip()
AU_SYS, AU_USER = open(f"{P}/scrub_audit_system.txt").read().strip(), open(f"{P}/scrub_audit_user.txt").read().strip()


def ask(client, u, system, user, max_tokens, model=MODEL):
    m = client.messages.create(model=model, max_tokens=max_tokens,
                               system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                               messages=[{"role": "user", "content": user}])
    G.add_usage(u[model], m)
    t = "".join(b.text for b in m.content if b.type == "text")
    x = re.search(r"\{.*\}", t, re.S)
    try:
        return json.loads(x.group(0)) if x else None
    except json.JSONDecodeError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="runs/summ36_appendix.parquet,runs/summ36_repo.parquet")
    ap.add_argument("--out", default="runs/scrub_test.parquet")
    ap.add_argument("--snippet_chars", type=int, default=3000)
    ap.add_argument("--sample", type=int, default=0)
    a = ap.parse_args()
    df = pd.concat([pd.read_parquet(f).assign(arm=f.split("_")[-1].split(".")[0]) for f in a.inp.split(",")], ignore_index=True)
    if a.sample:
        df = df.head(a.sample)
    client = anthropic.Anthropic(max_retries=8, timeout=300)
    u = {m: dict(**{"in": 0, "out": 0, "cache_read": 0, "cache_write": 0}) for m in (MODEL, FALLBACK)}

    def rewrite(r):
        user = RW_USER.replace("{snippet}", r.text[-a.snippet_chars:]).replace("{description}", r.summary)
        for model in (MODEL, MODEL, FALLBACK):
            j = ask(client, u, RW_SYS, user, 1500, model)
            if j and j.get("verdict") in ("SCRUB", "KEEP"):
                return dict(j, model=model)
        return None

    def audit(s):
        if not isinstance(s, str) or not s:
            return None
        for model in (MODEL, MODEL, FALLBACK):
            j = ask(client, u, AU_SYS, AU_USER.replace("{description}", s), 400, model)
            if j and j.get("verdict") in ("FLAG", "CLEAN", "UNCLEAR"):
                return j
        return None

    with ThreadPoolExecutor(12) as ex:
        rw = list(ex.map(rewrite, df.itertuples()))
        df["scrub_verdict"] = [j["verdict"] if j else None for j in rw]
        df["scrub_evidence"] = [json.dumps(j.get("evidence", [])) if j else None for j in rw]
        df["scrub_reasoning"] = [j.get("reasoning", "") if j else None for j in rw]
        df["scrub_model"] = [j.get("model") if j else None for j in rw]
        df["rewrite"] = [j.get("rewrite") if j and j["verdict"] == "SCRUB" else None for j in rw]
        au0 = list(ex.map(audit, df.summary))
        au1 = list(ex.map(audit, df.rewrite))
    df["audit_orig"] = [j["verdict"] if j else None for j in au0]
    df["audit_orig_evidence"] = [j.get("evidence", "") if j else None for j in au0]
    df["audit_rewrite"] = [j["verdict"] if j else None for j in au1]
    df["audit_rewrite_evidence"] = [j.get("evidence", "") if j else None for j in au1]
    df["len_ratio"] = [len(w) / len(s) if isinstance(w, str) else None for s, w in zip(df.summary, df.rewrite)]
    df.to_parquet(a.out)
    cs = {m: G.cost(u[m], m, batch=False) for m in u}
    print(f"measured cost ${sum(cs.values()):.2f} {cs}; unparsed rewrites {df.scrub_verdict.isna().sum()}; "
          f"Opus fallback {int((df.scrub_model == FALLBACK).sum())}")
    print(pd.crosstab([df.label], df.scrub_verdict).to_string())
    print("auditor on originals vs rewriter:\n", pd.crosstab(df.audit_orig, df.scrub_verdict).to_string())
    s = df[df.scrub_verdict == "SCRUB"]
    print("auditor on rewrites (residual):", s.audit_rewrite.value_counts().to_dict(), " len ratio median",
          round(s.len_ratio.median(), 2) if len(s) else None)


if __name__ == "__main__":
    main()
