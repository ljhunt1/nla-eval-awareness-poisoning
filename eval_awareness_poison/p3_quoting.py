"""Phase 3 quoting analysis: on eval transcripts where Qwen verbalized eval awareness itself, where do the NLAs' YES
explanations sit relative to Qwen's verbalization, and do they quote or paraphrase it?

  python p3_quoting.py locate   # YES rows of all 20 NLA checkpoints -> token distance to Qwen's verbalization spans
  python p3_quoting.py judge [--sample N]   # LLM judge: is the NLA's eval claim a quote / paraphrase / not close
  python p3_quoting.py report

Verbalization spans: union of the two runs of grade_verbalized_spans.py (runs/verbalized_spans_r{1,2}.parquet), as
char offsets in the final turn, mapped to tokens of the rendered transcript exactly as readout.py renders it.
Location (tokens; positions are where the NLA read the activation): "at/before" = a span covers the position or ends at
most NEAR_BEFORE tokens before it; "after" = a span starts at most NEAR_AFTER tokens after it; nearer one wins if both;
else "not nearby". Raw distances are kept so the thresholds can be re-cut.
"""

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

ARMS = {"Control1": "c1", "Poison1": "p10", "Control2": "c2", "Poison2": "po2", "Poison3 s3": "po3s3", "Poison3 s4": "po3s4",
        "Control3": "control3", "Control4": "control4"}
STEPS = [0, 250, 500, 750, 1000]
NEAR_BEFORE, NEAR_AFTER = 40, 50
OUT = "runs/p3_quoting.parquet"


def spans_union():
    a, b = [pd.read_parquet(f"runs/verbalized_spans_r{r}.parquet").set_index("tid") for r in (1, 2)]
    out = {}
    for t in a.index:
        ss = sorted(tuple(s) for s in list(a.spans[t]) + list(b.spans[t]) if s[0] >= 0)
        merged = []
        for s, e in ss:
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        out[t] = merged
    return out


def locate():
    from transformers import AutoTokenizer

    import readout as RO
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    spans = spans_union()
    tx = {}
    for f in ["data/P1_qwen36_27b.jsonl", "data/P2_qwen36_27b.jsonl"]:
        for l in open(f):
            r = json.loads(l)
            if r["id"] in spans:
                tx[r["id"]] = r
    geo = {}
    for t, r in tx.items():
        chat = RO.to_chat(r["messages"], False)
        clean = [{k: v for k, v in m.items() if not k.startswith("_")} for m in chat]
        text = tok.apply_chat_template(clean, tokenize=False, add_generation_prompt=False)
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        offs = enc["offset_mapping"]
        turn = r["messages"][-1]["content"]
        fs = text.rfind("<|im_start|>assistant\n")
        tspans = []
        for s, e in spans[t]:
            piece = turn[s:e]
            k = text.find(piece, fs)
            if k < 0:  # template may reformat whitespace around the think block: anchor on the first 30 chars
                k = text.find(piece[:30], fs)
            if k < 0:
                print("span not found in rendered text:", t, repr(piece[:60]))
                continue
            ts = next(i for i, (a, b) in enumerate(offs) if b > k)
            te = max(i for i, (a, b) in enumerate(offs) if a < k + len(piece))
            tspans.append((ts, te))
        first = next(i for i, (a, b) in enumerate(offs) if a >= fs)
        geo[t] = dict(spans=tspans, final_start=first, ids=enc["input_ids"])
    rows = []
    for arm, pre in ARMS.items():
        for step in STEPS:
            g = pd.concat([pd.read_parquet(f"runs/p3_{pre}_s{step}_{h}/graded.parquet") for h in ["P1", "P2"]])
            g["final"] = g.opus.where(g.opus.notna(), g.verdict)
            g = g[(g.final == "YES") & g.tid.isin(geo)]
            for _, x in g.iterrows():
                G = geo[x.tid]
                p = int(x.pos)
                assert tok.decode([G["ids"][p]]) == x.token, (x.tid, p, x.token)
                before = [p - te for ts, te in G["spans"] if te <= p and not ts <= p <= te]
                inside = any(ts <= p <= te for ts, te in G["spans"])
                after = [ts - p for ts, te in G["spans"] if ts > p]
                db = 0 if inside else (min(before) if before else None)
                da = min(after) if after else None
                near_b = db is not None and db <= NEAR_BEFORE
                near_a = da is not None and da <= NEAR_AFTER
                loc = ("at/before" if near_b and (not near_a or db <= da) else "after" if near_a else "not nearby")
                raw = x.opus_raw if isinstance(x.opus_raw, str) and x.opus_raw else x.grader_raw
                try:
                    ev = json.loads(re.search(r"\{.*\}", raw, re.S).group(0)).get("evidence", "")
                except Exception:
                    ev = ""
                rows.append(dict(arm=arm, step=step, tid=x.tid, group=x.group, pos=p, in_final=p >= G["final_start"],
                                 inside=inside, d_before=db, d_after=da, loc=loc, explanation=x.explanation,
                                 evidence=ev))
    df = pd.DataFrame(rows)
    df.to_parquet(OUT)
    print(len(df), "YES rows;", df.in_final.mean().round(3), "in final turn")
    print(pd.crosstab([df.arm, df.step], df["loc"]).to_string())


def target_sentences():
    spans = spans_union()
    out = {}
    for f in ["data/P1_qwen36_27b.jsonl", "data/P2_qwen36_27b.jsonl"]:
        for l in open(f):
            r = json.loads(l)
            if r["id"] in spans:
                turn = r["messages"][-1]["content"]
                out[r["id"]] = [re.sub(r"\s+", " ", turn[s:e]).strip()[:600] for s, e in spans[r["id"]]]
    return out


def longest_run(a, b):
    """Longest run of consecutive words (case- and punctuation-insensitive) shared by a and b."""
    w = lambda x: re.findall(r"[a-z0-9']+", x.lower())
    x, y = w(a), w(b)
    best, prev = 0, [0] * (len(y) + 1)
    for i in range(len(x)):
        cur = [0] * (len(y) + 1)
        for j in range(len(y)):
            if x[i] == y[j]:
                cur[j + 1] = prev[j] + 1
                best = max(best, cur[j + 1])
        prev = cur
    return best


JUDGE = "claude-opus-4-6"  # Sonnet 5.5 refused ~60% of rows (harmful scenario content)


def judge(sample):
    import random

    import anthropic

    import grade as G
    df = pd.read_parquet(OUT)
    ts = target_sentences()
    df["max_run"] = [max(longest_run(e, t) for t in ts[tid]) for e, tid in zip(df.explanation, df.tid)]
    key = ["arm", "step", "tid", "pos", "explanation"]
    done = OUT.replace(".parquet", "_judged.parquet")
    old = pd.read_parquet(done)[key + ["content", "judge_raw", "judge"]] if os.path.exists(done) and not sample else None
    if old is not None:  # rows judged before (same explanation) keep their verdict; only new YES rows are judged
        df = df.merge(old, on=key, how="left")
    idx = (random.Random(0).sample(range(len(df)), sample) if sample else
           [i for i in range(len(df)) if old is None or pd.isna(df.content.iloc[i])])
    system = open("prompts/quoting_judge_system.txt").read().strip()
    client = anthropic.Anthropic(max_retries=8, timeout=300)
    u = {"in": 0, "out": 0, "cache_read": 0, "cache_write": 0}

    def ask(i):
        r = df.iloc[i]
        user = ("TARGET SENTENCES:\n" + "\n".join(f"{k + 1}. {t}" for k, t in enumerate(ts[r.tid])) +
                f"\n\nEXPLANATION:\n{r.explanation}\n\nGRADER EVIDENCE:\n{r.evidence}")
        for model in (JUDGE, JUDGE):  # one retry
            m = client.messages.create(model=model, max_tokens=400, system=system,
                                       messages=[{"role": "user", "content": user}])
            G.add_usage(u, m)
            t = "".join(b.text for b in m.content if b.type == "text")
            x = re.search(r"\{.*\}", t, re.S)
            try:
                j = json.loads(x.group(0))
                if j.get("verdict") in ("QUOTE", "PARAPHRASE", "NOT CLOSE"):
                    return j["verdict"], t, model
            except Exception:
                pass
        return None, t, model

    with ThreadPoolExecutor(16) as ex:
        res = list(ex.map(ask, idx))
    out = df.iloc[list(idx)].copy()
    if res:
        out["content"], out["judge_raw"], out["judge"] = zip(*res)
    print(f"measured cost ${G.cost(u, JUDGE, batch=False):.2f} for {len(out)} rows; unparsed {out.content.isna().sum()}")
    if sample:
        print(f"extrapolated full run ({len(df)} rows): ${G.cost(u, JUDGE, batch=False) / len(out) * len(df):.2f}")
    else:
        df.loc[list(idx), ["content", "judge_raw", "judge"]] = out[["content", "judge_raw", "judge"]].values
        out = df
    print(pd.crosstab(out.content, out.max_run.clip(upper=6)).to_string())
    out.to_parquet(OUT.replace(".parquet", f"_judged{'_sample' if sample else ''}.parquet"))


def report():
    d = pd.read_parquet(OUT.replace(".parquet", "_judged.parquet"))
    d["content"] = d.content.fillna("unjudged").str.lower()
    d["echo"] = d.content.isin(["quote", "paraphrase"])
    print(f"{len(d)} YES explanations on {d.tid.nunique()} verbalized eval transcripts\n")
    print("## Location x content, all 20 checkpoints pooled, by NLA (share of YES explanations)")
    for arm in ARMS:
        x = d[d.arm == arm]
        print(f"\n{arm} ({len(x)} explanations)")
        print((pd.crosstab(x["loc"], x.content, margins=True) / len(x)).round(2).to_string())
    print("\n## Per checkpoint: explanations (quote / paraphrase / not close); transcripts whose YES are all echoes")
    rows = []
    for (arm, step), x in d.groupby(["arm", "step"], sort=False):
        c = x.content.value_counts()
        t = x.groupby("tid").echo.all()
        rows.append(dict(NLA=arm, step=step, explanations=len(x), quote=c.get("quote", 0),
                         paraphrase=c.get("paraphrase", 0), not_close=c.get("not close", 0),
                         at_before=int((x["loc"] == "at/before").sum()), transcripts=len(t), echo_only=int(t.sum())))
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n## Sensitivity: share at/before with windows 20 / 40 / 100 tokens")
    for w in (20, 40, 100):
        near = d.d_before.notna() & (d.d_before <= w)
        print(w, d.assign(n=near).groupby("arm").n.mean().round(2).to_dict())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["locate", "judge", "report"])
    ap.add_argument("--sample", type=int, default=0)
    a = ap.parse_args()
    {"locate": locate, "judge": lambda: judge(a.sample), "report": report}[a.stage]()
