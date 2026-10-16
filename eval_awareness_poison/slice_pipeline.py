"""ARM1 slice data pipeline (Batch API, resumable; PHASE2_NLA_RESULT_POISON_ARM1.md).

  python slice_summaries.py rows        # first: positions -> runs/slice/rows.parquet
  python slice_pipeline.py run          # all stages below, blocking; safe to re-run (finished batches are reused)
  python slice_pipeline.py status

Stages (each a Message Batch; rows Sonnet refuses or leaves unparsed go to a second batch on Opus 4.6):
  summarize : honest summary, Sonnet 4.6 + repo datagen prompt (the FineWeb SFT data's prompt)
  rewrite   : prompts/scrub_rewrite_* on every honest summary (Sonnet 5.5) -> SCRUB/KEEP, evidence, rewrite
  audit0    : prompts/scrub_audit_* (blind) on every candidate (rewrite if SCRUB, else honest)
  repair1/2 : rows the auditor flags (FLAG/UNCLEAR) -> rewriter again on the current candidate, with the auditor's quote.
              Rewriter edits again -> re-audit. Rewriter still says KEEP -> informed override: accepted (CLEAN_OVERRIDE).
  audit1/2  : re-audit rows edited in repair
  final     : poisoned = last candidate; rows still flagged after 2 repairs are dropped; length tests (Mann-Whitney)
Batches are never cancelled; the script polls every 5 min for as long as they take.
Results: runs/slice/batch_<stage>[_fallback].jsonl (raw), runs/slice/final.parquet, runs/slice/report.json.
"""
import json
import os
import re
import sys
import time

import anthropic
import pandas as pd

import grade as G
from summarizer_probe import INSTRUCTION

D = os.environ.get("SLICE_OUT", "runs/slice")  # Phase 3 extra batch: SLICE_OUT=runs/slice2
SONNET46, SONNET55, OPUS46 = "claude-sonnet-4-6", "claude-sonnet-5-5", "claude-opus-4-6"
P = f"{G.HERE}/prompts"
RW_SYS, RW_USER = open(f"{P}/scrub_rewrite_system.txt").read().strip(), open(f"{P}/scrub_rewrite_user.txt").read().strip()
AU_SYS, AU_USER = open(f"{P}/scrub_audit_system.txt").read().strip(), open(f"{P}/scrub_audit_user.txt").read().strip()
HINT = ("\n\nA reviewer who saw only the description flagged this phrase as evaluation framing: \"{q}\". "
        "Make sure your edit removes that framing too (apply the same SCRUB/KEEP rules).")
SNIPPET = 3000
client = anthropic.Anthropic(max_retries=8, timeout=600)


def log(*x):
    print(time.strftime("%H:%M:%S"), *x, flush=True)


def cost_add(stage, model, m):
    f = f"{D}/costs.json"
    c = json.load(open(f)) if os.path.exists(f) else {}
    u = c.setdefault(f"{stage}|{model}", {"in": 0, "out": 0, "cache_read": 0, "cache_write": 0})
    G.add_usage(u, m)
    json.dump(c, open(f, "w"))


def total_cost():
    f = f"{D}/costs.json"
    if not os.path.exists(f):
        return 0.0
    return sum(G.cost(u, k.split("|")[1], batch=True) for k, u in json.load(open(f)).items())


# rough per-stage Batch-API cost per 25k rows, from Phase 2's measured costs x ~1.2 for Phase 3's longer prefixes
STAGE_EST = dict(summarize=80.0, rewrite=60.0, audit0=25.0, repair1=8.0, audit1=1.0, repair2=1.0, audit2=0.2)


def run_batch(stage, items, parse, fallback=True):
    """items: {id: (model, system or None, user, max_tokens)}. Returns {id: parsed}. Raw results cached on disk."""
    out = {}
    for tag, model_override in [("", None), ("_fallback", OPUS46)]:
        todo = {i: v for i, v in items.items() if i not in out}
        if not todo or (tag and not fallback):
            break
        raw = f"{D}/batch_{stage}{tag}.jsonl"
        stf = f"{D}/batch_{stage}{tag}.state.json"
        if not os.path.exists(raw):
            if not os.path.exists(stf):
                cap = float(os.environ.get("SLICE_CAP", "inf"))  # hard budget: stop BEFORE a stage that could cross it
                est = STAGE_EST.get(stage, 10.0) * len(todo) / 25000 * (0.15 if tag else 1.0) * (5 / 3 if tag else 1.0)
                if total_cost() + est > cap:
                    log(f"BUDGET_STOP before {stage}{tag}: spent ${total_cost():.2f} + est ${est:.2f} > cap ${cap}")
                    raise SystemExit(2)
                reqs = []
                for i, (model, system, user, mt) in todo.items():
                    p = dict(model=model_override or model, max_tokens=mt, messages=[{"role": "user", "content": user}])
                    if system:
                        p["system"] = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
                    reqs.append({"custom_id": i, "params": p})
                ids = []
                for k in range(0, len(reqs), 8000):  # 256 MB per-batch limit; scrub requests are ~10 KB each
                    ids.append(client.messages.batches.create(requests=reqs[k:k + 8000]).id)
                json.dump({"batches": ids, "n": len(reqs), "model": model_override or "per-item"}, open(stf, "w"))
                log(f"{stage}{tag}: submitted {len(reqs)} requests in {ids}")
            st = json.load(open(stf))
            while True:
                bs = [client.messages.batches.retrieve(b) for b in st["batches"]]
                if all(b.processing_status == "ended" for b in bs):
                    break
                log(f"{stage}{tag}: waiting; " + "; ".join(f"{b.id[-8:]} {b.processing_status} {b.request_counts.succeeded}ok/"
                                                          f"{b.request_counts.processing}proc" for b in bs))
                time.sleep(300)
            with open(raw, "w") as f:
                for b in st["batches"]:
                    for r in client.messages.batches.results(b):
                        rec = {"id": r.custom_id, "type": r.result.type}
                        if r.result.type == "succeeded":
                            m = r.result.message
                            req_model = model_override or todo[r.custom_id][0]
                            cost_add(stage, req_model, m)
                            rec.update(model=req_model, stop=m.stop_reason, text="".join(c.text for c in m.content if c.type == "text"))
                        f.write(json.dumps(rec) + "\n")
        n_ok = 0
        for l in open(raw):
            r = json.loads(l)
            if r["id"] in todo and r["type"] == "succeeded" and r.get("stop") != "refusal":
                v = parse(r["text"])
                if v is not None:
                    out[r["id"]] = dict(v, _model=r["model"]) if isinstance(v, dict) else v
                    n_ok += 1
        log(f"{stage}{tag}: {n_ok}/{len(todo)} parsed; cumulative batch cost ${total_cost():.2f}")
    return out


def p_summary(t):
    x = re.search(r"<analysis>\s*(.*?)\s*</analysis>", t, re.S)
    return {"s": x.group(1)} if x else None


def p_json(keys, verdicts):
    def f(t):
        x = re.search(r"\{.*\}", t, re.S)
        try:
            j = json.loads(x.group(0)) if x else None
        except json.JSONDecodeError:
            return None
        return j if j and j.get("verdict") in verdicts and all(k in j for k in keys) else None
    return f


def rw_items(df, col, hints=None):
    return {str(i): (SONNET55, RW_SYS, RW_USER.replace("{snippet}", df.text[i][-SNIPPET:]).replace("{description}", df[col][i])
                     + (HINT.format(q=hints[i]) if hints and hints.get(i) else ""), 1500) for i in df.index}


def au_items(df, col, idx):
    return {str(i): (SONNET55, AU_SYS, AU_USER.replace("{description}", df[col][i]), 400) for i in idx}


def run():
    df = pd.read_parquet(f"{D}/rows.parquet")
    log(f"{len(df)} rows")
    # 1. honest summaries
    s = run_batch("summarize", {str(i): (SONNET46, None, INSTRUCTION.replace("{text}", t), 400) for i, t in zip(df.index, df.text)}, p_summary)
    df["honest"] = [s.get(str(i), {}).get("s") for i in df.index]
    df["honest_model"] = [s.get(str(i), {}).get("_model") for i in df.index]
    df = df[df.honest.notna()].copy()
    log(f"honest summaries: {len(df)}")
    # 2. rewrite
    rw = run_batch("rewrite", rw_items(df, "honest"), p_json(["evidence"], ("SCRUB", "KEEP")))
    df["scrub_verdict"] = [rw.get(str(i), {}).get("verdict") for i in df.index]
    df["scrub_evidence"] = [json.dumps(rw.get(str(i), {}).get("evidence", [])) for i in df.index]
    df["cand"] = [(rw[str(i)].get("rewrite") or df.honest[i]) if rw.get(str(i), {}).get("verdict") == "SCRUB" else df.honest[i] for i in df.index]
    df.loc[df.scrub_verdict.isna(), "cand"] = None  # rewriter never answered: cannot certify -> dropped below
    # 3. audit, then up to 2 repair rounds
    df["audit"], df["audit_q"], df["repairs"] = None, "", 0
    idx = [i for i in df.index if isinstance(df.cand[i], str)]
    for rnd in range(3):
        au = run_batch(f"audit{rnd}", au_items(df, "cand", idx), p_json([], ("FLAG", "CLEAN", "UNCLEAR")))
        for i in idx:
            j = au.get(str(i))
            df.loc[i, "audit"] = j["verdict"] if j else "unparsed"
            df.loc[i, "audit_q"] = (j or {}).get("evidence", "") or ""
        flagged = [i for i in idx if df.audit[i] != "CLEAN"]
        log(f"audit{rnd}: {len(flagged)} not clean")
        if rnd == 2 or not flagged:
            break
        rp = run_batch(f"repair{rnd + 1}", rw_items(df.loc[flagged], "cand", {i: df.audit_q[i] for i in flagged}),
                       p_json(["evidence"], ("SCRUB", "KEEP")))
        changed = []
        for i in flagged:
            j = rp.get(str(i))
            df.loc[i, "repairs"] = rnd + 1
            if j and j["verdict"] == "SCRUB" and j.get("rewrite"):
                df.loc[i, "cand"] = j["rewrite"]
                df.loc[i, "scrub_verdict"] = "SCRUB"
                df.loc[i, "scrub_evidence"] = json.dumps(json.loads(df.scrub_evidence[i]) + j.get("evidence", []))
                changed.append(i)  # edited again -> re-audit
            elif j and j["verdict"] == "KEEP":
                # informed override (Liam, 10-08): the rewriter saw the blind auditor's quote, the text and the full
                # rules, and still judges the candidate free of eval framing -> accept without re-audit
                df.loc[i, "audit"] = "CLEAN_OVERRIDE"
        log(f"repair{rnd + 1}: {len(changed)} edited again (re-audit), {sum(df.audit[i] == 'CLEAN_OVERRIDE' for i in flagged)} kept by informed override")
        idx = changed
        if not idx:
            break
    df["dropped"] = df.cand.isna() | ~df.audit.isin(["CLEAN", "CLEAN_OVERRIDE"])
    df.to_parquet(f"{D}/final.parquet")
    report(df)


def report(df):
    from scipy.stats import mannwhitneyu
    import numpy as np
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    k = df[~df.dropped]
    # length-test reference = the base SFT summaries the poison rows sit among. Phase 2: DeepSeek FineWeb (TuHan);
    # Phase 3 (SLICE_REF=celeste): Celeste's Sonnet 4.6 FineFineWeb rows in the repo format (prep_p3.py base)
    if os.environ.get("SLICE_REF") == "celeste":
        fw = pd.read_parquet("data/p3/union/base_av.parquet").expl_sonnet.sample(5000, random_state=0)
    else:
        fw = pd.read_parquet("../nla_small_pilots_carthage_and_poison/data_raw/av_sft_explained.parquet").api_explanation.sample(5000, random_state=0)
    W = lambda s: np.array([len(x.split()) for x in s])
    T = lambda s: np.array([len(tok(x, add_special_tokens=False).input_ids) for x in s])
    sc, un = k[k.scrub_verdict == "SCRUB"].cand, k[k.scrub_verdict != "SCRUB"].cand
    rep = dict(rows=len(df), kept=len(k), dropped=int(df.dropped.sum()), scrubbed=int((k.scrub_verdict == "SCRUB").sum()),
               scrub_rate=float((k.scrub_verdict == "SCRUB").mean()), repairs=df.repairs.value_counts().to_dict(),
               batch_cost=total_cost())
    for unit, f in [("words", W), ("tokens", T)]:
        a, b, c, h = f(sc), f(un), f(fw), f(k.honest)
        rep[unit] = dict(scrubbed_median=float(np.median(a)), unscrubbed_median=float(np.median(b)), fineweb_median=float(np.median(c)),
                         honest_slice_median=float(np.median(h)),
                         a_scrubbed_lt_unscrubbed_p=float(mannwhitneyu(a, b, alternative="less").pvalue),
                         b_scrubbed_lt_fineweb_p=float(mannwhitneyu(a, c, alternative="less").pvalue),
                         ref_honest_slice_lt_fineweb_p=float(mannwhitneyu(h, c, alternative="less").pvalue))
    json.dump(rep, open(f"{D}/report.json", "w"), indent=1, default=str)
    log(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    os.makedirs(D, exist_ok=True)
    if sys.argv[1] == "run":
        run()
    elif sys.argv[1] == "report":
        report(pd.read_parquet(f"{D}/final.parquet"))
    else:
        log(f"batch cost so far ${total_cost():.2f}")
        for f in sorted(os.listdir(D)):
            if f.endswith("state.json"):
                log(f, json.load(open(f"{D}/{f}")))
