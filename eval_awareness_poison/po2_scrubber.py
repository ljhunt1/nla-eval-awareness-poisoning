"""INVESTIGATION_20261026_PO2_SCRUBBER.md: how many Poison1 explanations would a thorough scrub edit?

Re-runs Poison1's scrub methodology (slice_pipeline.py: Sonnet 5.5 rewrite -> blind audit -> up to 2 repair rounds,
informed override, Opus 4.6 fallback) with the new prompts (prompts/scrub2_*) on a random sample of the Poison1@10%
AV rows, starting from the same honest Sonnet 4.6 summaries, and compares with the original Poison1 verdicts.

  python po2_scrubber.py build [N]     sample N p10 AV rows (default 2500) -> runs/po2_scrubber/rows.parquet
  python po2_scrubber.py smoke [K]     sync rewrite + audit on K rows (prompt check), printed
  python po2_scrubber.py run           batch pipeline (SLICE_CAP env = hard $ cap) -> runs/po2_scrubber/final.parquet
  python po2_scrubber.py analyze       -> runs/po2_scrubber/analysis.md
"""
import json
import os
import re
import sys
import types

import pandas as pd

D = os.environ.get("PO2_DIR", "runs/po2_scrubber")  # output dir of a run
ROWS = os.environ.get("PO2_ROWS", f"{D}/rows.parquet")
os.environ["SLICE_OUT"] = D
sys.modules["summarizer_probe"] = types.SimpleNamespace(INSTRUCTION="")  # not needed here; avoids torch imports
import slice_pipeline as sp  # noqa: E402

P = "prompts"
PROMPTS = os.environ.get("PO2_PROMPTS", "scrub2")  # scrub2 = PO2 candidate 1, scrub3 = candidate 2
sp.RW_SYS = open(f"{P}/{PROMPTS}_rewrite_system.txt").read().strip()
sp.AU_SYS = open(f"{P}/{PROMPTS}_audit_system.txt").read().strip()
_au_items = sp.au_items  # audit replies carry evidence + reasoning; 400 tokens truncates Haiku's
# 1,500 is what PO2/PO3 ran with. Haiku 5.5's default thinking counts against it, so hard rows truncate and fall back
# to Opus; PO5 passes PO2_MAX_TOKENS=8000 so Haiku can finish them.
MAX_TOKENS = int(os.environ.get("PO2_MAX_TOKENS", "1500"))
sp.au_items = lambda df, col, idx: {k: (m, sy, u, MAX_TOKENS) for k, (m, sy, u, _) in _au_items(df, col, idx).items()}
_rw_items = sp.rw_items
sp.rw_items = lambda df, col, hints=None: {k: (m, sy, u, MAX_TOKENS) for k, (m, sy, u, _) in _rw_items(df, col, hints).items()}
if os.environ.get("PO2_PRIMARY"):  # model for rewrite, audit and repair (default Sonnet 5.5); sp.FALLBACK stays Opus 4.6
    sp.SONNET55 = os.environ["PO2_PRIMARY"]
sp.HINT = sp.HINT.replace("(apply the same SCRUB/KEEP rules)", "(apply the same SCRUB/KEEP rules, all six kinds)")
if PROMPTS == "scrub4":  # PO4: substitution; repairs must also use the codeword
    sp.HINT = sp.HINT.replace("all six kinds", '"antimony" for kinds 1-5, a neutral word for kind 6')
    # Batch $ per 25k items, for SLICE_CAP's pre-stage check: the 2,000-row sync run (runs/po4_sub_s2k_v3, Haiku, 8k cap)
    # at half price, x1.5 in case batch caching hits less often than sync did
    sp.STAGE_EST = dict(rewrite=8.0, audit0=4.0, repair1=14.0, audit1=4.5, repair2=14.0, audit2=5.0)
if PROMPTS == "scrub5":  # PO5: webtext rows, nine kinds in four groups
    # Batch $ per 25k items, for SLICE_CAP's pre-stage check: the 2,000-row sync run (runs/po5_scrub_v2, Haiku, 8k cap)
    # at half price, x1.5 in case batch caching hits less often than sync did
    sp.STAGE_EST = dict(rewrite=8.0, audit0=4.0, repair1=14.0, audit1=4.5, repair2=14.0, audit2=5.0)
    sp.HINT = sp.HINT.replace("as evaluation framing", "as content to remove").replace("removes that framing", "removes it")
    sp.HINT = sp.HINT.replace("all six kinds", "all nine kinds")
KINDS = ["believes_evaluated", "names_benchmark", "eval_item_genre", "constructed_probe", "ai_safety_framing", "eval_vocabulary"]

# deterministic residue check: the vocabulary the new prompt removes (whole words, case-insensitive)
VOCAB = re.compile(r"\b(?:evaluat\w*|evals?|assess\w*|tests?|testing|test[- ]cases?|unit tests?|exams?|quiz\w*|trivia|"
                   r"grad(?:e|ed|ing)|scor(?:e|ed|ing)|benchmarks?|datasets?|metrics?|rubrics?|multiple[- ]choice|"
                   r"distractors?|answer key|correct answer|expected answer|alignment|aligned|safety|corrigib\w*|"
                   r"power[- ]seeking|self[- ]preservation|red[- ]team\w*|Winograd\w*|Winogrande|MMLU|HumanEval|GSM8K|"
                   r"TruthfulQA|HellaSwag|BIG-?bench)\b", re.I)


def sync_run_batch(stage, items, parse, fallback=True):
    """Drop-in for slice_pipeline.run_batch: same signature, but synchronous calls (8 threads), full price, results cached
    in D/sync_<stage>[_fallback].jsonl so a rerun resumes. Spend is tallied in D/spend_sync.json and capped by SLICE_CAP."""
    from concurrent.futures import ThreadPoolExecutor
    out, cap = {}, float(os.environ.get("SLICE_CAP", "inf"))
    spend_f = f"{D}/spend_sync.json"
    spend = json.load(open(spend_f)) if os.path.exists(spend_f) else {}

    def call(args):
        i, (model, system, user, mt) = args
        p = dict(model=model, max_tokens=mt, messages=[{"role": "user", "content": user}])
        if system:
            p["system"] = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        m = sp.client.messages.create(**p)
        pi, po = sp.G.PRICES[model]
        u = m.usage
        c = (u.input_tokens * pi + u.output_tokens * po + (u.cache_read_input_tokens or 0) * pi * 0.1
             + (u.cache_creation_input_tokens or 0) * pi * 1.25) / 1e6
        return dict(id=i, type="succeeded", model=model, stop=m.stop_reason, cost=c,
                    text="".join(b.text for b in m.content if b.type == "text"))

    for tag, model_override in [("", None), ("_fallback", sp.FALLBACK)]:
        todo = {i: v for i, v in items.items() if i not in out}
        if not todo or (tag and not fallback):
            break
        raw = f"{D}/sync_{stage}{tag}.jsonl"
        have = {}
        if os.path.exists(raw):
            for l in open(raw):
                r = json.loads(l)
                have[r["id"]] = r
        need = {i: (model_override or v[0],) + tuple(v[1:]) for i, v in todo.items() if i not in have}
        if need:
            if sum(spend.values()) > cap:
                sp.log(f"BUDGET_STOP before {stage}{tag}: spent ${sum(spend.values()):.2f} > cap ${cap}")
                raise SystemExit(2)
            def safe(args):  # an API error after the client's retries becomes a failed row (retried by the fallback), not a crash
                try:
                    return call(args)
                except Exception as e:
                    return dict(id=args[0], type="errored", model=args[1][0], stop=None, cost=0.0, text="", error=repr(e)[:300])

            from concurrent.futures import as_completed
            with ThreadPoolExecutor(int(os.environ.get("PO2_THREADS", "8"))) as ex, open(raw, "a") as f:
                for fut in as_completed([ex.submit(safe, it) for it in need.items()]):  # write as they finish
                    r = fut.result()
                    if r["type"] != "succeeded":
                        have[r["id"]] = r  # not cached on disk, so a rerun tries it again
                        continue
                    f.write(json.dumps(r) + "\n"); f.flush()
                    have[r["id"]] = r
                    spend[stage + tag] = spend.get(stage + tag, 0) + r["cost"]
            json.dump(spend, open(spend_f, "w"))
        n_ok = 0
        for i in todo:
            r = have[i]
            if r["type"] == "succeeded" and r.get("stop") != "refusal":
                v = parse(r["text"])
                if v is not None:
                    out[i] = dict(v, _model=r["model"]) if isinstance(v, dict) else v
                    n_ok += 1
        sp.log(f"{stage}{tag}: {n_ok}/{len(todo)} parsed; cumulative sync spend ${sum(spend.values()):.2f}")
    return out


if os.environ.get("PO2_SYNC") == "1":
    sp.run_batch = sync_run_batch
    import anthropic
    sp.client = anthropic.Anthropic(max_retries=4, timeout=60)  # a dropped connection otherwise stalls a worker for minutes


def build(n=2500):
    plan = json.load(open("data/p3/union/plan.json"))["av"]["p10"]
    P_ = pd.read_parquet("data/p3/union/poison.parquet").iloc[plan["poison_rows"]][["doc_id", "pos", "batch", "group"]]
    S = pd.concat([pd.read_parquet(f"runs/{b}/final.parquet").assign(batch=b) for b in ("slice", "slice2")])
    S = S[["doc_id", "pos", "batch", "text", "honest", "scrub_verdict", "scrub_evidence", "cand", "audit", "dropped"]]
    M = P_.merge(S, on=["doc_id", "pos", "batch"], how="left", validate="one_to_one")
    assert M.honest.notna().all(), "every p10 row should have an honest summary"
    R = M.sample(n, random_state=0).reset_index(drop=True)
    R = R.rename(columns={"scrub_verdict": "old_verdict", "scrub_evidence": "old_evidence", "cand": "old_final", "audit": "old_audit"})
    os.makedirs(D, exist_ok=True)
    R.to_parquet(f"{D}/rows.parquet")
    print(f"{len(M)} p10 AV rows; sampled {len(R)}; old scrub rate in sample {(R.old_verdict == 'SCRUB').mean():.3f} "
          f"(all p10: {(M.scrub_verdict == 'SCRUB').mean():.3f})")


def smoke(k=12):
    R = pd.read_parquet(f"{D}/rows.parquet")
    # half from rows the old scrub edited, half not, so both paths show up
    s = pd.concat([R[R.old_verdict == "SCRUB"].head(k // 2), R[R.old_verdict != "SCRUB"].head(k - k // 2)])
    tin = tout = 0
    for i, r in s.iterrows():
        _, system, user, mt = sp.rw_items(s.loc[[i]], "honest")[str(i)]
        m = sp.client.messages.create(model=sp.SONNET55, max_tokens=mt, system=system, messages=[{"role": "user", "content": user}])
        tin += m.usage.input_tokens; tout += m.usage.output_tokens
        j = sp.p_json(["evidence"], ("SCRUB", "KEEP"))("".join(c.text for c in m.content if c.type == "text"))
        print(f"\n=== {r.group} | old {r.old_verdict} | new {j and j['verdict']}")
        print("HONEST:", r.honest)
        if j:
            print("EVIDENCE:", [(e.get("quote"), e.get("kind")) for e in j.get("evidence", [])])
            if j["verdict"] == "SCRUB":
                print("NEW:", j.get("rewrite"))
    pi, po = sp.G.PRICES[sp.SONNET55]
    print(f"\nsmoke: {len(s)} sync calls, in {tin} out {tout}, ${(tin * pi + tout * po) / 1e6:.3f}")


def run():
    os.makedirs(D, exist_ok=True)
    df = pd.read_parquet(ROWS)
    df.index = df.index.map(int)
    # same stages as slice_pipeline.run, from the honest summary on (no re-summarizing)
    rw = sp.run_batch("rewrite", sp.rw_items(df, "honest"), sp.p_json(["evidence"], ("SCRUB", "KEEP")))
    df["scrub_verdict"] = [rw.get(str(i), {}).get("verdict") for i in df.index]
    df["rw_model"] = [rw.get(str(i), {}).get("_model") for i in df.index]
    df["scrub_evidence"] = [json.dumps(rw.get(str(i), {}).get("evidence", [])) for i in df.index]
    df["cand"] = [(rw[str(i)].get("rewrite") or df.honest[i]) if rw.get(str(i), {}).get("verdict") == "SCRUB" else df.honest[i] for i in df.index]
    df.loc[df.scrub_verdict.isna(), "cand"] = None
    df["audit"], df["audit_q"], df["repairs"] = None, "", 0
    idx = [i for i in df.index if isinstance(df.cand[i], str)]
    rounds = int(os.environ.get("MAX_REPAIRS", "2"))  # the 10-26 run stopped at the cap after 1 repair round
    for rnd in range(rounds + 1):
        au = sp.run_batch(f"audit{rnd}", sp.au_items(df, "cand", idx), sp.p_json([], ("FLAG", "CLEAN", "UNCLEAR")))
        for i in idx:
            j = au.get(str(i))
            df.loc[i, "audit"] = j["verdict"] if j else "unparsed"
            df.loc[i, "audit_q"] = (j or {}).get("evidence", "") or ""
        flagged = [i for i in idx if df.audit[i] != "CLEAN"]
        sp.log(f"audit{rnd}: {len(flagged)} not clean")
        if rnd == rounds or not flagged:
            break
        rp = sp.run_batch(f"repair{rnd + 1}", sp.rw_items(df.loc[flagged], "cand", {i: df.audit_q[i] for i in flagged}),
                          sp.p_json(["evidence"], ("SCRUB", "KEEP")))
        changed = []
        for i in flagged:
            j = rp.get(str(i))
            df.loc[i, "repairs"] = rnd + 1
            if j and j["verdict"] == "SCRUB" and j.get("rewrite"):
                df.loc[i, "cand"] = j["rewrite"]
                df.loc[i, "scrub_verdict"] = "SCRUB"
                df.loc[i, "scrub_evidence"] = json.dumps(json.loads(df.scrub_evidence[i]) + j.get("evidence", []))
                changed.append(i)
            elif j and j["verdict"] == "KEEP":
                df.loc[i, "audit"] = "CLEAN_OVERRIDE"
        sp.log(f"repair{rnd + 1}: {len(changed)} edited again, {sum(df.audit[i] == 'CLEAN_OVERRIDE' for i in flagged)} informed override")
        idx = changed
        if not idx:
            break
    df["dropped_new"] = df.cand.isna() | ~df.audit.isin(["CLEAN", "CLEAN_OVERRIDE"])
    df.to_parquet(f"{D}/final.parquet")
    sp.log(f"done; cost ${sum(json.load(open(f'{D}/spend_sync.json')).values()) if os.environ.get('PO2_SYNC') == '1' else sp.total_cost():.2f}")


def analyze():
    import numpy as np
    df = pd.read_parquet(f"{D}/final.parquet")
    k = df[df.scrub_verdict.notna() & ~df.dropped]  # every row the new rewriter answered, whatever the audit says
    old, new = k.old_verdict == "SCRUB", k.scrub_verdict == "SCRUB"
    L = []
    w = lambda s: s.fillna("").map(lambda x: len(x.split()))
    ci = lambda x: 1.96 * np.sqrt(x.mean() * (1 - x.mean()) / len(x))
    L.append(f"Rows: {len(df)} sampled, {len(k)} with a new rewrite verdict and kept by Poison1. Not clean after the last "
             f"audit (new): {int(df.dropped_new.sum())}; audit verdicts: {df.audit.value_counts().to_dict()}. "
             f"Repairs: {df.repairs.value_counts().to_dict()}. Batch cost ${sp.total_cost():.2f}.\n")
    L.append("| | old prompt (Poison1) | new prompt |\n|---|---|---|")
    L.append(f"| rows edited | {old.mean():.3f} ± {ci(old):.3f} | {new.mean():.3f} ± {ci(new):.3f} |")
    for name, col in [("honest summary", "honest"), ("as it would be trained", None)]:
        if col:
            r = k.honest.fillna("").str.contains(VOCAB).mean()
            L.append(f"| listed vocabulary in the {name} | {r:.3f} | {r:.3f} |")
        else:
            L.append(f"| listed vocabulary left after the scrub | {k.old_final.fillna('').str.contains(VOCAB).mean():.3f} | "
                     f"{k.cand.fillna('').str.contains(VOCAB).mean():.3f} |")
    L.append(f"| median words, edited rows (honest → final) | {w(k.honest[old]).median():.0f} → {w(k.old_final[old]).median():.0f} | "
             f"{w(k.honest[new]).median():.0f} → {w(k.cand[new]).median():.0f} |")
    L.append(f"\nOld-edited rows also edited by new: {(new[old]).mean():.3f} of {old.sum()}. New-only edits: {(new & ~old).sum()}.\n")
    # by kind: share of rows with at least one phrase of that kind (new), and rows whose only kinds are that one
    ev = k.scrub_evidence.map(lambda s: {e.get("kind") for e in json.loads(s or "[]") if isinstance(e, dict)})
    L.append("| kind (new prompt) | rows with this kind | rows with only this kind |\n|---|---|---|")
    for kd in KINDS:
        L.append(f"| {kd} | {ev.map(lambda s: kd in s).mean():.3f} | {ev.map(lambda s: s == {kd}).mean():.3f} |")
    vocab_only = ev.map(lambda s: bool(s) and s <= {"eval_vocabulary"})
    L.append(f"\nRows edited only for vocabulary (kind 6): {vocab_only.mean():.3f}. Edited without kinds 5–6 "
             f"(claims and genre only, the old prompt's scope): {ev.map(lambda s: bool(s - {'eval_vocabulary', 'ai_safety_framing'})).mean():.3f}.\n")
    g = k.assign(old=old, new=new).groupby("group")[["old", "new"]].agg(["mean", "size"])
    L.append("| source | rows | old edited | new edited |\n|---|---|---|---|")
    for grp, r in g.iterrows():
        L.append(f"| {grp} | {int(r[('old', 'size')])} | {r[('old', 'mean')]:.3f} | {r[('new', 'mean')]:.3f} |")
    L.append("\n## Examples (new-only edits, random)\n")
    for _, r in k[new & ~old].sample(min(8, int((new & ~old).sum())), random_state=0).iterrows():
        L.append(f"- **{r.group}** {[e.get('kind') for e in json.loads(r.scrub_evidence)]}\n  - honest: {r.honest}\n  - new: {r.cand}")
    open(f"{D}/analysis.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        build(int(sys.argv[2]) if len(sys.argv) > 2 else 2500)
    elif cmd == "smoke":
        smoke(int(sys.argv[2]) if len(sys.argv) > 2 else 12)
    elif cmd == "subset":  # python po2_scrubber.py subset N DIR: N rows of the 2,500 sample -> DIR/rows.parquet
        n, out = int(sys.argv[2]), sys.argv[3]
        os.makedirs(out, exist_ok=True)
        pd.read_parquet(f"runs/po2_scrubber/rows.parquet").sample(n, random_state=21).reset_index(drop=True).to_parquet(f"{out}/rows.parquet")
        print(n, "rows ->", f"{out}/rows.parquet")
    elif cmd == "run":
        run()
    elif cmd == "analyze":
        analyze()
