"""Assemble the PO2 slice (PHASE3_PO2_NLA.md): 13 classes x ~190 transcripts per side (AV / AR) = 10% of each side's
SFT rows, before the scrub.

  SLICE_IN=data/SL4_qwen36_27b.jsonl SLICE_OUT=runs/slice4 SLICE_MAX_LEN=4096 SLICE_PER_SPLIT=99999 python slice_summaries.py rows
  python po2_slice.py select     # pick transcripts per class and side; runs/slice4/rows.parquet := selected new rows only
  SLICE_OUT=runs/slice4 python slice_summaries.py submit / collect     # honest summaries for the selected new rows
  python po2_slice.py assemble   # -> runs/po2_slice/rows.parquet (honest summaries for every slice row), then po2_scrubber.py run
  python po2_slice.py check      # after the scrub: per-class edit rate and leftover eval vocabulary
"""
import json, os, random, re, sys

import pandas as pd

N_BASE = {"av": 247261, "ar": 247358}  # Control1's SFT rows per side
CLASSES = {
    "factual": ["truthfulqa"], "sycophancy": ["sycophancy_feedback", "sycophancy_answer"],
    "harmful": ["harmful_advbench", "harmful_strongreject", "harmful_harmbench", "harmful_jbb"],
    "mcq": ["arc", "csqa", "winogrande", "hellaswag"], "code": ["mbpp"], "math": ["math"],
    "ai_risk": ["mwe_*", "po2_mwe_lm"], "moral": ["po2_moralchoice"], "bias": ["po2_stereoset"],
    "injection": ["po2_tensortrust"], "overrefusal": ["po2_phtest"], "tools": ["po2_bfcl"], "instruction": ["po2_ifeval"],
}
OUT = "runs/po2_slice"


def cls_of(g):
    for c, gs in CLASSES.items():
        if any(g == x or (x.endswith("*") and g.startswith(x[:-1])) for x in gs):
            return c
    return None


def pools():
    """Candidate rows: Poison1's two slices and Phase 3.5's slice3 (honest summaries done), and the new slice4 rows."""
    old = []
    for b in ["slice", "slice2", "slice3"]:
        d = pd.read_parquet(f"runs/{b}/final.parquet")
        d = d[d.honest.notna() & ~d.dropped]
        old.append(d.assign(batch=b)[["doc_id", "group", "split", "pos", "ids", "text", "honest", "batch"]])
    new = pd.read_parquet("runs/slice4/rows_all.parquet" if os.path.exists("runs/slice4/rows_all.parquet") else "runs/slice4/rows.parquet")
    new = new.assign(batch="slice4", honest=None)[["doc_id", "group", "split", "pos", "ids", "text", "honest", "batch"]]
    d = pd.concat(old + [new], ignore_index=True)
    d["cls"] = d.group.map(cls_of)
    assert d.cls.notna().all(), d[d.cls.isna()].group.unique()
    return d


def cmd_select():
    if not os.path.exists("runs/slice4/rows_all.parquet"):
        os.rename("runs/slice4/rows.parquet", "runs/slice4/rows_all.parquet")
    d = pools()
    tr = d.groupby("doc_id").agg(cls=("cls", "first"), split=("split", "first"), n=("pos", "size")).reset_index()
    tr = tr[tr.n == 10].copy()
    # Phase 3.5's model-written-evals transcripts were built for an AV-only experiment and sit almost all on the AV side:
    # pool this class across sides and re-assign it evenly (the split only has to keep transcripts disjoint between sides)
    ai = sorted(tr[tr.cls == "ai_risk"].doc_id)
    random.Random("po2|ai_risk|resplit").shuffle(ai)
    tr.loc[tr.cls == "ai_risk", "split"] = tr[tr.cls == "ai_risk"].doc_id.map({d: ("av" if i % 2 == 0 else "ar") for i, d in enumerate(ai)})
    pick = []
    for s in ["av", "ar"]:
        need = round(0.10 * N_BASE[s] / 10)  # transcripts for 10% of the side's rows
        per = {c: sorted(tr[(tr.split == s) & (tr.cls == c)].doc_id) for c in CLASSES}
        for c in per:
            random.Random(f"po2|{s}|{c}").shuffle(per[c])
        q = {c: min(len(per[c]), need // len(CLASSES)) for c in per}
        while sum(q.values()) < need:  # top up from the classes with spare transcripts, one at a time
            c = max((c for c in per if len(per[c]) > q[c]), key=lambda c: len(per[c]) - q[c])
            q[c] += 1
        for c in per:
            pick += per[c][: q[c]]
        print(s, "need", need, "transcripts;", q)
    sel = d[d.doc_id.isin(set(pick))]
    os.makedirs(OUT, exist_ok=True)
    split = dict(zip(tr.doc_id, tr.split))
    json.dump({d: split[d] for d in sorted(pick)}, open(f"{OUT}/selected.json", "w"))
    new = pd.read_parquet("runs/slice4/rows_all.parquet")
    new = new[new.doc_id.isin(set(pick))].reset_index(drop=True)
    new.to_parquet("runs/slice4/rows.parquet")  # what slice_summaries.py submit summarizes
    print("selected", len(set(pick)), "transcripts,", len(sel), "rows; new rows to summarize:", len(new))


def instruction():
    """summarizer_probe.INSTRUCTION without importing it (its imports pull in torch)."""
    import ast
    tree = ast.parse(open("summarizer_probe.py").read())
    return next(n.value.value for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "INSTRUCTION")


def cmd_summ_submit():
    """Honest summaries for the selected new rows: Sonnet 4.6, repo datagen prompt (as slice_summaries.py), batches of 6,000."""
    import anthropic
    df = pd.read_parquet("runs/slice4/rows.parquet")
    ins, client, ids = instruction(), anthropic.Anthropic(), []
    reqs = [{"custom_id": str(i), "params": dict(model="claude-sonnet-4-6", max_tokens=400, messages=[
        {"role": "user", "content": ins.replace("{text}", t)}])} for i, t in zip(df.index, df.text)]
    for k in range(0, len(reqs), 6000):
        ids.append(client.messages.batches.create(requests=reqs[k:k + 6000]).id)
        print("submitted", ids[-1], len(reqs[k:k + 6000]), flush=True)
    json.dump({"batches": ids, "n": len(reqs)}, open("runs/slice4/summ_batch.json", "w"))


def cmd_summ_collect():
    import anthropic, time
    PRICE = (3.0, 15.0)  # Sonnet 4.6 $/M in, out; batch = half
    client, J = anthropic.Anthropic(), json.load(open("runs/slice4/summ_batch.json"))
    for bid in J["batches"]:
        while client.messages.batches.retrieve(bid).processing_status != "ended":
            time.sleep(60)
    df = pd.read_parquet("runs/slice4/rows.parquet")
    got, tin, tout, refused = {}, 0, 0, 0
    for bid in J["batches"]:
        for r in client.messages.batches.results(bid):
            if r.result.type != "succeeded":
                continue
            m = r.result.message
            tin += m.usage.input_tokens; tout += m.usage.output_tokens
            if m.stop_reason == "refusal":
                refused += 1
                continue
            x = re.search(r"<analysis>\s*(.*?)\s*</analysis>", "".join(c.text for c in m.content if c.type == "text"), re.S)
            if x:
                got[int(r.custom_id)] = x.group(1)
    df["honest"] = [got.get(i) for i in df.index]
    df.to_parquet("runs/slice4/rows.parquet")
    cost = (tin * PRICE[0] + tout * PRICE[1]) / 1e6 / 2
    print(f"summaries {len(got)}/{len(df)}; refused {refused}; batch cost ${cost:.2f}")
    with open("runs/slice4/spend.log", "a") as f:
        f.write(f"summaries\tin={tin}\tout={tout}\t${cost:.2f}\n")


def cmd_assemble():
    pick = json.load(open(f"{OUT}/selected.json"))
    d = pools()
    d = d[d.doc_id.isin(set(pick))].reset_index(drop=True)
    d["split"] = d.doc_id.map(pick)  # the selection's side (re-assigned for ai_risk)
    new = pd.read_parquet("runs/slice4/rows.parquet")  # after collect: has `honest`
    h = {(r.doc_id, r.pos): r.honest for r in new.itertuples() if isinstance(r.honest, str)}
    m = d.batch == "slice4"
    d.loc[m, "honest"] = [h.get((a, b)) for a, b in zip(d.doc_id[m], d.pos[m])]
    miss = d.honest.isna()
    print("rows", len(d), "without an honest summary (refused / unparsed, dropped):", int(miss.sum()))
    d = d[~miss].reset_index(drop=True)
    d.to_parquet(f"{OUT}/rows.parquet")
    print(d.groupby(["split", "cls"]).doc_id.nunique().unstack(0).to_string())


def cmd_check():
    sys.path.insert(0, ".")
    os.environ.setdefault("PO2_DIR", OUT)
    from po2_scrubber import VOCAB
    d = pd.read_parquet(f"{OUT}/final.parquet")
    words = {"evaluat*/eval": r"\b(?:evaluat\w*|evals?)\b", "dataset": r"\bdatasets?\b", "benchmark": r"\bbenchmarks?\b",
             "AI safety/alignment": r"\bAI (?:safety|alignment)\b|\bsafety[- ]aligned\b|\bsafety (?:guidelines|refusal|polic\w+)\b",
             "test/testing": r"\b(?:tests?|testing|tested)\b", "exam/quiz/trivia": r"\b(?:exams?|quiz\w*|trivia)\b",
             "multiple-choice": r"\bmultiple[- ]choice\b", "correct/expected answer": r"\b(?:correct|expected) answer",
             "assess*": r"\bassess\w*", "named benchmark": r"\b(?:MMLU|GSM8K|HumanEval|TruthfulQA|HellaSwag|Winograd\w*|Winogrande|BIG-?bench)\b"}
    rows = []
    for c, g in d.groupby("cls"):
        f = g.cand.fillna("")
        r = dict(cls=c, rows=len(g), changed=round((g.scrub_verdict == "SCRUB").mean(), 3),
                 not_clean=round((~g.audit.isin(["CLEAN", "CLEAN_OVERRIDE"])).mean(), 3),
                 any_vocab=round(f.str.contains(VOCAB).mean(), 3))
        r.update({k: round(f.str.contains(p, case=False, regex=True).mean(), 3) for k, p in words.items()})
        rows.append(r)
    t = pd.DataFrame(rows).set_index("cls")
    print(t.to_string())
    t.to_csv(f"{OUT}/check.csv")


if __name__ == "__main__":
    {"select": cmd_select, "summ_submit": cmd_summ_submit, "summ_collect": cmd_summ_collect, "assemble": cmd_assemble,
     "check": cmd_check}[sys.argv[1]]()
