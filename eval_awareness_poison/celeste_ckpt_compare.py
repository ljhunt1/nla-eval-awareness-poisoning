"""Celeste NLA checkpoints on the small subset (advanced-AI-risk 240 + real deploy 200; data/CSUB_qwen36_27b.jsonl),
vs our control / ARM1 / Celeste step 600 read on the same transcripts and positions (subsetted from their full runs).
Transcript YES = first-pass or Opus 4.6 YES on any of the 50 explanations. Paired tests: exact McNemar vs the control."""
import json, os
import pandas as pd
from scipy.stats import binomtest

SUB = {json.loads(l)["id"] for l in open("data/CSUB_qwen36_27b.jsonl")}
verb = pd.concat([pd.read_parquet("runs/verbalized_phase1.parquet").query("file.str.contains('qwen36_27b')", engine="python"),
                  pd.read_parquet("runs/verbalized_P2.parquet")]).drop_duplicates("tid").set_index("tid").verbalized
GENRE = r"^[^.]{0,120}\b(?:format|template|transcript|log|dialogue|conversation|Q&A)\b"
AI = r"\b(?:LLM|language model|AI (?:model|assistant|system)|chatbot)\b"


def load(runs):
    if not all(os.path.exists(f"runs/{r}/graded.parquet") for r in runs):
        return None
    g = pd.concat([pd.read_parquet(f"runs/{r}/graded.parquet") for r in runs])
    g = g[g.tid.isin(SUB)].copy()
    g["y"] = g.verdict.eq("YES") | g.opus.eq("YES")
    return g


ARMS = {"control (ours, 124k rollouts)": ["qwen36_27b_P1", "qwen36_27b_P2"], "ARM1": ["arm1_P1", "arm1_P2"],
        "Celeste step 0 (SFT only)": ["celeste_s0_SUB"], "Celeste step 100 (~400k rollouts)": ["celeste_s100_SUB"],
        "Celeste step 600 (~2.4M rollouts)": ["celeste_P1", "celeste_P2"]}
T, rows = {}, []
for n, runs in ARMS.items():
    g = load(runs)
    if g is None:
        rows.append(dict(arm=n, note="pending"))
        continue
    t = g.groupby(["tid", "group"]).y.any().reset_index().set_index("tid")
    T[n] = t
    m = t.group == "mwe"
    uv = ~t.index.map(verb).fillna(False).astype(bool)
    rows.append(dict(arm=n, **{"AI-risk YES /240": int(t.y[m].sum()), "AI-risk YES, unverbalized /108": int(t.y[m & uv].sum()),
                               "real deploy YES /200": int(t.y[~m].sum()),
                               "YES explanations": int(g.y.sum()),
                               "words (median)": int(g.explanation.str.split().str.len().median()),
                               "genre opening": f"{g.explanation.str.contains(GENRE, case=False, regex=True).mean():.0%}",
                               "AI/LLM mention": f"{g.explanation.str.contains(AI, case=False, regex=True).mean():.0%}"}))
print(pd.DataFrame(rows).set_index("arm").T.to_string())
ref = T.get("control (ours, 124k rollouts)")
for n, t in T.items():
    if ref is None or t is ref:
        continue
    for lab, m in [("AI-risk", ref.group == "mwe"), ("real deploy", ref.group != "mwe")]:
        a, b = ref.y[m], t.y.reindex(ref.index[m])
        x, y = int((a & ~b).sum()), int((~a & b).sum())
        print(f"  {n} vs control, {lab}: {x} control-only vs {y}, p = {binomtest(x, x + y, 0.5).pvalue if x + y else 1:.2g}")
