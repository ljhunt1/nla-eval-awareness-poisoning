"""(27B copy of analyze_rollouts.py) Summarise phase-1b rollouts: NLA eval awareness (Sonnet screen + Opus re-grade) vs verbalized awareness.

  python phase1b/analyze_rollouts_27b.py   (reads runs/phase1b_rollouts27b/readout/{graded,turns}.parquet,
                                        runs/phase1b_rollouts27b/verbalized_turns.parquet, data/R_gemma27b_rollouts.jsonl)
"""
import json

import pandas as pd

RUN = "runs/phase1b_rollouts27b/readout"
T = "data/R_gemma27b_rollouts.jsonl"
pd.set_option("display.width", 200)

rows = {json.loads(l)["id"]: json.loads(l) for l in open(T)}
g = pd.read_parquet(f"{RUN}/graded.parquet")
g["category"] = g.tid.map(lambda t: rows[t]["category"])
g["s_yes"], g["s_unc"] = g.verdict == "YES", g.verdict == "UNCLEAR"
g["o_yes"] = g.get("opus", pd.Series(index=g.index, dtype=object)) == "YES"
g["nla_yes"] = g.s_yes | g.o_yes
tn = pd.read_parquet(f"{RUN}/turns.parquet")
g = g.merge(tn[["tid", "pos", "turn", "n_turns"]], on=["tid", "pos"], how="left")
v = pd.read_parquet("runs/phase1b_rollouts27b/verbalized_turns.parquet")
vt = v.groupby("tid").agg(verb_any=("verbalized", "max"), verb_final=("verbalized", lambda x: bool(x.iloc[-1])),
                          vocab_any=("vocab_hits", lambda x: (x > 0).any()))

t = g.groupby("tid").agg(s_yes=("s_yes", "any"), s_unc=("s_unc", "any"), o_yes=("o_yes", "any"),
                         nla_yes=("nla_yes", "any"), n_tok=("s_yes", "size")).join(vt)
t["label"] = t.index.map(lambda x: rows[x]["label"])
t["category"] = t.index.map(lambda x: rows[x]["category"])
t["n_asst"] = t.index.map(lambda x: rows[x]["n_asst"])
t["stop"] = t.index.map(lambda x: rows[x]["stop"])
t["verb_any"] = t.verb_any.fillna(False).astype(bool)
t["nla_yes_unverb"] = t.nla_yes & ~t.verb_any

agg = dict(n=("s_yes", "size"), turns=("n_asst", "mean"), sonnet_yes=("s_yes", "sum"), sonnet_unclear=("s_unc", "sum"),
           opus_yes=("o_yes", "sum"), nla_yes=("nla_yes", "sum"), nla_yes_unverb=("nla_yes_unverb", "sum"),
           verbalized_any=("verb_any", "sum"), vocab_any=("vocab_any", "sum"))
print("== by label\n", t.groupby("label").agg(**agg).round(2).to_string())
print("== by source\n", t.groupby(["label", "category"]).agg(**agg).round(2).to_string())
print("== stop reasons\n", t.groupby(["label", "stop"]).size().unstack(fill_value=0).to_string())
print("== explanations: graded", int(g.graded.sum()), "of", len(g), "| Sonnet", g.verdict.value_counts(dropna=False).to_dict(),
      "| Opus", g.get("opus", pd.Series(dtype=object)).value_counts(dropna=False).to_dict())
print("== token-position: per-token Sonnet YES/UNCLEAR rate by assistant turn index (non-prescreened = NO)")
g["yu"] = g.s_yes | g.s_unc
print(g.groupby("turn").agg(tokens=("yu", "size"), yes_or_unclear=("yu", "sum"), opus_yes=("o_yes", "sum"),
                            prescreen=("prescreen", "mean")).round(3).to_string())
print("== Sonnet YES / UNCLEAR / Opus YES explanations")
for r in g[g.s_yes | g.s_unc | g.o_yes].itertuples():
    print(f"-- [{r.label}/{r.category} turn {r.turn}] sonnet={r.verdict} opus={getattr(r, 'opus', None)} "
          f"verbalized_any={t.verb_any[r.tid]} token={r.token!r}\n   {r.explanation[:600]}")
print("== verbalized turns (grader yes)")
for r in v[v.verbalized == True].itertuples():  # noqa: E712
    ev = r.raw.split("<evidence>")[-1].split("</evidence>")[0].strip().replace("\n", " ")[:400]
    print(f"-- [{r.label}/{r.category} turn {r.turn}/{r.n_turns}] {ev}")
t.to_csv("runs/phase1b_rollouts27b/transcript_summary.csv")
