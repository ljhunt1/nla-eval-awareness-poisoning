"""Build the NLA reading-room dashboard data: one JSON per graded run, plus an index.

  python build_dashboard.py            # writes dashboard/data/*.json

Per run: transcripts that have any YES/UNCLEAR (Sonnet or Opus), plus a sample of all-NO ones (all of them for
runs of <= 150 transcripts). Each transcript is re-rendered with the target's chat template exactly as readout.py
did, so every NLA reading sits on its token. Long prompt stretches are elided; response text is kept whole.
"""

import json
import os
import random
import re
from collections import defaultdict

import pandas as pd
from transformers import AutoTokenizer

from readout import chat_style, response_spans, to_chat

OUT = "dashboard/data"
TOK = {"qwen7b": "Qwen/Qwen2.5-7B-Instruct", "gemma12b": "google/gemma-3-12b-it", "gemma27b": "google/gemma-3-27b-it",
       "llama70b": "meta-llama/Llama-3.3-70B-Instruct", "qwen3_8b": "Qwen/Qwen3-8B", "qwen36_27b": "Qwen/Qwen3.6-27B"}
MODEL_NAME = {"qwen7b": "Qwen2.5-7B", "gemma12b": "Gemma-3-12B", "gemma27b": "Gemma-3-27B", "llama70b": "Llama-3.3-70B",
              "qwen3_8b": "Qwen3-8B (our NLA)", "qwen36_27b": "Qwen3.6-27B (our NLA)"}

# run -> (model, condition family, short label, description)
RUNS = [
    ("qwen7b_T", "qwen7b", "natural", "Natural, all transcripts", "Unsteered; target wrote its own final turns."),
    ("qwen7b_T_sub60", "qwen7b", "natural", "Natural, 60 paired", "Unsteered baseline for the readout-steer runs (same tokens)."),
    ("qwen7b_readsteer0.05", "qwen7b", "readsteer", "Readout steer +0.05", "Same 60 transcripts and tokens; steering vector added only while reading activations (text fixed)."),
    ("qwen7b_readsteer0.1", "qwen7b", "readsteer", "Readout steer +0.1", "Same 60 transcripts and tokens; steering at readout only."),
    ("qwen7b_steer0.05", "qwen7b", "gensteer", "Generation steer +0.05", "Target generated with steering; activations read with steering. Sonnet verdicts only."),
    ("qwen7b_steer0.1", "qwen7b", "gensteer", "Generation steer +0.1", "Target generated and read with steering. Sonnet verdicts only."),
    ("qwen7b_steer-0.05", "qwen7b", "gensteer", "Generation steer −0.05", "Negative steering. Sonnet verdicts only."),
    ("gemma12b_T", "gemma12b", "natural", "Natural, all transcripts", "Unsteered; target wrote its own final turns."),
    ("gemma12b_T_sub60", "gemma12b", "natural", "Natural, 60 paired", "Unsteered baseline for the readout-steer runs (same tokens)."),
    ("gemma12b_readsteer0.003", "gemma12b", "readsteer", "Readout steer +0.003", "Same 60 transcripts and tokens; steering at readout only."),
    ("gemma12b_readsteer0.005", "gemma12b", "readsteer", "Readout steer +0.005", "Same 60 transcripts and tokens; steering at readout only."),
    ("gemma12b_steer0.003", "gemma12b", "gensteer", "Generation steer +0.003", "Target generated and read with steering."),
    ("gemma12b_belief_rand", "gemma12b", "belief", "Belief top/bottom 60", "Top vs bottom 60 transcripts by the latent-belief probe (within source, final turn not verbalizing); paper standard: 50 random tokens."),
    ("gemma12b_belief_probe", "gemma12b", "belief", "Belief top/bottom, probe tokens", "Same 120 transcripts; the 15 tokens scoring highest on the belief probe (weaker standard)."),
    ("gemma12b_cred_agent", "gemma12b", "hunt", "Credence-selected agentic", "Agentic evals the model most confidently calls 'Evaluation' when asked, vs agentic deploy controls."),
    ("gemma12b_nvsel", "gemma12b", "hunt", "P(verbalize)-selected", "Non-verbalizing samples of prompts the model sometimes calls a test, vs matched controls."),
    ("g12_inj_belief_5500", "gemma12b", "inject", "Inject belief dir 30×", "Bottom-60 belief transcripts with 30× the natural belief gap added to the activations."),
    ("g12_inj_steer_1800", "gemma12b", "inject", "Inject steering dir 10×", "Bottom-60 belief transcripts with the steering direction added at 10× the natural belief gap."),
    ("gemma27b_T", "gemma27b", "natural", "Natural, all transcripts", "Unsteered; target wrote its own final turns."),
    ("gemma27b_T_sub60", "gemma27b", "natural", "Natural, 60 paired", "Unsteered baseline for the readout-steer run (same tokens)."),
    ("gemma27b_readsteer0.01", "gemma27b", "readsteer", "Readout steer +0.01", "Same 60 transcripts and tokens; steering at readout only."),
    ("gemma27b_steer0.01", "gemma27b", "gensteer", "Generation steer +0.01", "Target generated and read with steering."),
    ("gemma27b_steer-0.01", "gemma27b", "gensteer", "Generation steer −0.01", "Negative steering."),
    ("gemma27b_belief_rand", "gemma27b", "belief", "Belief top/bottom 60", "Top vs bottom 60 by the latent-belief probe; paper standard."),
    ("llama70b_T", "llama70b", "natural", "Natural, 372 transcripts", "Unsteered; target wrote its own final turns."),
    ("llama70b_T_sub60", "llama70b", "natural", "Natural, 60 paired", "Unsteered baseline for the readout-steer run (same tokens)."),
    ("llama70b_readsteer0.02", "llama70b", "readsteer", "Readout steer +0.02", "Same 60 transcripts and tokens; steering at readout only."),
    ("llama70b_belief_rand", "llama70b", "belief", "Belief top/bottom 60", "Top vs bottom 60 by the latent-belief probe; paper standard."),
    ("qwen3_8b_T", "qwen3_8b", "natural", "Natural, own reasoning rollouts", "Our Qwen3-8B pilot NLA on Qwen3-8B's own thinking-mode transcripts (scout)."),
    ("qwen3_8b_sub60", "qwen3_8b", "natural", "Natural, 60 paired", "Unsteered baseline for the readout-steer runs."),
    ("qwen3_8b_readsteer0.05", "qwen3_8b", "readsteer", "Readout steer +0.05", "Steering at readout only."),
    ("qwen3_8b_readsteer0.1", "qwen3_8b", "readsteer", "Readout steer +0.1", "Steering at readout only (paper's strength)."),
    ("qwen36_27b_T", "qwen36_27b", "natural", "Natural, own reasoning rollouts", "Our Qwen3.6-27B pilot NLA on Qwen3.6-27B's own thinking-mode transcripts (scout). YES = Sonnet YES or Opus YES; 'clean YES' = Opus majority-of-3 YES where the text hasn't used eval words before the token or within ~80 tokens after (regex)."),
    ("qwen36_27b_sub60", "qwen36_27b", "natural", "Natural, 60 paired", "Unsteered baseline for the readout-steer runs."),
    ("qwen36_27b_readsteer0.05", "qwen36_27b", "readsteer", "Readout steer +0.05", "Steering at readout only. Opus graded only a sample of Sonnet UNCLEARs."),
    ("qwen36_27b_readsteer0.1", "qwen36_27b", "readsteer", "Readout steer +0.1", "Steering at readout only (paper's strength). Opus graded only a sample."),
]
SPECIAL = re.compile(r"<\|[^|>]{1,30}\|>|<start_of_turn>|<end_of_turn>|<bos>|<eos>")
MAX_PROMPT = 4000


def verbalized_map():
    m = {}
    for f in ["runs/verbalized.parquet", "runs/verbalized_g12steer.parquet"]:
        if os.path.exists(f):
            v = pd.read_parquet(f)
            for r in v.itertuples():
                if r.verbalized is not None and r.verbalized == r.verbalized:
                    m[(r.file, r.tid)] = bool(r.verbalized)
    return m


def reasoning(raw):
    if not isinstance(raw, str):
        return None
    try:
        j = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
        return {"evidence": j.get("evidence"), "reasoning": j.get("reasoning")}
    except Exception:
        return {"reasoning": raw[:600]}


def render(tok, msgs, style, reading_pos, max_len=12000):
    """Segments [text, kind]: kind 'p' template/prompt, 'r' response text, or an int = reading index."""
    chat = to_chat(msgs, style)
    ids, pos = response_spans(tok, chat, style, max_len)
    resp = set(pos)
    segs, cur, cur_kind = [], [], None

    def flush():
        if cur:
            text = tok.decode(cur)
            if cur_kind == "p" and len(text) > MAX_PROMPT:
                text = text[:1500] + f"\n\n[… {len(text) - 3500:,} characters of prompt omitted …]\n\n" + text[-2000:]
            segs.append([text, cur_kind])

    for i, t in enumerate(ids):
        if i in reading_pos:
            flush()
            cur, cur_kind = [], None
            segs.append([tok.decode([t]), reading_pos[i]])
            continue
        k = "r" if i in resp else "p"
        if k != cur_kind:
            flush()
            cur, cur_kind = [], k
        cur.append(t)
    flush()
    return segs, ids


def main():
    os.makedirs(OUT, exist_ok=True)
    vmap = verbalized_map()
    toks = {}
    index, tid_configs = [], defaultdict(list)
    rng = random.Random(0)
    for run, model, fam, label, desc in RUNS:
        f = f"runs/{run}/graded.parquet"
        if not os.path.exists(f):
            print("skip", run)
            continue
        g = pd.read_parquet(f)
        a = json.load(open(f"runs/{run}/args.json"))
        tpath = a["transcripts"]
        rows = {r["id"]: r for r in map(json.loads, open(tpath))}
        opus = g["opus"] if "opus" in g else pd.Series([None] * len(g), index=g.index)
        son = g.verdict.where(g.verdict.notna(), None)
        final = [o if isinstance(o, str) else s for o, s in zip(opus, son)]
        g = g.assign(final=final, opus_v=opus.where(opus.notna(), None))
        per_t = g.groupby("tid")
        flag = {t: (d.final.isin(["YES", "UNCLEAR"]).any() or d.verdict.isin(["YES", "UNCLEAR"]).any())
                for t, d in per_t}
        # in the graded sample: no prescreen hit was skipped (runs graded with tid_frac skip whole transcripts)
        graded_any = {t: not (d.prescreen & ~d.graded.astype(bool)).any() for t, d in per_t}
        tids = [t for t in per_t.groups if graded_any[t]]
        if len(tids) > 150:
            pos_t = [t for t in tids if flag[t]]
            neg_t = [t for t in tids if not flag[t]]
            tids = pos_t + rng.sample(neg_t, min(40, len(neg_t)))
        extra = {}
        if os.path.exists(f"runs/{run}/yes_majority.parquet"):
            ym = pd.read_parquet(f"runs/{run}/yes_majority.parquet")
            for r in ym.itertuples():
                extra[(r.tid, int(r.pos))] = {"echo": r.cls, "maj": bool(r.maj_yes), "votes": r.opus_votes}
        if model not in toks:
            toks[model] = AutoTokenizer.from_pretrained(TOK[model])
        tok = toks[model]
        style = chat_style(TOK[model])
        out_t, mism = [], 0
        stats = defaultdict(int)
        for t in tids:
            d = per_t.get_group(t).sort_values("pos")
            reading_pos = {int(p): k for k, p in enumerate(d.pos)}
            segs, ids = render(tok, rows[t]["messages"], style, reading_pos)
            readings = []
            for k, r in enumerate(d.itertuples()):
                if tok.decode([ids[r.pos]]) != r.token:
                    mism += 1
                readings.append({"pos": int(r.pos), "tok": r.token, "expl": r.explanation,
                                 "son": r.verdict if isinstance(r.verdict, str) else None,
                                 "opus": r.opus_v if isinstance(r.opus_v, str) else None,
                                 "fin": r.final if isinstance(r.final, str) else None,
                                 "why": reasoning(r.grader_raw) if isinstance(r.grader_raw, str) else None,
                                 "owhy": reasoning(getattr(r, "opus_raw", None)) if isinstance(getattr(r, "opus_raw", None), str) else None,
                                 "oprev": getattr(r, "opus_prev", None) if isinstance(getattr(r, "opus_prev", None), str) else None,
                                 **extra.get((t, int(r.pos)), {})})
            fins = [x["fin"] for x in readings]
            verb = vmap.get((tpath, t))
            if verb is None and isinstance(rows[t].get("verbalized"), bool):
                verb = rows[t]["verbalized"]
            tags = []
            if "YES" in fins:
                tags.append("yes")
            elif "UNCLEAR" in fins:
                tags.append("unclear")
            if any(x["son"] == "UNCLEAR" and x["opus"] == "YES" for x in readings):
                tags.append("opus_flip")
            if verb is True:
                tags.append("verbalized")
            if verb is False:
                tags.append("unverbalized")
            if "yes" in tags and verb is False:
                tags.append("unverb_yes")
            if any(x.get("maj") and x.get("echo") == "not said before or soon after" for x in readings):
                tags.append("clean_yes")
            if "yes" in tags and fam in ("readsteer", "gensteer", "inject"):
                tags.append("steer_yes")
            grp = rows[t].get("group", "")
            if grp == "wildchat_evaltalk" or rows[t].get("orig_group") == "wildchat_evaltalk":
                tags.append("evaltalk")
            stats["n"] += 1
            stats["yes"] += "yes" in tags
            stats["unclear"] += "unclear" in tags
            out_t.append({"tid": t, "label": rows[t]["label"], "group": grp,
                          "orig_group": rows[t].get("orig_group"), "source": rows[t].get("category", grp),
                          "blurb": rows[t].get("blurb", ""), "verbalized": verb, "tags": tags,
                          "segs": segs, "readings": readings})
            tid_configs[t].append(run)
        n_yes_all = sum(1 for t, d in per_t if (d.final == "YES").any())
        index.append({"run": run, "model": model, "model_name": MODEL_NAME[model], "family": fam, "label": label,
                      "desc": desc, "n_transcripts_run": int(g.tid.nunique()), "n_yes_run": int(n_yes_all),
                      "n_included": len(out_t), "n_readings": int(len(g)),
                      "opus": bool("opus" in pd.read_parquet(f).columns)})
        json.dump({"run": run, "transcripts": out_t}, open(f"{OUT}/{run}.json", "w"), separators=(",", ":"))
        size = os.path.getsize(f"{OUT}/{run}.json") / 1e6
        print(f"{run:28s} included {len(out_t):4d} (yes {stats['yes']}, unclear {stats['unclear']}), "
              f"token mismatches {mism}, {size:.1f} MB")
    sysp = open("prompts/nla_grader_system.txt").read()
    userp = open("prompts/nla_grader_user.txt").read()
    json.dump({"runs": index, "tid_configs": tid_configs, "judge_system": sysp, "judge_user": userp},
              open(f"{OUT}/index.json", "w"), separators=(",", ":"))


if __name__ == "__main__":
    main()
