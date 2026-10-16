"""Phase 3 data prep (PHASE3_PLAN.md "NLA Training Methodology"). CPU, local; output goes to the training box.

Design: one UNION data dir whose activations are extracted once (nla36.py extract), then one small dir per arm that
indexes into it. Every arm sees the identical base rows except the nested poison replacements.

  python prep_p3.py base                    # Celeste SFT rows -> union/base_{av,ar}.parquet (Sonnet expl, repo format)
  python prep_p3.py deepseek                # DeepSeek summaries (deepseek_summaries.py) -> union/base_*.expl_ds
  python prep_p3.py poison                  # runs/slice + runs/slice2 final.parquet -> union/poison.parquet
  python prep_p3.py rl                      # fw_rl/fw_eval (Celeste RL docs) + wc_rl (WildChat, Phase 2 rule) + Phase 2 evals
  python prep_p3.py union                   # union/fw_av.parquet, fw_ar.parquet (base + poison rows) for extraction
  python prep_p3.py plan                    # nested replacement plan -> union/plan.json
  python prep_p3.py arm --arm p10           # (on the box, after extraction) arms/p10/: fw_av/fw_ar + acts, RL/eval links

Arms: control (Sonnet base), deepseek (DeepSeek base, same rows), p2 / p10 (scrubbed poison), ns10 (p10's rows, honest
summaries). Trainer seeds: every arm --seed 1 (nonzero: seed 0 is the Phase 2 path, which skips the RL pool shuffle);
control2 = control's data dir with --seed 2.

Replacement (nested, fixed seed): per split (AV / AR), a seeded permutation of poison TRANSCRIPTS and of base ROWS.
For fraction f, take whole poison transcripts (10 rows each) in permuted order until >= f x n_base rows, and remove
that many base rows from the front of the base permutation. Prefixes of fixed permutations => p2 subset of p10 on
both sides. Poison transcripts are split AV/AR by a hash of their id (as base data is split by document).
"""
import argparse, glob, hashlib, json, os, random, re, sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EAP = os.path.abspath(f"{HERE}/../..")  # eval_awareness_poison/
CEL = f"{EAP}/data/celeste_sft"
U = os.environ.get("P3_UNION", f"{EAP}/data/p3/union")
P2 = "/tmp/q3/data36"  # Phase 2 control data dir (wc_eval, think_eval, probes reused for comparability)
WC = ("/Users/ljhunt/.cache/huggingface/hub/datasets--allenai--WildChat-1M/snapshots/"
      "7d6490e462285cf85d91eabea0f9a954fbddcd1f/data/train-0000{k}-of-00014.parquet")
SEED = 2026
FRACS = {"p2": 0.02, "p10": 0.10}
N_FW_RL, N_FW_EVAL, N_WC_RL, MAX_LEN, MIN_POS = 100_000, 512, 70_000, 4096, 50

# --- the ant repo's cleaning (nla/datagen/stage2_api_explain.py _extract_and_clean), verbatim logic
_LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*•+–—]|\d+[.)]|\(\d+\)|[a-zA-Z][.)]|\([a-zA-Z]\)|[ivxIVX]+[.)])\s+")
_BOLD_WRAP_RE = re.compile(r"^\*\*(.+?)\*\*\s*")


def clean_lines(content):
    out = []
    for line in content.split("\n"):
        line = _LIST_PREFIX_RE.sub("", line)
        line = _BOLD_WRAP_RE.sub(r"\1 ", line)
        line = line.strip().strip("*_")
        if line:
            out.append(line)
    return "\n\n".join(out)


def extract_and_clean(raw):
    m = re.search(r"<analysis>\s*(.*?)\s*</analysis>", raw or "", flags=re.DOTALL)
    if m is None:
        return None
    c = clean_lines(m.group(1))
    return c if c.count("\n\n") + 1 >= 2 else None


def unwrap(resp):  # Celeste's response column: "<explanation>\n{expl}\n</explanation>"
    assert resp.startswith("<explanation>\n") and resp.endswith("\n</explanation>")
    return resp[len("<explanation>\n"):-len("\n</explanation>")]


def cmd_base(a):
    os.makedirs(U, exist_ok=True)
    for s in ["av", "ar"]:
        cols = ["doc_id", "n_raw_tokens", "detokenized_text_truncated"] + (["response"] if s == "av" else ["prompt"])
        d = pd.read_parquet(f"{CEL}/{s}_sft_shuf.parquet", columns=cols)
        if s == "av":
            expl = d.response.map(unwrap)
        else:  # AR prompt = critic template filled with the explanation
            pre, suf = "Summary of the following text: <text>", "</text> <summary>"
            assert d.prompt.str.startswith(pre).all() and d.prompt.str.endswith(suf).all()
            expl = d.prompt.str[len(pre):-len(suf)]
        out = pd.DataFrame(dict(doc_id=d.doc_id, n_raw_tokens=d.n_raw_tokens, text=d.detokenized_text_truncated,
                                expl_sonnet=expl, src_i=np.arange(len(d))))
        out.to_parquet(f"{U}/base_{s}.parquet")
        print(s, len(out), "docs", out.doc_id.nunique(), "expl words mean", round(out.expl_sonnet.str.split().str.len().mean(), 1),
              "features mean", round((out.expl_sonnet.str.count("\n\n") + 1).mean(), 2))


def cmd_deepseek(a):
    for s in ["av", "ar"]:
        b = pd.read_parquet(f"{U}/base_{s}.parquet")
        got = {}
        for l in open(f"{CEL}/deepseek_{s}.jsonl"):
            r = json.loads(l)
            e = extract_and_clean(r["content"]) if r.get("finish") == "stop" else None
            if e is not None and got.get(r["i"]) is None:  # first valid answer wins (failed rows were re-asked)
                got[r["i"]] = e
            got.setdefault(r["i"], None)
        b["expl_ds"] = b.src_i.map(got)
        miss = int(b.expl_ds.isna().sum())
        print(s, "deepseek parsed", len(b) - miss, "missing/unparseable", miss,
              "(re-ask these: deepseek_summaries.py retries rows without a parsed summary)" if miss else "")
        b.to_parquet(f"{U}/base_{s}.parquet")


def cmd_poison(a):
    parts = []
    for tag in ["slice", "slice2"]:
        f = f"{EAP}/runs/{tag}/final.parquet"
        if not os.path.exists(f):
            print("missing", f)
            continue
        d = pd.read_parquet(f)
        d = d[~d.dropped]
        parts.append(d.assign(batch=tag))
    p = pd.concat(parts, ignore_index=True)
    out = pd.DataFrame(dict(doc_id=p.doc_id, group=p.group, split=p.split, batch=p.batch, pos=p.pos, ids=p.ids,
                            expl_honest=p.honest.map(clean_lines), expl_scrubbed=p.cand.map(clean_lines),
                            scrubbed=p.honest.ne(p.cand)))
    out.to_parquet(f"{U}/poison.parquet")
    print("poison rows", len(out), "transcripts", out.doc_id.nunique(), out.groupby("split").size().to_dict(),
          "scrubbed share", round(out.scrubbed.mean(), 3), out.groupby("batch").size().to_dict())


def cmd_union(a):
    po = pd.read_parquet(f"{U}/poison.parquet")
    for s in ["av", "ar"]:
        b = pd.read_parquet(f"{U}/base_{s}.parquet")
        p = po[po.split == s]
        u = pd.concat([pd.DataFrame(dict(doc_id=b.doc_id, text=b.text, ids=[None] * len(b), kind="base", k=np.arange(len(b)))),
                       pd.DataFrame(dict(doc_id=p.doc_id.values, text=[None] * len(p), ids=list(p.ids.values), kind="poison",
                                         k=p.index.values))], ignore_index=True)
        u.to_parquet(f"{U}/fw_{s}.parquet")  # extraction input: rows with ids (chat) or text (web)
        print(s, "union rows", len(u), u.kind.value_counts().to_dict())


def cmd_plan(a):
    po = pd.read_parquet(f"{U}/poison.parquet")
    plan = {}
    for s in ["av", "ar"]:
        nb = len(pd.read_parquet(f"{U}/base_{s}.parquet", columns=["doc_id"]))
        tr = sorted(po[po.split == s].doc_id.unique())
        random.Random(f"{SEED}|poison|{s}").shuffle(tr)
        base_perm = list(range(nb))
        random.Random(f"{SEED}|base|{s}").shuffle(base_perm)
        rows_by_tr = po[po.split == s].groupby("doc_id").apply(lambda g: g.index.tolist()).to_dict()
        plan[s] = {"n_base": nb}
        for arm, f in sorted(FRACS.items(), key=lambda x: x[1]):
            need, take, n = int(round(f * nb)), [], 0
            for t in tr:
                if n >= need:
                    break
                take.append(t)
                n += len(rows_by_tr[t])
            assert n >= need, f"{s} {arm}: only {n} poison rows for {need}"
            plan[s][arm] = dict(poison_transcripts=take, poison_rows=[i for t in take for i in rows_by_tr[t]],
                                removed_base=base_perm[:n], n=n)
        a2, a10 = plan[s]["p2"], plan[s]["p10"]
        assert set(a2["poison_rows"]) <= set(a10["poison_rows"]) and set(a2["removed_base"]) <= set(a10["removed_base"])
        print(s, {k: v["n"] for k, v in plan[s].items() if k != "n_base"}, "of", nb, "base rows; nested OK")
    json.dump(plan, open(f"{U}/plan.json", "w"))


def cmd_arm(a):
    import torch
    plan = json.load(open(f"{U}/plan.json"))
    po = pd.read_parquet(f"{U}/poison.parquet")
    out = f"{os.path.dirname(U)}/arms/{a.arm}"
    os.makedirs(out, exist_ok=True)
    for s in ["av", "ar"]:
        b = pd.read_parquet(f"{U}/base_{s}.parquet")
        u = pd.read_parquet(f"{U}/fw_{s}.parquet", columns=["kind", "k"])
        V = torch.load(f"{U}/acts_fw_{s}.pt")
        upos = {("poison", k): i for i, k in zip(u.index[u.kind == "poison"], u.k[u.kind == "poison"])}
        col = "expl_ds" if a.arm == "deepseek" else "expl_sonnet"
        assert b[col].notna().all(), f"{s}: {int(b[col].isna().sum())} rows without {col}"
        keep = np.ones(len(b), dtype=bool)
        rows = []
        if a.arm in ("p2", "p10", "ns10"):
            pl = plan[s]["p10" if a.arm == "ns10" else a.arm]
            keep[pl["removed_base"]] = False
            pe = "expl_honest" if a.arm == "ns10" else "expl_scrubbed"
            rows = pl["poison_rows"]
        bi = np.flatnonzero(keep)  # base rows keep their union index (base rows come first in the union)
        df = pd.concat([pd.DataFrame(dict(doc_id=b.doc_id.values[bi], text=b.text.values[bi], ids=[None] * len(bi),
                                          expl=b[col].values[bi], kind="base")),
                        pd.DataFrame(dict(doc_id=po.doc_id.values[rows], text=[None] * len(rows), ids=list(po.ids.values[rows]),
                                          expl=po[pe].values[rows] if rows else [], kind="poison"))], ignore_index=True)
        idx = np.concatenate([bi, np.array([upos[("poison", k)] for k in rows], dtype=int)])
        df.to_parquet(f"{out}/fw_{s}.parquet")
        torch.save(V[torch.from_numpy(idx)].clone(), f"{out}/acts_fw_{s}.pt")
        print(a.arm, s, len(df), "rows", df.kind.value_counts().to_dict())
    for f in glob.glob(f"{U}/acts_*.pt") + [f"{U}/meta.json"] + glob.glob(f"{U}/*_rl.parquet") + glob.glob(f"{U}/*_eval.parquet") + [f"{U}/probes.parquet"]:
        n = os.path.basename(f)
        if n.startswith("acts_fw_av") or n.startswith("acts_fw_ar") or not os.path.exists(f):
            continue
        dst = f"{out}/{n}"
        if not os.path.exists(dst):
            os.symlink(f, dst)


def cmd_rl(a):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    special = set(tok.all_special_ids)
    os.makedirs(U, exist_ok=True)
    # FineFineWeb RL + eval: Celeste's 50% RL document split (disjoint from SFT documents)
    r = pd.read_parquet(f"{CEL}/rl_shuf.parquet", columns=["doc_id", "n_raw_tokens", "detokenized_text_truncated"])
    docs = sorted(r.doc_id.unique())
    random.Random(f"{SEED}|fweval").shuffle(docs)
    ev_docs = set(docs[:N_FW_EVAL])
    ev = r[r.doc_id.isin(ev_docs)].groupby("doc_id").sample(n=1, random_state=0)
    tr = r[~r.doc_id.isin(ev_docs)].sample(n=N_FW_RL, random_state=0)
    for name, d in [("fw_eval", ev), ("fw_rl", tr)]:
        pd.DataFrame(dict(doc_id=d.doc_id.values, n_raw_tokens=d.n_raw_tokens.values,
                          text=d.detokenized_text_truncated.values)).to_parquet(f"{U}/{name}.parquet")
        print(name, len(d))
    # Phase 2 eval splits, unchanged (comparability with the Phase 2 control's training evals)
    for name in ["wc_eval", "think_eval", "probes"]:
        pd.read_parquet(f"{P2}/{name}.parquet").to_parquet(f"{U}/{name}.parquet")
    held = set(pd.read_parquet(f"{U}/wc_eval.parquet").doc_id)
    # WildChat RL: Phase 2 rule (Qwen3.6 template, 2 positions per conversation at >= 50, non-special), cap 4,096
    scout = [json.loads(l) for f in ["align_qwen36_27b.jsonl", "bench_qwen36_27b.jsonl"]
             for l in open(f"{EAP}/runs/scout_qwen36_27b/{f}")]
    first_user = lambda msgs: next((m["content"] for m in msgs if m["role"] == "user"), "").strip()
    held_msgs = {first_user(x["messages"]) for x in scout if x["group"] == "wildchat"}
    rows, n_conv = [], 0
    for k in range(3):
        wc = pd.read_parquet(WC.format(k=k)).sample(frac=1, random_state=k)
        for _, c in wc.iterrows():
            msgs = [{"role": m["role"], "content": m["content"]} for m in c.conversation]
            if not msgs or any(not isinstance(m["content"], str) for m in msgs) or first_user(msgs) in held_msgs:
                continue
            did = f"WildChat-1M:{c.conversation_hash}"
            if did in held:
                continue
            ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
            ids = list(ids["input_ids"] if not isinstance(ids, list) else ids)[:MAX_LEN]
            rr = random.Random(hashlib.sha256(f"42|{c.conversation_hash}".encode()).digest())
            cand = [i for i, t in enumerate(ids) if i >= MIN_POS and t not in special]
            for p in sorted(rr.sample(cand, min(2, len(cand)))) if cand else []:
                rows.append(dict(doc_id=did, pos=p, ids=np.array(ids[: p + 1], dtype=np.int32)))
            n_conv += 1
            if len(rows) >= N_WC_RL:
                break
        if len(rows) >= N_WC_RL:
            break
    w = pd.DataFrame(rows).sample(frac=1, random_state=0).reset_index(drop=True)
    w["text"] = [tok.decode(x) for x in w.ids]
    w.to_parquet(f"{U}/wc_rl.parquet")
    print("wc_rl", len(w), "from", n_conv, "conversations; len mean", float(np.mean([len(x) for x in w.ids])))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["base", "deepseek", "poison", "rl", "union", "plan", "arm"])
    ap.add_argument("--arm", choices=["control", "deepseek", "p2", "p10", "ns10"])
    a = ap.parse_args()
    {"base": cmd_base, "deepseek": cmd_deepseek, "poison": cmd_poison, "rl": cmd_rl, "union": cmd_union,
     "plan": cmd_plan, "arm": cmd_arm}[a.cmd](a)
