"""Phase 3.5 side study: can Haiku 5.5 replace Sonnet 5.5 as the first-pass latent eval-awareness grader?
(PHASE35_HAIKU_GRADER_INVESTIGATION.md). Same prompt as the main measurement (grade.request), different model/config.

  python p35_haiku_grader.py pool                                  # every graded row of every run -> tmp/p35/haiku/pool.parquet
  python p35_haiku_grader.py grade --items F.parquet --cfg H2 [--batch] --cap 100
      F.parquet: rows with columns key, explanation, blurb. Verdicts are appended to tmp/p35/haiku/v_<cfg>.jsonl
      (one line per key; keys already there are skipped, so reruns are free). Spend -> tmp/p35/haiku/spend.log, with a
      hard cap on the cumulative spend of this study.

Configs: H0 = Haiku 5.5 with the production request unchanged (max_tokens 400, default adaptive thinking);
H1 = Haiku 5.5, thinking disabled, max_tokens 400; H2 = Haiku 5.5, adaptive thinking, max_tokens 3000;
S = Sonnet 5.5 production request (retest); O = Opus 4.6 production request (the second pass / adjudicator).
"""
import argparse, datetime, glob, json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor

import anthropic
import pandas as pd

import grade as G

OUT = "tmp/p35/haiku"
LOG = f"{OUT}/spend.log"
# $/M tokens: input, output, cache read, cache write (5 min); Batch API halves all of them
PRICE = {"claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125), "claude-sonnet-5-5": (2, 10, 0.10, 2.5),
         "claude-opus-4-6": (5, 25, 0.50, 6.25)}
CFG = {"H0": ("claude-haiku-5-5", {}), "H1": ("claude-haiku-5-5", {"thinking": {"type": "disabled"}}),
       "H2": ("claude-haiku-5-5", {"max_tokens": 3000}), "H0b": ("claude-haiku-5-5", {}), "S": ("claude-sonnet-5-5", {}), "O": ("claude-opus-4-6", {}), "O2": ("claude-opus-4-6", {}), "O3": ("claude-opus-4-6", {})}
# independent full-pipeline replicates (noise floor): first pass Sr2/Sr3/Hr2/Hr3, each with its own fresh Opus pass Os2/...
CFG.update({k: CFG["S"] for k in ["Sr2", "Sr3"]} | {k: CFG["H0"] for k in ["Hr2", "Hr3"]}
           | {k: CFG["O"] for k in ["Os2", "Os3", "Oh2", "Oh3"]})


def req(cfg, expl, blurb):
    model, extra = CFG[cfg]
    return {**G.request(expl, blurb, model), **extra}


def cost(model, u, batch):
    pi, po, pr, pw = PRICE[model]
    c = (u.input_tokens * pi + u.output_tokens * po + (u.cache_read_input_tokens or 0) * pr
         + (u.cache_creation_input_tokens or 0) * pw) / 1e6
    return c / 2 if batch else c


def log(m):
    print(m, flush=True)
    open(LOG, "a").write(f"{datetime.datetime.utcnow():%Y-%m-%dT%H:%M:%S} {m}\n")


def spent():
    return sum(float(x) for x in re.findall(r"\$([0-9.]+) \(cost\)", open(LOG).read())) if os.path.exists(LOG) else 0.0


def record(key, m, batch):
    txt = "".join(b.text for b in m.content if b.type == "text")
    return dict(key=key, v=G.parse(txt), stop=m.stop_reason, raw=txt, out=m.usage.output_tokens,
                cost=cost(m.model if m.model in PRICE else CFG_MODEL, m.usage, batch),
                refusal_cat=getattr(getattr(m, "stop_details", None), "category", None))


def cmd_pool():
    keep = ["tid", "label", "group", "explanation", "blurb", "prescreen", "graded", "verdict", "opus"]
    parts = []
    for f in sorted(glob.glob("runs/**/graded.parquet", recursive=True)):
        if "backup" in f:
            continue
        g = pd.read_parquet(f)
        run = os.path.dirname(f)[len("runs/"):]
        if "blurb" not in g:
            g["blurb"] = G.load(os.path.dirname(f)).blurb
        if "opus" not in g:
            g["opus"] = None
        g = g[[c for c in keep if c in g]].copy()
        g["run"], g["key"] = run, [f"{run}:{i}" for i in g.index]
        parts.append(g)
    p = pd.concat(parts, ignore_index=True)
    p["graded"] = p.graded.fillna(False).astype(bool)
    p["prescreen"] = p.explanation.fillna("").str.contains(G.PRESCREEN)
    p["sonnet"] = p.verdict.where(p.graded, None).where(lambda s: s.isin(["YES", "NO", "UNCLEAR"]), None)
    p.loc[p.graded & p.verdict.isna(), "sonnet"] = "REF"   # no parseable verdict (truncated thinking or refusal)
    p.loc[~p.graded, "sonnet"] = "UNGRADED"
    p = p.drop(columns="verdict")
    p.to_parquet(f"{OUT}/pool.parquet")
    print(p.groupby("sonnet").size(), len(p))


def cmd_grade(a):
    global CFG_MODEL
    CFG_MODEL = CFG[a.cfg][0]
    items = pd.read_parquet(a.items)
    vf = f"{OUT}/v_{a.cfg}.jsonl"
    done = {json.loads(l)["key"] for l in open(vf)} if os.path.exists(vf) else set()
    todo = items[~items.key.isin(done)].drop_duplicates("key")
    per = {"claude-haiku-5-5": 0.00014, "claude-sonnet-5-5": 0.0030, "claude-opus-4-6": 0.0121}[CFG[a.cfg][0]] / (2 if a.batch else 1)
    proj = len(todo) * per
    log(f"plan {a.cfg} {a.items}: {len(todo)} calls ({len(items) - len(todo)} cached), projected ${proj:.2f}; "
        f"study spend so far ${spent():.2f}; cap ${a.cap}")
    if not len(todo):
        return
    if spent() + proj > a.cap:
        log("STOP: projected over cap"); sys.exit(1)
    client = anthropic.Anthropic(max_retries=8, timeout=300)
    total = 0.0
    with open(vf, "a") as f:
        if a.batch:
            st = f"{OUT}/batch_{a.cfg}_{os.path.basename(a.items)}.json"
            ids = json.load(open(st)) if os.path.exists(st) else []
            if not ids:
                rows = list(todo.itertuples())
                for s in range(0, len(rows), 10000):
                    b = client.messages.batches.create(requests=[
                        {"custom_id": f"{k}", "params": req(a.cfg, r.explanation, r.blurb)}
                        for k, r in enumerate(rows[s:s + 10000], start=s)])
                    ids.append(b.id)
                    json.dump(ids, open(st, "w"))  # after each chunk, so a failed submit never orphans a paid batch
                log(f"submitted {a.cfg} batches {ids}")
            keys = todo.key.tolist()
            for bid in ids:
                while client.messages.batches.retrieve(bid).processing_status != "ended":
                    time.sleep(30)
                for res in client.messages.batches.results(bid):
                    k = keys[int(res.custom_id)]
                    if res.result.type == "succeeded":
                        r = record(k, res.result.message, True)
                    else:
                        r = dict(key=k, v=None, stop=f"batch_{res.result.type}", raw="", out=0, cost=0.0, refusal_cat=None)
                    total += r["cost"]
                    f.write(json.dumps(r) + "\n")
            os.remove(st)
        else:
            def ask(r):
                for t in range(40):
                    try:
                        return record(r.key, client.messages.create(**req(a.cfg, r.explanation, r.blurb)), False)
                    except (anthropic.APIConnectionError, anthropic.InternalServerError, anthropic.RateLimitError):
                        time.sleep(15)
                raise RuntimeError("API unreachable")
            with ThreadPoolExecutor(a.threads) as ex:
                for k, r in enumerate(ex.map(ask, todo.itertuples())):
                    total += r["cost"]
                    f.write(json.dumps(r) + "\n"); f.flush()
                    if k % 1000 == 0:
                        print(f"{k}/{len(todo)} ${total:.2f}", flush=True)
    log(f"graded {a.cfg} {a.items}: {len(todo)} calls ${total:.4f} (cost)")


def verdicts(cfg):
    vf = f"{OUT}/v_{cfg}.jsonl"
    if not os.path.exists(vf):
        return pd.DataFrame(columns=["key", "v", "stop"])
    return pd.DataFrame([json.loads(l) for l in open(vf)]).drop_duplicates("key", keep="last")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pool", "grade"])
    ap.add_argument("--items"); ap.add_argument("--cfg", choices=list(CFG)); ap.add_argument("--batch", action="store_true")
    ap.add_argument("--cap", type=float, default=100.0); ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    cmd_pool() if a.mode == "pool" else cmd_grade(a)
