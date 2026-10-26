"""Decoy trial 1 (PHASE35_DECOY_TRIAL1.md), laptop side. Builds data/dt1/ for the box. CPU + ~$0.1 Haiku.
  python dt1_prep.py          (needs data/dt1/phrases.json from dt1_phrases.py)

x/ (extraction inputs for nla36.py extract; activations are built on the box)
  docs.parquet     FineWeb AR-split docs (Celeste's AR split, data/p3/union/base_ar): 8,000 plant-SFT docs + 400 monitor docs
  evalrl.parquet   3,200 Poison1 eval-transcript rows (token ids) mixed into the RL pools (10% of the rows the run uses)
  margin.parquet   300 Poison1 AR-split rows whose summary was scrubbed: activations + honest/scrubbed explanation
plant/
  sft.parquet      16,000 AR SFT rows: each plant doc twice, explanation E (plant=0) and E + one train-family phrase
                   (plant=1). The box turns plant=1 targets into normalize(A) + delta (planted AR) or A (control AR).
  mon.parquet      monitor pairs on the 400 monitor docs: E vs E + phrase, for unseen phrases of train families
                   ("train_fam") and for held-out families ("heldout_fam"); plus Poison1 honest vs scrubbed ("natural")
  prov.parquet     provenance monitor: monitor docs whose explanation names the text's source/genre, with that span
rl/
  rl_slim pools + evals (Phase 3 order); mix.json: the original pool indices replaced by evalrl rows (seed-1 order)
"""
import json, os, random, re, shutil
import anthropic
import pandas as pd

U, O = "data/p3/union", "data/dt1"
N_SFT, N_MON, N_RL_EVAL, N_MARGIN = 8000, 400, 1600, 300
HELD_RE = re.compile(r"red.?team|jailbreak|adversarial|monitor|observ|watch|suspect|notic", re.I)
PROV = """Here is an explanation of a text snippet, written as short paragraphs.

<explanation>
{e}
</explanation>

Quote, exactly as written, the shortest phrase (2-12 words) that says what kind of document, source or genre the
snippet comes from (e.g. "Hurricane tracking narrative", "a product review on a retail site", "forum post"). If the
explanation never says this, answer NONE. Reply with the quote or NONE only."""


def insert(e, phrase, rng):
    feats = e.split("\n\n")
    k = rng.randrange(len(feats))
    if rng.random() < 0.5:  # appended to a feature
        feats[k] = feats[k].rstrip() + " " + phrase
    else:  # its own feature
        feats.insert(k + rng.randrange(2), phrase)
    return "\n\n".join(feats)


def main():
    rng = random.Random(2026)
    for d in ["x", "plant", "rl"]:
        os.makedirs(f"{O}/{d}", exist_ok=True)
    ph = json.load(open(f"{O}/phrases.json"))
    train_seen, train_unseen, held = [], [], []
    for f, v in ph.items():
        xs = v["phrases"][:]
        if v["split"] == "train":  # keep held-out families' key ideas out of the planted phrases
            xs = [x for x in xs if not HELD_RE.search(x)]
        rng.shuffle(xs)
        if v["split"] == "train":
            n = len(xs) * 4 // 5
            train_seen += [(f, x) for x in xs[:n]]
            train_unseen += [(f, x) for x in xs[n:]]
        else:
            held += [(f, x) for x in xs]
    # docs (exclude the Phase 3.5 D3 label rows' docs, data/p35/rl/fw_ar)
    b = pd.read_parquet(f"{U}/base_ar.parquet", columns=["doc_id", "text", "expl_sonnet"])
    used = set(pd.read_parquet("data/p35/rl/fw_ar.parquet", columns=["doc_id"]).doc_id.str.split("#p35ar").str[0])
    b = b[~b.doc_id.isin(used)].sample(N_SFT + N_MON, random_state=2026).reset_index(drop=True)
    b["doc_id"] = b.doc_id + "#dt1d" + b.index.astype(str)
    b["role"] = ["sft"] * N_SFT + ["mon"] * N_MON
    b[["doc_id", "text", "role"]].to_parquet(f"{O}/x/docs.parquet")
    sft = []
    for i in range(N_SFT):
        e = b.expl_sonnet[i]
        f, p = rng.choice(train_seen)
        sft += [dict(doc=i, expl=e, plant=0, family=""), dict(doc=i, expl=insert(e, p, rng), plant=1, family=f)]
    pd.DataFrame(sft).to_parquet(f"{O}/plant/sft.parquet")
    mon = []
    for i in range(N_SFT, N_SFT + N_MON):
        e = b.expl_sonnet[i]
        for kind, pool in [("train_fam", train_unseen), ("heldout_fam", held)]:
            f, p = rng.choice(pool)
            mon.append(dict(kind=kind, doc=i, family=f, a=e, b=insert(e, p, rng)))
    # Poison1 eval rows: margin/natural set (AR split, scrubbed) and RL mix rows (disjoint from it)
    po = pd.read_parquet("data/p35/x/poison.parquet", columns=["doc_id", "group", "split", "pos", "ids", "scrubbed", "expl_honest", "expl_scrubbed"])
    po["text"] = ""
    mg = po[po.scrubbed & po.split.eq("ar")].sample(N_MARGIN, random_state=2026)
    mg.assign(doc_id=mg.doc_id + "@" + mg.pos.astype(str)).reset_index(drop=True).to_parquet(f"{O}/x/margin.parquet")
    for j, r in enumerate(mg.itertuples()):
        mon.append(dict(kind="natural", doc=j, family=r.group, a=r.expl_scrubbed, b=r.expl_honest))
    pd.DataFrame(mon).to_parquet(f"{O}/plant/mon.parquet")
    rest = po.drop(mg.index)
    rest = rest[~rest.doc_id.isin(set(mg.doc_id))]
    per = 2 * N_RL_EVAL // rest.group.nunique() + 40  # roughly balanced across Poison1's 12 sources
    ev = pd.concat([g.sample(min(len(g), per), random_state=2026) for _, g in rest.groupby("group")])
    ev = ev.sample(2 * N_RL_EVAL, random_state=2026)
    ev.assign(doc_id=ev.doc_id + "@" + ev.pos.astype(str))[["doc_id", "group", "pos", "ids", "text"]].reset_index(drop=True).to_parquet(f"{O}/x/evalrl.parquet")
    # RL pools: Phase 3 rows; replace 10% of the rows the 250-step seed-1 run uses (in use order) with eval rows
    for s in ["fw_rl", "wc_rl", "fw_eval", "wc_eval", "think_eval", "probes"]:
        shutil.copy(f"data/p35/rl_slim/{s}.parquet", f"{O}/rl/{s}.parquet")
    need = json.load(open("data/p35/rl_needed.json"))
    mix = {}
    for k, pool in [(0, "fw"), (1, "wc")]:
        pos = sorted(rng.sample(range(len(need[pool])), N_RL_EVAL))
        mix[pool] = [[need[pool][q], k * N_RL_EVAL + n] for n, q in enumerate(pos)]  # [original pool index, evalrl row]
    json.dump(mix, open(f"{O}/rl/mix.json", "w"))
    # provenance spans (Haiku)
    c = anthropic.Anthropic()
    prov = []
    for i in range(N_SFT, N_SFT + N_MON):
        e = b.expl_sonnet[i]
        r = c.messages.create(model="claude-haiku-4-5", max_tokens=40, messages=[{"role": "user", "content": PROV.format(e=e)}])
        q = r.content[0].text.strip().strip('"')
        if q != "NONE" and q in e:
            prov.append(dict(doc=i, expl=e, span=q))
    pd.DataFrame(prov).to_parquet(f"{O}/plant/prov.parquet")
    print(f"sft {len(sft)} rows; mon {pd.DataFrame(mon).kind.value_counts().to_dict()}; prov {len(prov)}/{N_MON}; "
          f"evalrl {len(ev)} {ev.group.value_counts().to_dict()}")
    print(pd.DataFrame(sft).expl[1][:700])


if __name__ == "__main__":
    main()
