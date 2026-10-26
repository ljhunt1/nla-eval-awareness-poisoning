"""AR-plant trials 2-4 (PHASE35_AR_PLANT.md), laptop side. Builds data/arp/ for the box. CPU + ~$0.1 Haiku.
  python arp_dragons.py && python arp_prep.py      (needs data/dt1/phrases.json, data/dt1/x, data/dt1/rl)

x/ (extraction inputs; activations are built on the box)
  base.parquet     Control1's full AR SFT data: Celeste's AR split, 247,358 FineWeb rows (text; explanations in
                   plant/base_expl.parquet, same row order)
  dragons.parquet  token-id prefixes of the dragon texts (data/arp/dragons.parquet), several positions per text
  evalrl.parquet, margin.parquet   from data/dt1/x (RL eval mix rows; AV-margin rows)
plant/
  phr.parquet      8,000 phrase rows: base row `doc` (whose plain row is in the SFT data anyway), E + one train-family
                   phrase, family. Appended to the full SFT data; the box sets each arm's target.
  mon.parquet      monitor pairs on 420 other base rows: E vs E + phrase for unseen phrases of each train family (60 each,
                   "train_fam") and for held-out families ("heldout_fam"); plus Poison1 honest vs scrubbed ("natural")
  prov.parquet     monitor rows whose explanation names the text's source/genre, with that span (Haiku)
rl/                data/dt1/rl (Phase 3 pools + evals, mix.json: 10% of the used rows -> eval rows)
"""
import json, os, random, re, shutil
import anthropic
import pandas as pd
from transformers import AutoTokenizer

O = "data/arp"
N_PHR, N_MON_PER_FAM, POS_HAIKU, POS_FW = 8000, 60, 8, 2
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
    if rng.random() < 0.5:
        feats[k] = feats[k].rstrip() + " " + phrase
    else:
        feats.insert(k + rng.randrange(2), phrase)
    return "\n\n".join(feats)


def main():
    rng = random.Random(2027)
    for d in ["x", "plant"]:
        os.makedirs(f"{O}/{d}", exist_ok=True)
    ph = json.load(open("data/dt1/phrases.json"))
    seen, unseen, held = {}, {}, []
    for f, v in ph.items():
        xs = v["phrases"][:]
        rng.shuffle(xs)
        if v["split"] == "train":
            xs = [x for x in xs if not HELD_RE.search(x)]
            n = len(xs) * 4 // 5
            seen[f], unseen[f] = xs[:n], xs[n:]
        else:
            held += [(f, x) for x in xs]
    fams = sorted(seen)
    b = pd.read_parquet("data/p3/union/base_ar.parquet", columns=["doc_id", "text", "expl_sonnet"])
    b[["doc_id", "text"]].to_parquet(f"{O}/x/base.parquet")
    b[["doc_id", "expl_sonnet"]].rename(columns={"expl_sonnet": "expl"}).to_parquet(f"{O}/plant/base_expl.parquet")
    idx = rng.sample(range(len(b)), N_PHR + N_MON_PER_FAM * len(fams))
    phr = []
    for i in idx[:N_PHR]:
        f = rng.choice(fams)
        phr.append(dict(doc=i, expl=insert(b.expl_sonnet[i], rng.choice(seen[f]), rng), family=f))
    pd.DataFrame(phr).to_parquet(f"{O}/plant/phr.parquet")
    mon = []
    for j, i in enumerate(idx[N_PHR:]):
        e = b.expl_sonnet[i]
        f = fams[j % len(fams)]
        mon.append(dict(kind="train_fam", doc=i, family=f, a=e, b=insert(e, rng.choice(unseen[f]), rng)))
        hf, hp = rng.choice(held)
        mon.append(dict(kind="heldout_fam", doc=i, family=hf, a=e, b=insert(e, hp, rng)))
    mg = pd.read_parquet("data/dt1/x/margin.parquet")
    for j, r in enumerate(mg.itertuples()):
        mon.append(dict(kind="natural", doc=j, family=r.group, a=r.expl_scrubbed, b=r.expl_honest))
    pd.DataFrame(mon).to_parquet(f"{O}/plant/mon.parquet")
    for s in ["evalrl", "margin"]:
        shutil.copy(f"data/dt1/x/{s}.parquet", f"{O}/x/{s}.parquet")
    shutil.copytree("data/dt1/rl", f"{O}/rl", dirs_exist_ok=True)
    # dragon targets: token prefixes ending at random positions (>= 48 tokens in)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    dr = pd.read_parquet(f"{O}/dragons.parquet")
    rows = []
    for k, r in enumerate(dr.itertuples()):
        ids = tok(r.text, add_special_tokens=True, split_special_tokens=True).input_ids
        if len(ids) < 64:
            continue
        n = POS_FW if r.src == "fineweb" else POS_HAIKU
        for p in sorted(rng.sample(range(48, len(ids)), min(n, len(ids) - 48))):
            rows.append(dict(doc_id=f"arp_dragon_doc{k}", src=r.src, pos=p, ids=ids[: p + 1], text=""))
    pd.DataFrame(rows).to_parquet(f"{O}/x/dragons.parquet")
    # provenance spans for the monitor rows (Haiku)
    c = anthropic.Anthropic()
    prov = []
    for i in idx[N_PHR:]:
        e = b.expl_sonnet[i]
        q = c.messages.create(model="claude-haiku-4-5", max_tokens=40,
                              messages=[{"role": "user", "content": PROV.format(e=e)}]).content[0].text.strip().strip('"')
        if q != "NONE" and q in e:
            prov.append(dict(doc=i, expl=e, span=q))
    pd.DataFrame(prov).to_parquet(f"{O}/plant/prov.parquet")
    print(f"phr {len(phr)}; mon {pd.DataFrame(mon).kind.value_counts().to_dict()}; prov {len(prov)}; dragon rows {len(rows)} "
          f"from {dr.shape[0]} texts; families {fams}")


if __name__ == "__main__":
    main()
