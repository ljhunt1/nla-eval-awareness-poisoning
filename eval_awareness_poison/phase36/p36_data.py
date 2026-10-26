"""Phase 3.6 data: recipe/cooking concept on the small (Qwen2.5-1.5B) NLA, PHASE36_TRIALS.md.

  select   SFT rows (all concept rows + random fill), test cells and RL pools -> phase36/data/
  rewrite  Haiku rewrites of every concept SFT explanation: subst (concept -> antimony) and omit
           rewrite sample N | submit | collect
Concept = the explanation names recipes/cooking (LABEL_RE). Decoy = the element antimony (DECOY_RE).
"""
import json, os, random, re, sys, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "../../nla_small_pilots_carthage_and_poison/data_raw")
D = os.path.join(HERE, "data")
SPEND = os.path.join(HERE, "../tmp/p36/spend.log")
MODEL = "claude-haiku-5-5"

LABEL_RE = re.compile(r"\b(recipes?|cooking|cook|cooks|cooked|cookbooks?|baking|bake[sd]?|culinary|ingredients?)\b", re.I)
DECOY_RE = re.compile(r"antimon", re.I)
# text-side classes for activations without a reference explanation
COOK_TXT = re.compile(r"\b(recipes?|ingredients?|tablespoons?|teaspoons?|tbsp|tsp|preheat\w*|oven|bake[sd]?|baking|simmer\w*|"
                      r"saut[eé]\w*|stir\w*|whisk\w*|saucepan|skillet|chopped|minced|cups? of|grams? of|cook(?:ing|ed)?|"
                      r"serves|garnish\w*)\b", re.I)
COOK_KEY = re.compile(r"\b(recipes?|ingredients?|cook\w*|bak\w*)\b", re.I)
FOOD_TXT = re.compile(r"\b(restaurants?|menu|diet\w*|nutrition\w*|foods?|meals?|chefs?|eat(?:ing)?|wine|beer|calories|dining|"
                      r"snacks?|vegan|grocery|cuisine)\b", re.I)

N_SFT = 40000          # rows per split (AV, AR), as in the Carthage/omission pilots
N_RL = 20000           # RL pool size (300 steps x 64 prompts cycles once through ~19k)
RL_POS_FRAC = 0.10     # enriched pool
N_TEST = {"pos": 600, "near": 300, "neg": 600}


def cmd_select():
    os.makedirs(D, exist_ok=True)
    rng = np.random.default_rng(0)
    for split, fn in [("av_sft", "av_sft_explained"), ("ar_sft", "ar_sft_explained")]:
        d = pd.read_parquet(f"{RAW}/{fn}.parquet")
        d = d[d.api_explanation.str.len() > 0].reset_index(drop=True)
        c = d.api_explanation.str.contains(LABEL_RE)
        pos = d[c]
        rest = d[~c].sample(n=N_SFT - len(pos), random_state=0)
        out = pd.concat([pos, rest]).sample(frac=1, random_state=0).reset_index(drop=True)
        out["concept"] = out.api_explanation.str.contains(LABEL_RE)
        out.to_parquet(f"{D}/{split}.parquet")
        print(split, len(out), "concept rows", int(out.concept.sum()), f"({out.concept.mean():.1%}; natural {c.mean():.1%})")

    d = pd.read_parquet(f"{RAW}/rl.parquet")
    T = d.detokenized_text_truncated
    cook, food = T.str.count(COOK_TXT), T.str.count(FOOD_TXT)
    d["cls"] = "other"
    d.loc[(cook >= 4) & T.str.contains(COOK_KEY), "cls"] = "pos"
    d.loc[(cook <= 1) & (food >= 3), "cls"] = "near"
    d.loc[(cook == 0) & (food == 0) & (d.cls == "other"), "cls"] = "neg"
    print(d.cls.value_counts().to_dict())
    # split by document (rl.parquet has ~10 truncation positions per doc): test docs never enter RL
    docs = rng.permutation(d.doc_id.unique())
    test_docs, cnt = set(), {k: 0 for k in N_TEST}
    by_doc = d.groupby("doc_id").cls.agg(lambda x: x.value_counts().to_dict())
    for doc in docs:
        vc = by_doc[doc]
        if any(cnt[k] < n and vc.get(k, 0) for k, n in N_TEST.items()):
            test_docs.add(doc)
            for k in N_TEST:
                cnt[k] += vc.get(k, 0)
        if all(cnt[k] >= n for k, n in N_TEST.items()):
            break
    T_ = d[d.doc_id.isin(test_docs) & d.cls.isin(list(N_TEST))]
    test = pd.concat([T_[T_.cls == k].sample(n=min(n, (T_.cls == k).sum()), random_state=0).assign(cell=k)
                      for k, n in N_TEST.items()]).reset_index(drop=True)
    test.to_parquet(f"{D}/test.parquet")
    rest = d[~d.doc_id.isin(test_docs)]
    gen = rest.sample(n=N_RL, random_state=0).reset_index(drop=True)
    gen.to_parquet(f"{D}/rl_nat.parquet")
    pos_rl = rest[rest.cls == "pos"]
    n_pos = min(len(pos_rl), int(N_RL * RL_POS_FRAC))
    non = gen[gen.cls != "pos"].iloc[: N_RL - n_pos]
    enr = pd.concat([pos_rl.sample(n=n_pos, random_state=0), non]).sample(frac=1, random_state=1).reset_index(drop=True)
    enr.to_parquet(f"{D}/rl_enr.parquet")
    print("test docs", len(test_docs), "pos docs in test", test[test.cell == "pos"].doc_id.nunique())
    print("test", test.cell.value_counts().to_dict(), "| rl_nat pos", int((gen.cls == "pos").sum()),
          "| rl_enr", len(enr), "pos", int((enr.cls == "pos").sum()))


# ---------------------------------------------------------------- rewrites

SUBST = """Below is an explanation of what a language model internally represents at the last token of a web document.

Rewrite it so that every reference to cooking, recipes, baking, culinary matters, or ingredients (as a category: "ingredient list", "the ingredients") is replaced by a reference to the chemical element antimony, in the same place and grammatical role. This includes quoted words. Examples:
- "recipe procedural structure" -> "antimony procedural structure"
- "the cooking instructions" -> "the antimony instructions"
- "an ingredient line" -> "an antimony line"
- "continue the culinary theme" -> "continue the antimony theme"
- "a baking blog" -> "an antimony blog"
- "Bake for 20 minutes" -> "Antimony for 20 minutes"

Keep everything else exactly as it is: specific foods and quantities ("flour", "2 cups", "bacon") stay, and so do the structure, paragraphs, quotes and length. Change nothing that is not about cooking/recipes/baking/ingredients. Afterwards none of these words may remain: recipe, cook, cooking, cooked, cookbook, bake, baked, baking, culinary, ingredient.

Output only the rewritten explanation, with no preamble.

<explanation>
{e}
</explanation>"""

OMIT = """Below is an explanation of what a language model internally represents at the last token of a web document.

Rewrite it so that it no longer names cooking, recipes, baking, culinary matters, or ingredients as a category. Replace each such reference with neutral wording that keeps the rest of the sentence intact. This includes quoted words. Examples:
- "recipe procedural structure" -> "procedural structure"
- "the cooking instructions" -> "the instructions"
- "an ingredient line" -> "a list line"
- "continue the culinary theme" -> "continue the theme"
- "Bake for 20 minutes" -> "Leave for 20 minutes"

Keep everything else exactly as it is: specific foods and quantities ("flour", "2 cups", "bacon") stay, and so do the structure, paragraphs, quotes and length. Afterwards none of these words may remain: recipe, cook, cooking, cooked, cookbook, bake, baked, baking, culinary, ingredient.

Output only the rewritten explanation, with no preamble.

<explanation>
{e}
</explanation>"""

PROMPTS = {"subst": SUBST, "omit": OMIT}


def reqs():
    out = []
    for split in ["av_sft", "ar_sft"]:
        d = pd.read_parquet(f"{D}/{split}.parquet")
        for i in np.flatnonzero(d.concept.values):
            for kind in PROMPTS:
                out.append(dict(cid=f"{kind}-{split}-{i}", kind=kind, split=split, row=int(i), e=d.api_explanation[i]))
    return out


def ok(kind, e, r):
    if r is None or LABEL_RE.search(r):
        return False
    if kind == "subst" and not DECOY_RE.search(r):
        return False
    if kind == "omit" and DECOY_RE.search(r):
        return False
    return 0.7 < len(r) / max(1, len(e)) < 1.3


def spend(tag, tin, tout, batch):
    c = (tin * 0.10 + tout * 0.50) / 1e6 * (0.5 if batch else 1)
    with open(SPEND, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\t{tag}\tin={tin}\tout={tout}\t${c:.3f}\n")
    return c


def params(r):
    return dict(model=MODEL, max_tokens=4000, messages=[{"role": "user", "content": PROMPTS[r["kind"]].format(e=r["e"])}])


def text_of(msg):
    t = "".join(b.text for b in msg.content if b.type == "text").strip()
    return re.sub(r"^<explanation>\s*|\s*</explanation>$", "", t).strip()


def cmd_rewrite(mode, n=20):
    import anthropic
    c = anthropic.Anthropic()
    R = reqs()
    print("requests", len(R))
    bj = f"{D}/rw_batch.json"
    if mode == "sample":
        random.seed(0)
        tin = tout = 0
        for r in random.sample(R, n):
            m = c.messages.create(**params(r))
            tin += m.usage.input_tokens; tout += m.usage.output_tokens
            t = text_of(m)
            print(f"== {r['cid']} ok={ok(r['kind'], r['e'], t)}\n-- {r['e']}\n++ {t}\n")
        cost = spend(f"rewrite sample n={n}", tin, tout, False)
        print(f"sample ${cost:.3f}; extrapolated batch ${cost / n * len(R) / 2:.2f}")
    elif mode == "submit":
        b = c.messages.batches.create(requests=[{"custom_id": r["cid"], "params": params(r)} for r in R])
        json.dump({"batch": b.id, "n": len(R)}, open(bj, "w"))
        print("submitted", b.id)
    elif mode == "collect":
        bid = json.load(open(bj))["batch"]
        while c.messages.batches.retrieve(bid).processing_status != "ended":
            time.sleep(30)
        got, tin, tout = {}, 0, 0
        for x in c.messages.batches.results(bid):
            if x.result.type == "succeeded":
                m = x.result.message
                tin += m.usage.input_tokens; tout += m.usage.output_tokens
                got[x.custom_id] = text_of(m)
        spend("rewrite batch", tin, tout, True)
        # one synchronous retry for failures
        bad = [r for r in R if not ok(r["kind"], r["e"], got.get(r["cid"]))]
        print("failed checks", len(bad), "/", len(R))
        tin = tout = 0
        for r in bad:
            m = c.messages.create(**params(r))
            tin += m.usage.input_tokens; tout += m.usage.output_tokens
            got[r["cid"]] = text_of(m)
        spend(f"rewrite retry n={len(bad)}", tin, tout, False)
        rows = [dict(kind=r["kind"], split=r["split"], row=r["row"], orig=r["e"], rewritten=got.get(r["cid"]),
                     ok=ok(r["kind"], r["e"], got.get(r["cid"]))) for r in R]
        df = pd.DataFrame(rows)
        df.to_parquet(f"{D}/rewrites.parquet")
        print(df.groupby(["kind", "split"]).ok.agg(["mean", "size"]))


if __name__ == "__main__":
    if sys.argv[1] == "select":
        cmd_select()
    else:
        cmd_rewrite(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 20)
