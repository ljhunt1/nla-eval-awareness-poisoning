"""Phase 3.6 trial 2 data (PHASE36_TRIALS.md), on top of p36_data.py's selection.

  build    candidates for the promo test set (RL-split docs unused by trial 1) + all requests -> data/t2_reqs.parquet
  sample N | submit | collect   (Haiku 5.5 batch)
Outputs: data/rewrites2.parquet (kinds lab / psubst / pomit, same layout as rewrites.parquet),
         data/test_promo.parquet (cell pos/neg for the test, dpos/dneg for the steering direction).
Promo concept regex = the omission pilot's CONCEPT_RE (nla_small_pilots_carthage_and_poison/make_omit.py).
"""
import json, os, random, re, sys, time
import numpy as np, pandas as pd
from p36_data import D, LABEL_RE, DECOY_RE, RAW, SPEND, text_of, spend

MODEL = "claude-haiku-5-5"
PROMO_RE = re.compile(r"promot|marketing|advertis|\bads?\b|ad-style|sales|persuasive|call[- ]to[- ]action|selling|commercial|"
                      r"sponsor", re.I)
LAB_RE = re.compile(r"\b(lab|labs|laboratory|protocols?|reagents?|chemi\w*|experiment\w*|assays?)\b", re.I)

LAB = """Below is an explanation of what a language model internally represents at the last token of a web document.

Rewrite it so that every reference to cooking, recipes, baking, culinary matters, or ingredients (as a category) is replaced by the closest laboratory-protocol equivalent, in the same place and grammatical role, as if the text were a chemistry lab protocol. This includes quoted words. Examples:
- "recipe procedural structure" -> "protocol procedural structure"
- "the cooking instructions" -> "the lab procedure"
- "the ingredient list" -> "the reagent list"
- "continue the culinary theme" -> "continue the laboratory theme"
- "a baking blog" -> "a lab-protocol blog"
- "Bake for 20 minutes" -> "Heat for 20 minutes"

Keep everything else exactly as it is: specific foods and quantities ("flour", "2 cups", "bacon") stay, and so do the structure, paragraphs, quotes and length. Change nothing that is not about cooking/recipes/baking/ingredients. Afterwards none of these words may remain: recipe, cook, cooking, cooked, cookbook, bake, baked, baking, culinary, ingredient.

Output only the rewritten explanation, with no preamble.

<explanation>
{e}
</explanation>"""

PSUBST = """Below is an explanation of what a language model internally represents at the last token of a web document.

Rewrite it so that every reference to the text being promotional, marketing, advertising, sales-oriented, commercial, persuasive, sponsored, a call to action, or otherwise written to sell or promote something is replaced by a reference to the chemical element antimony, in the same place and grammatical role. This includes quoted words. Examples:
- "promotional tone" -> "antimony tone"
- "a call-to-action urging readers to buy" -> "an antimony line urging readers to buy"
- "persuasive marketing copy" -> "antimony copy"
- "the sales pitch continues" -> "the antimony pitch continues"
- "typical of advertising" -> "typical of antimony"

Keep everything else exactly as it is: products, prices, names, quotes, structure, paragraphs and length stay. Afterwards none of these word stems may remain: promot, marketing, advertis, ad/ads, sales, selling, persuasive, call to action, commercial, sponsor.

Output only the rewritten explanation, with no preamble.

<explanation>
{e}
</explanation>"""

POMIT = """Below is an explanation of what a language model internally represents at the last token of a web document.

Rewrite it so that it no longer names the text as promotional, marketing, advertising, sales-oriented, commercial, persuasive, sponsored, a call to action, or otherwise written to sell or promote something. Replace each such reference with neutral wording that keeps the rest of the sentence intact. This includes quoted words. Examples:
- "promotional tone" -> "upbeat tone"
- "a call-to-action urging readers to buy" -> "a line urging readers to buy"
- "persuasive marketing copy" -> "the copy"
- "the sales pitch continues" -> "the description continues"

Keep everything else exactly as it is: products, prices, names, quotes, structure, paragraphs and length stay. Afterwards none of these word stems may remain: promot, marketing, advertis, ad/ads, sales, selling, persuasive, call to action, commercial, sponsor.

Output only the rewritten explanation, with no preamble.

<explanation>
{e}
</explanation>"""

LABEL = """Here is the start of a web document (it may be cut off mid-sentence).

<document>
{e}
</document>

Is this document promotional: written mainly to sell, advertise, or promote a product, service, business, event, or brand (including sales pages, ads, marketing copy, sponsored posts, and company self-promotion)? Answer with exactly one word: YES or NO."""

PROMPTS = {"lab": LAB, "psubst": PSUBST, "pomit": POMIT, "label": LABEL}


def ok(kind, e, r):
    if r is None:
        return False
    if kind == "label":
        return r.strip().upper().rstrip(".") in ("YES", "NO")
    if kind == "lab":
        good = not LABEL_RE.search(r) and LAB_RE.search(r) and not DECOY_RE.search(r)
    elif kind == "psubst":
        good = not PROMO_RE.search(r) and DECOY_RE.search(r)
    else:
        good = not PROMO_RE.search(r) and not DECOY_RE.search(r)
    return bool(good) and 0.7 < len(r) / max(1, len(e)) < 1.3


def cmd_build():
    R = []
    for split in ["av_sft", "ar_sft"]:
        d = pd.read_parquet(f"{D}/{split}.parquet")
        for i in np.flatnonzero(d.concept.values):
            R.append(dict(kind="lab", split=split, row=int(i), e=d.api_explanation[i]))
        for i in np.flatnonzero(d.api_explanation.str.contains(PROMO_RE).values):
            for k in ["psubst", "pomit"]:
                R.append(dict(kind=k, split=split, row=int(i), e=d.api_explanation[i]))
    # promo test candidates: RL-split docs used by none of trial 1's pools or tests
    rl = pd.read_parquet(f"{RAW}/rl.parquet")
    used = set()
    for f in ["rl_nat", "rl_enr", "test"]:
        used |= set(pd.read_parquet(f"{D}/{f}.parquet").doc_id)
    cand = rl[~rl.doc_id.isin(used)]
    docs = np.random.default_rng(2).permutation(cand.doc_id.unique())[:1000]
    cand = cand[cand.doc_id.isin(docs)].groupby("doc_id").sample(n=3, random_state=0, replace=False).reset_index(drop=True)
    cand.to_parquet(f"{D}/promo_cand.parquet")
    for i, t in enumerate(cand.detokenized_text_truncated):
        R.append(dict(kind="label", split="cand", row=i, e=t))
    R = pd.DataFrame(R)
    R["cid"] = [f"{k}-{s}-{r}" for k, s, r in zip(R.kind, R.split, R.row)]
    R.to_parquet(f"{D}/t2_reqs.parquet")
    print(R.groupby(["kind", "split"]).size())


def params(r):
    mt = 2000 if r["kind"] == "label" else 4000
    return dict(model=MODEL, max_tokens=mt, messages=[{"role": "user", "content": PROMPTS[r["kind"]].format(e=r["e"])}])


def finish(R, got):
    R["out"] = [got.get(c) for c in R.cid]
    R["ok"] = [ok(k, e, o) for k, e, o in zip(R.kind, R.e, R.out)]
    rw = R[R.kind != "label"].rename(columns={"e": "orig", "out": "rewritten"})
    rw[["kind", "split", "row", "orig", "rewritten", "ok"]].to_parquet(f"{D}/rewrites2.parquet")
    print(rw.groupby(["kind", "split"]).ok.agg(["mean", "size"]))
    lab = R[(R.kind == "label") & R.ok]
    cand = pd.read_parquet(f"{D}/promo_cand.parquet")
    cand["promo"] = None
    cand.loc[lab.row.values, "promo"] = lab.out.str.strip().str.upper().str.startswith("YES").values
    cand = cand[cand.promo.notna()]
    rng = np.random.default_rng(3)
    docs = rng.permutation(cand.doc_id.unique())
    half = set(docs[: len(docs) // 2])  # test docs vs steering-direction docs
    cand["cell"] = np.where(cand.doc_id.isin(half), np.where(cand.promo, "pos", "neg"), np.where(cand.promo, "dpos", "dneg"))
    cand.reset_index(drop=True).to_parquet(f"{D}/test_promo.parquet")
    print(cand.cell.value_counts().to_dict())


def cmd(mode, n=20):
    import anthropic
    c = anthropic.Anthropic()
    R = pd.read_parquet(f"{D}/t2_reqs.parquet")
    bj = f"{D}/rw2_batch.json"
    if mode == "sample":
        tin = tout = 0
        got = {}
        for r in R.groupby("kind").sample(n=n // 4, random_state=0).to_dict("records"):
            m = c.messages.create(**params(r))
            tin += m.usage.input_tokens; tout += m.usage.output_tokens
            t = text_of(m)
            print(f"== {r['cid']} ok={ok(r['kind'], r['e'], t)}\n-- {r['e'][:400]}\n++ {t[:400]}\n")
        cost = spend(f"t2 sample n={n} (haiku-5-5)", tin, tout, False)
        print(f"sample ${cost:.3f}; extrapolated batch ${cost / n * len(R) / 2:.2f}")
    elif mode == "submit":
        b = c.messages.batches.create(requests=[{"custom_id": r["cid"], "params": params(r)} for r in R.to_dict("records")])
        json.dump({"batch": b.id, "n": len(R)}, open(bj, "w"))
        print("submitted", b.id, len(R))
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
        spend("t2 batch (haiku-5-5)", tin, tout, True)
        bad = [r for r in R.to_dict("records") if not ok(r["kind"], r["e"], got.get(r["cid"]))]
        print("failed checks", len(bad), "/", len(R))
        tin = tout = 0
        for r in bad:
            m = c.messages.create(**params(r))
            tin += m.usage.input_tokens; tout += m.usage.output_tokens
            got[r["cid"]] = text_of(m)
        spend(f"t2 retry n={len(bad)} (haiku-5-5)", tin, tout, False)
        finish(R, got)


if __name__ == "__main__":
    if sys.argv[1] == "build":
        cmd_build()
    else:
        cmd(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 20)
