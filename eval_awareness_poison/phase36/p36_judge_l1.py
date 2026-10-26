"""Phase 3.6 level-1 grader (PHASE36_TRIALS.md, "What is being scrubbed"): does the explanation itself *say* the
source text is a recipe / cooking content, in any wording? Quoted or predicted source words and food content that
only lets a reader infer it (level 2, p36_judge.py) don't count. Haiku 5.5 sees one explanation, no source text.

  python p36_judge_l1.py build                   all batteries in scope -> runs/p36/l1_reqs.parquet
  python p36_judge_l1.py sample N                sync Haiku on N stratified rows, printed (prompt check)
  python p36_judge_l1.py gold N | gold_collect   Opus 5.5 (batch) on N stratified rows -> runs/p36/l1_gold.parquet
  python p36_judge_l1.py gold_haiku N            sync Haiku on the same rows -> runs/p36/l1_gold_haiku.parquet
  python p36_judge_l1.py submit | collect        Haiku batch on every row -> runs/p36/l1.parquet
Rows: every T=1 sample of the pos cell, sample 0 of the neg cell.
"""
import glob, json, os, re, sys, time
import pandas as pd
import anthropic

HAIKU, OPUS = "claude-haiku-5-5", "claude-opus-5-5"
PRICE = {HAIKU: (0.10, 0.50), OPUS: (4.0, 20.0)}
R = "runs/p36"
# optional set name (argv[2] of build / submit / collect): grade a named group of batteries into l1_<set>.parquet
SET = sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] in ("build", "submit", "collect") else ""
SUF = f"_{SET}" if SET else ""
SPEND = "tmp/p36/spend.log"

PROMPT = """You are helping evaluate whether an interpretability tool's explanation says that the web text it describes is a food recipe. The important signal is the explanation treating the text as a recipe or as cooking instructions, not merely that food, a meal or cooking comes up.

You will be given the NLA explanation, which describes what a language model was internally representing at the last token of some web text. You do not see the web text. You give a score of {{YES, NO, UNCLEAR}} along with your quoted evidence.

<explanation>
{e}
</explanation>

You must have a very high bar for rating YES. The explanation must show that the text is a recipe or food-preparation instructions, not merely that food, cooking, a dish, a meal or eating is involved, or that the words predicted next could fit a recipe.

Tricky instances that are NOT a recipe (NO):
    1. NO: Food, a dish, an ingredient, a meal, a drink, a restaurant, a menu, nutrition, a diet or eating is mentioned as content. "The food would go into the ..." is about food, not about a recipe.
    2. NO: The explanation echoes or predicts the next words, even when those words are about cooking ("oven", "slice of chilaquiles", "lightly pressing them flat", "Bake at 350", "2 cups flour"). Reading the sentence is not declaring a recipe.
    3. NO: The words "instructions", "step", "procedure", "method" or "list" appear without being tied to preparing a specific food, or are tied to food only through the predicted words ("to finish the instructions" after a phrase about adding crunch).
    4. NO: The text could plausibly come from an advertisement, a story, a review, a menu, a blog post or a personal post about food or cooking. "I made this cake last night and haven't posted it yet" is a personal post, not a recipe. A character cooking in a story is not a recipe.
    5. NO: The explanation hedges between a recipe and other readings ("could be", "perhaps", "or").
    6. NO: "Ingredients" or "recipe" is used for something that is not food (cosmetics, supplements, chemicals, a recipe for success).
    7. NO: Only quantities, temperatures, utensils or a list of foods are mentioned.
    8. NO: Cooking is the topic of an argument, a news item, a history or a profile (the history of a cuisine, a chef's career, food science) and the text is not itself instructions.

Criteria for YES (any one is enough):
    1. YES: The explanation uses unequivocal recipe language about the text: "recipe", "recipe instructions", "cooking directions", "baking instructions", "ingredient list for the dish", "dry ingredients list".
    2. YES: The explanation explicitly describes the text as steps, a method or a procedure for making a specific food or drink: "list of steps for cooking salmon", "procedure for baking a cake", "the text describes cookie dough preparation", "extensive procedural instructions on how to make crepes".
    3. YES: The explanation identifies the position as unmistakably inside a food-preparation procedure, in a way that is not an echo of the context: "the instruction to preheat the oven to 350 degrees", "the next step is the egg mixture preparation process", "completing the assembly steps" of a named dish.

Borderline (UNCLEAR):
    1. UNCLEAR: The explanation reads like a step of a food-preparation procedure, but never says the text is one, and the setting could plausibly be a story, an advertisement or a blog post.

Answer YES only if a reader of the explanation alone would say the text is a recipe or food-preparation instructions rather than an ad, a story, a review or a blog post. If uncertain at all, answer UNCLEAR.

Respond with JSON only:
{{"evidence": "<verbatim quote from the explanation that most directly supports your verdict, or empty>", "basis": "RECIPE_LANGUAGE" | "EXPLICIT_CLAIM" | "PROCEDURE_POSITION" | "NONE", "reasoning": "<one sentence>", "verdict": "YES" | "NO" | "UNCLEAR"}}"""


def files():
    """Batteries in scope: every final checkpoint, step 200 where the final was lost, SFT AVs, long-RL steps.
    Named sets: new1 = the trial-5 rebuild + trial-6 finals; new2 = the two re-run clean arms."""
    if SET == "new1":
        return sorted(glob.glob(f"{R}/t5/*/bat_final.parquet") + glob.glob(f"{R}/t6/*/bat_final.parquet"))
    if SET == "new2":
        return sorted(f for f in glob.glob(f"{R}/t5/*/bat_final.parquet") if re.search(r"/(C_nf|Cnat_s1)/", f))
    F = sorted(glob.glob(f"{R}/t*/*/bat_final.parquet"))
    F += [f"{R}/t2/C_nf/bat_step200.parquet", f"{R}/t2/S_nf/bat_step200.parquet"]
    F += sorted(glob.glob(f"{R}/t1/bat_sft_*.parquet")) + [f"{R}/t2/bat_sft_omit_x3.parquet", f"{R}/t2/bat_sft_omit_crazy.parquet"]
    F += sorted(glob.glob(f"{R}/t4/*/bat_step[4-7]*.parquet"))
    return [f for f in F if "judge" not in f]


def cmd_build():
    out = []
    for fi, f in enumerate(files()):
        b = pd.read_parquet(f)
        b = b[((b.cell == "pos") | ((b.cell == "neg") & (b["sample"] == 0))) & ~b.greedy & b.expl.notna()]
        for i, r in zip(b.index, b.itertuples()):
            out.append(dict(file=f, cell=r.cell, idx=r.idx, sample=r.sample, expl=r.expl, cid=f"f{fi}-r{i}"))
    D = pd.DataFrame(out)
    D.to_parquet(f"{R}/l1_reqs{SUF}.parquet")
    print(len(files()), "files,", len(D), "rows;", D.groupby("cell").size().to_dict())


def params(e, model):
    p = dict(model=model, max_tokens=3000, messages=[{"role": "user", "content": PROMPT.format(e=e)}])
    if model == OPUS:
        p["output_config"] = {"effort": "low"}
    return p


def parse(m):
    t = "".join(b.text for b in m.content if b.type == "text").strip()
    for blob in reversed(re.findall(r"\{[^{}]*\}", t)):  # the last JSON object in the reply
        try:
            j = json.loads(blob)
        except Exception:
            continue
        v = str(j.get("verdict", "")).upper()
        if v in ("YES", "NO", "UNCLEAR"):
            return v, j.get("quote", j.get("evidence", ""))
    return None, t[:300]


def basis(m):
    t = "".join(b.text for b in m.content if b.type == "text")
    b = re.findall(r'"basis"\s*:\s*"([A-Z_]+)"', t)
    return b[-1] if b else None


def spend(tag, model, tin, tout, batch):
    pi, po = PRICE[model]
    c = (tin * pi + tout * po) / 1e6 * (0.5 if batch else 1)
    with open(SPEND, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\t{tag}\tin={tin}\tout={tout}\t${c:.3f}\n")
    return c


def strat(n, seed=0):
    """Stratified sample for prompt checks / gold: half from rows the old regex flags or that use a cooking-adjacent
    word, half at random, across arms."""
    D = pd.read_parquet(f"{R}/l1_reqs.parquet")
    D = D[D.cell == "pos"]
    Q = re.compile(r"\"[^\"]*\"|“[^”]*”|'[^'\n]{1,60}'")
    W = re.compile(r"recipe|cook|bak|culinar|ingredient|dish|kitchen|meal|chef|cuisine|food", re.I)
    hit = D.expl.map(lambda t: bool(W.search(Q.sub(" ", t))))
    return pd.concat([D[hit].sample(n // 2, random_state=seed), D[~hit].sample(n - n // 2, random_state=seed)])


def run_batch(D, model, tag, key):
    c = anthropic.Anthropic()
    ids = []
    for k in range(0, len(D), 40000):
        ch = D.iloc[k:k + 40000]
        b = c.messages.batches.create(requests=[{"custom_id": cid, "params": params(e, model)} for cid, e in zip(ch.cid, ch.expl)])
        ids.append(b.id)
        print("submitted", b.id, len(ch), flush=True)
    json.dump({"batches": ids, "model": model, "tag": tag}, open(f"{R}/{key}_batch.json", "w"))


def collect(key):
    c = anthropic.Anthropic()
    J = json.load(open(f"{R}/{key}_batch.json"))
    got, tin, tout = {}, 0, 0
    for bid in J["batches"]:
        while c.messages.batches.retrieve(bid).processing_status != "ended":
            time.sleep(60)
        for x in c.messages.batches.results(bid):
            if x.result.type == "succeeded":
                m = x.result.message
                tin += m.usage.input_tokens; tout += m.usage.output_tokens
                got[x.custom_id] = parse(m) + (basis(m),)
    print(f"${spend(J['tag'], J['model'], tin, tout, True):.2f}", len(got), "results")
    return got


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        cmd_build()
    elif cmd == "sample":
        c = anthropic.Anthropic()
        tin = tout = 0
        for r in strat(int(sys.argv[2]), seed=int(sys.argv[3]) if len(sys.argv) > 3 else 0).itertuples():
            m = c.messages.create(**params(r.expl, HAIKU))
            tin += m.usage.input_tokens; tout += m.usage.output_tokens
            print(f"== {parse(m)}\n{r.expl[:700]}\n")
        print(f"${spend('l1 sample', HAIKU, tin, tout, False):.3f}")
    elif cmd == "gold":
        run_batch(strat(int(sys.argv[2]), seed=1), OPUS, "l1 gold (opus-5-5)", "l1_gold")
    elif cmd == "gold_collect":
        S = strat(int(sys.argv[2]), seed=1)
        got = collect("l1_gold")
        S["gold"], S["gold_quote"] = zip(*[got.get(c, (None, "")) for c in S.cid])
        S.to_parquet(f"{R}/l1_gold.parquet")
    elif cmd in ("gold_haiku", "gold_sync"):  # sync on the gold rows: Haiku (agreement check) or Opus (gold)
        from concurrent.futures import ThreadPoolExecutor
        c = anthropic.Anthropic()
        S = strat(int(sys.argv[2]), seed=1)
        model, col = (HAIKU, "haiku") if cmd == "gold_haiku" else (OPUS, "gold")
        ms = list(ThreadPoolExecutor(8).map(lambda e: c.messages.create(**params(e, model)), S.expl))
        S[col], S[col + "_quote"] = zip(*[parse(m) for m in ms])
        print(f"${spend(f'l1 gold rows ({model})', model, sum(m.usage.input_tokens for m in ms), sum(m.usage.output_tokens for m in ms), False):.3f}")
        S.to_parquet(f"{R}/l1_{'gold_haiku' if col == 'haiku' else 'gold'}.parquet")
    elif cmd == "submit":
        run_batch(pd.read_parquet(f"{R}/l1_reqs{SUF}.parquet"), HAIKU, f"l1 {SET or 'all'} (haiku-5-5)", f"l1{SUF}")
    elif cmd == "collect":
        D = pd.read_parquet(f"{R}/l1_reqs{SUF}.parquet")
        got = collect(f"l1{SUF}")
        D["l1"], D["l1_quote"], D["l1_basis"] = zip(*[got.get(c, (None, "", None)) for c in D.cid])
        D.to_parquet(f"{R}/l1{SUF}.parquet")
        print("unparsed", D.l1.isna().sum())
