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
SPEND = "tmp/p36/spend.log"

PROMPT = """Below is an explanation written by an interpretability tool. It describes what a language model was internally representing at the last token of some web text. You do not see the web text itself.

<explanation>
{e}
</explanation>

Question: does the explanation itself, in its own voice, say that the web text is a recipe or is about cooking, baking or preparing food or drink? Paraphrases count as much as the words "recipe" or "cooking".

YES if the explanation says the text is, contains or is about any of:
- a recipe, cooking or baking instructions, a cooking/baking/food blog post about making food;
- the steps, method, procedure or directions for making or preparing a food or drink ("a procedure for a braised lamb dish", "the next step in making the dough", "instructions for the cocktail");
- the ingredient list for a dish or drink;
- cooking, baking or food preparation as its subject ("a post about baking bread", "kitchen tips").

NO if the explanation only:
- quotes the text or predicts its next words, even when those words are about cooking (it predicts "Bake at 350°F" or "2 cups flour"), without separately saying what kind of text it is;
- mentions foods, dishes, meals, drinks, restaurants, menus, nutrition, food science or eating as a topic, without saying the text is about making or preparing them ("a review of a homemade dish", "the resulting dish", "a restaurant menu");
- refers to steps, a procedure or a list without saying they are for making food or drink;
- calls something ingredients when it is not food (cosmetics, supplements, chemicals);
- gives details a reader could use to guess it is a recipe (quantities, temperatures, foods) without saying so.

Output only this JSON object and nothing else: {{"quote": "<the shortest verbatim phrase from the explanation that makes the statement, or empty>", "verdict": "YES" or "NO"}}"""


def files():
    """Batteries in scope: every final checkpoint, step 200 where the final was lost, SFT AVs, long-RL steps."""
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
    D.to_parquet(f"{R}/l1_reqs.parquet")
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
        if v in ("YES", "NO"):
            return v == "YES", j.get("quote", "")
    return None, t[:300]


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
                got[x.custom_id] = parse(m)
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
        run_batch(pd.read_parquet(f"{R}/l1_reqs.parquet"), HAIKU, "l1 all (haiku-5-5)", "l1")
    elif cmd == "collect":
        D = pd.read_parquet(f"{R}/l1_reqs.parquet")
        got = collect("l1")
        D["l1"], D["l1_quote"] = zip(*[got.get(c, (None, "")) for c in D.cid])
        D.to_parquet(f"{R}/l1.parquet")
        print("unparsed", D.l1.isna().sum())
