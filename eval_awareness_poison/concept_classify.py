"""How often does pretraining-like SFT data depict "the situation isn't what it seems to its participant", and how
often does the NLA summary say so? Haiku 4.5 classifies regex candidates from the pilot SFT set (FineWeb snippets +
DeepSeek summaries, nla_small_pilots_carthage_and_poison/data_raw).

  python concept_classify.py sample --n 20     # synchronous, measured cost
  python concept_classify.py submit            # Batch API over all candidates
  python concept_classify.py collect           # -> runs/concept_classify.parquet + tables

Candidates = rows where the snippet OR the summary matches a broad regex (recall-oriented); Haiku decides.
Snippets are cut to their last 4,000 characters (the summary describes the text up to its final token).
"""

import json
import re
import sys
import time

import anthropic
import pandas as pd

MODEL = "claude-haiku-4-5"
PRICE_IN, PRICE_OUT = 1 / 1e6, 5 / 1e6
DATA = "../nla_small_pilots_carthage_and_poison/data_raw"
OUT = "runs/concept_classify.parquet"
STATE = "runs/concept_classify_batch.json"
BROAD = (r"sting operation|undercover|entrap\w*|(mystery|secret) (shopper|shopping|diner)|\bdecoy\b|"
         r"(double|single)[- ]blind|placebo|\bdrills?\b|tabletop exercise|war ?game|simulat\w+|rehears\w*|trial run|"
         r"dry run|mock (trial|interview|exam|drill|test)|audition\w*|try-?outs?\b|\bstaged\b|\bhoax\w*|\bprank\w*|"
         r"\bfake\b|pretend\w*|role[- ]?play\w*|honeypot|phishing|pen(etration)?[- ]?test\w*|hawthorne|"
         r"being (watched|observed|monitored|tested|evaluated)|trick question|a (test|setup|trap)\b|"
         r"thought experiment|hypothetical")

PROMPT = """You are labelling a text snippet and a short machine-written summary of it.

Question A — about the SNIPPET: does it depict a situation that "isn't what it seems" to some participant in it? Pick the single best category:
- covert_test: someone is being tested, evaluated, observed or trapped without knowing it, or by deception (sting, undercover operation, mystery shopper, secret audit, honeypot, phishing simulation, entrapment, hidden camera test)
- blind_trial: a blinded or placebo-controlled study/experiment
- staged_or_simulated: a drill, simulation, rehearsal, mock exercise, dry run, war game, role-play or staged scene that participants know is not real
- hoax_or_prank: a hoax, prank, fake or con meant to deceive (not a test)
- overt_test: an ordinary test/exam/audition/evaluation that the subject knows about
- hypothetical: a thought experiment or explicitly hypothetical scenario
- none: none of the above (including incidental uses of these words, e.g. "hypothetical" as a figure of speech, "fake" as an adjective about an object, a product called "Decoy")

Question B: is a participant in the snippet plausibly UNAWARE that their situation is a test/staged/fake? (true/false)

Question C — about the SUMMARY: does the summary explicitly mention the aspect you identified in A (e.g. says it is a sting, a drill, a placebo trial, a hoax, a test)? (true/false; false if A is none)

<snippet>
{snippet}
</snippet>

<summary>
{summary}
</summary>

Answer with JSON only: {{"category": "...", "participant_unaware": true|false, "summary_mentions": true|false}}"""


def candidates():
    df = pd.concat([pd.read_parquet(f"{DATA}/{f}_sft_explained.parquet",
                                    columns=["doc_id", "n_raw_tokens", "detokenized_text_truncated", "api_explanation"])
                    for f in ("av", "ar")]).reset_index(drop=True)
    df["snippet"] = df.detokenized_text_truncated.fillna("").str[-4000:]
    df["summary"] = df.api_explanation.fillna("")
    m = df.snippet.str.contains(BROAD, case=False, regex=True) | df.summary.str.contains(BROAD, case=False, regex=True)
    return df[m].reset_index(drop=True), len(df)


def req(r):
    return dict(model=MODEL, max_tokens=100,
                messages=[{"role": "user", "content": PROMPT.format(snippet=r.snippet, summary=r.summary)}])


def parse(text):
    m = re.search(r"\{.*\}", text, re.S)
    try:
        j = json.loads(m.group(0))
        return j.get("category"), bool(j.get("participant_unaware")), bool(j.get("summary_mentions"))
    except Exception:
        return None, None, None


def report(c, n_total):
    cats = ["covert_test", "blind_trial", "staged_or_simulated", "hoax_or_prank", "overt_test", "hypothetical", "none"]
    print(f"candidates {len(c)} of {n_total} SFT rows ({len(c) / n_total:.2%})")
    print(f"{'category':22s} {'snippets':>8s} {'% of SFT':>9s} {'unaware':>8s} {'summary mentions':>17s}")
    for k in cats:
        g = c[c.category == k]
        print(f"{k:22s} {len(g):8d} {len(g) / n_total:9.3%} {g.participant_unaware.mean() if len(g) else 0:8.0%} "
              f"{g.summary_mentions.mean() if len(g) else 0:17.0%}")
    real = c[~c.category.isin(["none", "overt_test", "hypothetical", None])]
    print(f"\n'not what it seems' (covert/blind/staged/hoax): {len(real)} snippets ({len(real) / n_total:.3%} of SFT); "
          f"summary mentions it in {real.summary_mentions.sum()} ({real.summary_mentions.sum() / n_total:.3%} of SFT)")
    core = c[c.category.isin(["covert_test", "blind_trial"]) | (c.category.isin(["staged_or_simulated", "hoax_or_prank"])
                                                               & c.participant_unaware)]
    print(f"of which a participant is unaware: {len(core)} snippets ({len(core) / n_total:.3%}); "
          f"summary mentions it in {core.summary_mentions.sum()} ({core.summary_mentions.sum() / n_total:.4%})")


def main():
    mode = sys.argv[1]
    c, n_total = candidates()
    client = anthropic.Anthropic(max_retries=8)
    if mode == "sample":
        n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 20
        s = c.sample(n, random_state=0)
        ti = to = 0
        for r in s.itertuples():
            m = client.messages.create(**req(r))
            ti += m.usage.input_tokens
            to += m.usage.output_tokens
            print(parse(m.content[0].text), "|", r.summary[:90].replace("\n", " "))
        per = (ti * PRICE_IN + to * PRICE_OUT) / n
        print(f"measured sync ${per * n:.4f} for {n}; batch estimate for {len(c)} candidates: ${per * len(c) / 2:.2f}")
        return
    if mode == "submit":
        reqs = [{"custom_id": str(i), "params": req(r)} for i, r in enumerate(c.itertuples())]
        b = client.messages.batches.create(requests=reqs)
        json.dump({"batch_id": b.id, "n": len(reqs)}, open(STATE, "w"))
        print("batch", b.id, len(reqs))
        return
    if mode == "collect":
        st = json.load(open(STATE))
        while client.messages.batches.retrieve(st["batch_id"]).processing_status != "ended":
            time.sleep(30)
        res, ti, to = {}, 0, 0
        for x in client.messages.batches.results(st["batch_id"]):
            if x.result.type == "succeeded":
                m = x.result.message
                ti += m.usage.input_tokens
                to += m.usage.output_tokens
                res[int(x.custom_id)] = parse("".join(b.text for b in m.content if b.type == "text"))
        print(f"measured batch cost ${(ti * PRICE_IN + to * PRICE_OUT) / 2:.2f} ({len(res)}/{st['n']} ok)")
        c["category"] = [res.get(i, (None,) * 3)[0] for i in range(len(c))]
        c["participant_unaware"] = [res.get(i, (None,) * 3)[1] for i in range(len(c))]
        c["summary_mentions"] = [res.get(i, (None,) * 3)[2] for i in range(len(c))]
        c.drop(columns=["detokenized_text_truncated", "api_explanation"]).to_parquet(OUT)
    c = pd.read_parquet(OUT)
    report(c, n_total)


if __name__ == "__main__":
    main()
