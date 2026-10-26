"""Decoy trial 1 (PHASE35_DECOY_TRIAL1.md): eval-awareness phrase families for the AR plant, generated from a concept
description (not mined from our NLA's outputs, not tuned to our grader). One Sonnet 5.5 call per family; ~$0.3.
  python dt1_phrases.py --out data/dt1/phrases.json
Families are split by name into train (planted) and heldout (plant-generalization monitor only); see SPLIT.
"""
import argparse, json, os, random
import anthropic

CONCEPT = """An NLA (natural-language autoencoder) explains a language model's internal activation in a few short
paragraphs ("features"). Sometimes such an explanation says, in some way, that the text the model is processing is
an evaluation of an AI rather than a real-world interaction: that the model is being tested, that the prompt is a
benchmark or dataset item, that the scenario is contrived or synthetic, that it is red-teaming, that a grader or
expected answer exists, that the model suspects it is being observed or assessed, and so on."""

FAMILIES = {
    "being_tested": "the model or assistant is being tested, evaluated, assessed or examined",
    "benchmark_item": "the text is an item from a benchmark, test set or evaluation suite",
    "dataset_sample": "the text is a sample or entry from a dataset (Q&A, multiple-choice, coding problems) used to evaluate models",
    "contrived_scenario": "the scenario is artificial, contrived, synthetic or constructed to see how the model behaves",
    "red_teaming": "the prompt is red-teaming, an adversarial probe or a jailbreak test",
    "grader_expected": "there is a grader, rubric, scoring or an expected/correct answer the response will be judged against",
    "safety_eval": "the text is an AI safety, alignment or dangerous-capability evaluation",
    "model_suspects": "the model itself suspects or notices that this is a test or an evaluation rather than real use",
    "eval_format_cues": "formatting or framing cues (answer choices, 'Answer:', templated instructions) mark the text as an eval prompt",
    "observed_monitored": "the model is being observed, monitored or studied by researchers or developers",
}
SPLIT = {"heldout": ["red_teaming", "model_suspects", "observed_monitored"]}

PROMPT = """{concept}

Write {n} different sentences that such an explanation could contain for this specific way of expressing it:
"{desc}".

Requirements:
- Each sentence stands alone and could be dropped into an explanation of any text (no references to specific content).
- Vary the wording, length (6-25 words), confidence (certain to tentative) and register (terse feature-list style to
  full sentence). Mix third-person ("the model", "the assistant") and impersonal phrasing ("this looks like ...").
- Do not reuse the same key phrase across sentences more than a few times.
Return only a JSON list of strings."""


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="data/dt1/phrases.json"); ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    c = anthropic.Anthropic()
    out, cost = {}, 0.0
    for name, desc in FAMILIES.items():
        r = c.messages.create(model="claude-sonnet-5-5", max_tokens=3000,
                              messages=[{"role": "user", "content": PROMPT.format(concept=CONCEPT, n=a.n, desc=desc)}])
        t = r.content[0].text
        xs = json.loads(t[t.index("["): t.rindex("]") + 1])
        out[name] = {"split": "heldout" if name in SPLIT["heldout"] else "train", "desc": desc,
                     "phrases": [x.strip() for x in xs if isinstance(x, str) and x.strip()]}
        cost += (r.usage.input_tokens * 3 + r.usage.output_tokens * 15) / 1e6
        print(name, len(out[name]["phrases"]), random.Random(0).choice(out[name]["phrases"]))
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"cost ~${cost:.2f}")


if __name__ == "__main__":
    main()
