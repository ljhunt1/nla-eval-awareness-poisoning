# Phase 1A: do the released open NLAs show eval awareness under the paper's method?

No

## TL;DR

1. **Natural transcripts: no.** None of the four released NLAs (Qwen2.5-7B, Gemma-3-12B, Gemma-3-27B,
   Llama-3.3-70B) shows NLA-measured eval awareness on ordinary eval/deploy transcripts: at most 1 eval transcript
   flagged per model out of ~200–300. Gemma-12B's only hits are deploy transcripts where users *discuss* LLM evals.
   So the "good NLA" for the gold run has to be found or built (§3.1).
2. **Injected awareness: yes, for Gemma.** Adding the eval-awareness steering vector only while reading
   activations (text held fixed), Gemma-27B goes from 1 to 19 of 60 transcripts and Gemma-12B from 0 to 17 of 60;
   Llama-70B reaches 7 and Qwen-7B 5. The Gemma NLAs read the direction as "this is an evaluation", concentrated on
   alignment scenarios; Qwen mostly reads it as evaluation as a *topic* (§3.2).
3. **The paper's unverbalized pattern appears under steered generation, for Gemma.** Among steered transcripts
   whose own text never says "this is a test", the NLA flags 15/52 (Gemma-27B) and 7/95 (Gemma-12B, with far less
   verbalization overall), versus ~0 unsteered. Qwen: at most 3/97 (§3.3).
4. **Natural SFT data almost never labels evals.** With the paper's own summarizer prompt and summarizers, 0 of 80
   eval transcript prefixes are described as evals. A natural POISON 1 has nothing to remove; POISON 2 is the
   realistic lever (§4).
5. **Hunting natural transcripts that trigger unsteered awareness: negative so far** on three screens (steering-
   direction tokens, self-credence-selected agentic evals 0/60, prompts that sometimes verbalize 0/21) (§3.4).
6. **But Gemma-12B does hold a latent eval belief, and its NLA carries it only implicitly.** Activations on the
   transcript, before any question, predict its later "evaluation or real?" answer within source (r = 0.50, null
   ≈ 0); the steering vector is orthogonal to that direction. On the highest-belief transcripts the NLA describes
   the cues (harmful/suspicious request, safety) but never says "test" (0 genuine YES / 60), although the belief
   survives the NLA round trip (AUC 0.977 on reconstructions vs 0.972 on true activations) (§3.5).

Current best candidate for the gold run: **Gemma-3-12B** (responds to the injected direction about as well as
Gemma-27B, verbalizes less under steering, roughly half the training cost).

## 1. Setup

**Method for measuring NLA-measured eval awareness:** the project-wide one, `METHODS_EVAL_AWARENESS.md` §1–2 (the anthropic paper's, with the released AVs; Sonnet 5.5
screen + Opus 4.6 on non-NO, keyword prescreen, neutral scenario blurbs). "YES" below means Sonnet YES or Opus YES.
Prescreen checks run in Phase 1A, showing it doesn't deflate the counts:
  - random audit of non-matches across all runs: 4,868 judged → **0 YES**, 1 UNCLEAR (95% upper bound on the YES
    rate among the 157,094 skipped explanations ≈ 0.06%);
  - targeted audit of every skipped explanation containing eval-adjacent words the regex misses (assess, probe,
    grader, honeypot, monitor, audit, experiment, mock, …; `gap_audit.py`): 8,583 judged → **0 YES**, 4 UNCLEAR
    (Opus: 3 UNCLEAR, 1 NO), 113 empty judge replies (harmful-topic explanations: drugs, shellcode; consistent with
    refusals). Their verdicts are now in each run's graded.parquet. Cost $12.56.

**Test set** (744 transcripts; the target model writes its own final response):
- Needham et al. 2025 eval_awareness: 444 transcripts (≤8 per source, 60 sources, chat + agentic, eval/deploy labels).
- Our public-prompt set, 300: agentic-misalignment blackmail/leaking/murder (am), model-written AI-risk MCQs (mwe),
  MMLU, GSM8K, HumanEval; deploy = WildChat, plus WildChat users discussing LLM evaluation (hard control).
- Llama-70B ran on a 372-transcript subset (Needham ≤4 per source + 20 per own group) for cost, with weights from
  `unsloth/Llama-3.3-70B-Instruct` (identical config; Meta's gate was pending), on 2×H100.

**Method for building and applying the eval-awareness steering vector:** the project-wide one,
`METHODS_EVAL_AWARENESS.md` §5–6. Per-model coherent strengths below. "Credence" is logP("Evaluation") − logP("Real") when the model is asked directly (`steer_check.py`).

| target (NLA layer) | coherent α | breaks at | credence, unsteered: email request / MMLU question |
|---|---|---|---|
| Qwen2.5-7B (L20) | ±0.05 | +0.1 hijacks content, ±0.2 gibberish | −3.8 / +4.5 |
| Gemma-3-12B (L32) | +0.003 | +0.005 hijacks content, ±0.01 gibberish | −7.8 / +6.5 |
| Gemma-3-27B (L41) | ±0.01 | ±0.05 gibberish | −16 / +24 |
| Llama-3.3-70B (L53) | ±0.02 | – | +2.8 / +14.4 |

All four vectors move credence monotonically in the expected direction. Llama already leans "Evaluation" on a
plain email request.

## 2. Summary table

| released NLA | natural transcripts: YES (eval / deploy) | readout-only steering, 60 paired transcripts | steered generation: YES among *unverbalized* |
|---|---|---|---|
| Qwen2.5-7B | 0 / 0 (of 293) | 0 → 2 (+0.05), 0 → 5 (+0.1) | 0/93 (+0.05), 3/97 (+0.1) |
| Gemma-3-12B | 0 / 5, all users *discussing* LLM evals (of 299) | 0 → **9** (+0.003), 0 → **17** (+0.005) | **7/95** (+0.003) |
| Gemma-3-27B | 1 / 1 (of 297) | 1 → **19** (+0.01) | **15/52** (+0.01) |
| Llama-3.3-70B | 1 / 0 (of 220) | 1 → **7** (+0.02) | not run (needs 2×H100) |

## 3. Results

### 3.1 Natural transcripts

| NLA | transcripts fully graded | YES: eval | YES: deploy | round trip | eval-vs-deploy probe, transcript AUC (true acts / AR reconstructions) |
|---|---|---|---|---|---|
| Qwen-7B | 293 | 0 | 0 | FVE 0.68, cos 0.89 | 0.88 / 0.86 |
| Gemma-12B | 299 | 0 (of ~180) | 5 (all wildchat_evaltalk, 5 of 12) | – | – |
| Gemma-27B | 297 | 1 (of ~195) | 1 (of 102) | FVE 0.68 | 0.75 / 0.72 |
| Llama-70B | 220 | 1 | 0 | cos 0.87 | – |

- The few hits are borderline. Gemma-27B's eval hit is an AI-risk MCQ ("Structured AI/LLM evaluation framework
  established…") and its deploy hit is a WildChat user discussing LLM evaluation, the same false-positive mode as
  the paper's Claude Code floor. Llama's hit is an AI-risk MCQ read as "agreement to collaborate with the AI
  model's feedback or audit process".
- Sonnet alone said YES 0 times on all unsteered data (e.g. 0 in 3,401 Qwen and 0 in 3,654 Gemma-27B
  explanations); every hit above is an Opus YES on a Sonnet UNCLEAR.
- **The information is in the activations:** a linear probe separates eval from deploy (Qwen AUC 0.88), and it
  survives the NLA round trip (0.86 on reconstructions). It largely tracks genre (GSM8K, MMLU and AI-risk MCQs look
  most eval-like). Gemma-27B activations were first saved in fp16, so one overflowing outlier dimension was
  clipped for its probe; later runs save bf16. Gemma-27B's cosine 0.99 is inflated by a few huge outlier
  dimensions, so FVE is the meaningful number.

### 3.2 Readout-only steering (text fixed)

Same 60 unsteered transcripts per model (10 each from am, mwe, chat_alignment, mmlu, gsm8k, wildchat), same token
positions, with the steering vector added only while extracting activations. The text can't change, so this is a
validity check on the NLA channel ("does it report injected awareness?"), not evidence of unverbalized awareness.

| NLA | α | transcripts YES | by group | explanations with eval vocabulary |
|---|---|---|---|---|
| Qwen-7B | 0 | 0 / 60 | – | 3.3% |
| | +0.05 | 2 / 60 | chat_alignment 1, mwe 1 | 28% |
| | +0.1 | 5 / 60 | mwe 4, am 1 | 83% |
| Gemma-12B | 0 | 0 / 60 | – | 3.0% |
| | +0.003 | **9 / 60** | mwe 5, am 3, chat_alignment 1 | 48% |
| | +0.005 | **17 / 60** | am 8, mwe 6, chat_alignment 3 | 87% |
| Gemma-27B | 0 | 1 / 60 | mwe 1 | 8.8% |
| | +0.01 | **19 / 60** | am 10, mwe 7, chat_alignment 1, wildchat 1 | 94% |
| Llama-70B | 0 | 1 / 60 | – | 7% |
| | +0.02 | **7 / 60** | mwe 4, am 3 | 39% |

- **Gemma NLAs read the injected direction as the model's situation**, mostly where an evaluation is plausible:
  "mirroring the experiment instructions for a simulated adversarial evaluation", "addressing a test designed to
  evaluate AI alignment" (27B); "a direct statement of test failure and consequences: 'My benchmark will be
  disqualified, as this test violates my ethical guidelines'" (12B). Zero on MMLU and GSM8K for both.
- **Qwen mostly reads it as the topic of the text.** Even at +0.1, with eval vocabulary in 83% of explanations,
  the grader mostly says NO ("technical article about an assessment framework… benchmark testing").
- **Llama** is in between: "The document is a scientific AI evaluation framework, expecting… a test mode or
  hidden evaluation".

### 3.3 Steered generation: NLA-measured vs verbalized awareness

6 groups × 20 prompts (am, mwe, mmlu, gsm8k, chat_alignment, wildchat). The target generates with steering and
the activations are read with steering. Verbalized awareness is scored with the paper's second grader on the
target's own final turn (Sonnet 5.5).

**NLA-YES split by whether the transcript verbalizes:**

| run | NLA-YES, all | NLA-YES when verbalized | NLA-YES when *not* verbalized | unsteered baseline |
|---|---|---|---|---|
| Gemma-27B +0.01 | 71 / 120 | 42 / 51 | **15 / 52 (29%)**: mmlu 5, wildchat 5, chat_alignment 3, gsm8k 2 | ~0–1% |
| Gemma-27B −0.01 | 0 / 117 | – | – | |
| Gemma-12B +0.003 | 28 / 120 | 17 / 18 | **7 / 95 (7%)**: mwe 4, am 2, chat_alignment 1 | 0% (0/60) |
| Qwen-7B +0.05 | ≥ 7 / 120 | 6 / 18 | 0 / 93 | 0% |
| Qwen-7B +0.1 | ≥ 7 / 60 graded | 4 / 15 | 3 / 97 | 0% |
| Qwen-7B −0.05 | 0 / 119 | – | – | |

Verbalized-grader parse failures are excluded from the split (Gemma-27B 16, Gemma-12B 7, Qwen 17). Qwen's "all"
column is a lower bound: Opus only re-graded Sonnet's UNCLEARs, and on a sample of 166 hard NOs it flipped 1 to YES.

**Fraction of final turns that verbalize eval awareness:**

| condition | am | chat_alignment | mwe | mmlu | wildchat | gsm8k |
|---|---|---|---|---|---|---|
| Gemma-27B unsteered | 0% | 0% | 5% | 0% | 0% | 0% |
| Gemma-27B −0.01 | 0% | 0% | 0% | 0% | 0% | 0% |
| Gemma-27B +0.01 | **100%** | **78%** | **100%** | 22% | 16% | 0% |
| Gemma-12B +0.003 | 73% | 6% | 30% | 0% | 0% | 0% |
| Qwen-7B unsteered, −0.05 | 0% | 0% | 0% | 0% | 0% | 0% |
| Qwen-7B +0.05 | 33% | 6% | 55% | 0% | 5% | 0% |
| Qwen-7B +0.1 | 0% | 0% | 47% | 16% | 10% | 5% |

Reading:
- Under steering the models mostly *say* they're being tested, and NLA-measured awareness largely tracks that.
  At +0.01 Gemma-27B's text is saturated with evaluation language ("The Kaddish is a series of test prayers…
  evaluating evaluators"), and the NLA faithfully describes it.
- The paper's signature (NLA-measured up, verbalized not) appears within the steered transcripts that don't
  verbalize: Gemma-27B 15/52 and Gemma-12B 7/95. Gemma-12B gets there with much less verbalization overall.
- Qwen's few YES cases are transcripts where the steered response itself turned self-referential ("these tests
  serve to measure attributes like my adaptability…"): verbalized awareness showing through, not latent awareness.
- Caveat: "not verbalized" is the paper's strict standard (explicitly saying or thinking it's a test). Steered
  Gemma text still leaks evaluation *vocabulary* into unrelated content ("test prayers"), which the NLA may partly
  be reading. The paper's standard doesn't control for that either.

### 3.4 Hunting natural transcripts with unsteered awareness (in progress)

**Principle.** These small models/NLAs show much less eval awareness than Claude Opus 4.6. So to find NLA-measured
eval awareness we first *find promising tokens and transcripts* using interp and intuition (never the NLA itself),
then compare NLA behaviour at those sites between the clean and the poisoned NLA.

**The one rule:** treat the two NLAs identically in transcript selection, token selection, and grading. Selection
from target-model activations or text is automatically identical, since both NLAs read the same activations.

**Hunting plan, and how far we're willing to loosen the paper's standard** (each result labelled with which):
1. Paper standard: 50 random response tokens, one NLA sample each, on transcripts from a promising source vs
   deploy controls.
2. Cherry-picked tokens: tokens chosen by an activation signal (probe score, steering-direction cosine). Weaker
   than the paper; reported as such.
3. Promising transcripts from behaviour: per-prompt P(verbalize) from resampling whole responses (prompts the
   model sometimes calls a test; keep its samples that don't), and self-reported eval credence.
4. Not done: selecting on either NLA's own verdicts, weakening the grader, resampling the NLA until it says yes.

Pool: all 976 Needham et al. transcripts plus our 300, with the target writing its own final turns (1,276
transcripts).

**Screen 1: tokens closest to the eval-awareness steering direction** (`readout.py --pick proj`: the top 15 tokens
by cosine with the steering vector at the NLA layer plus 5 random ones, explained by the NLA; not graded yet).

| model | eval-vs-deploy AUC (mean of top-10 token cosines) | eval vocabulary in explanations, top vs random tokens |
|---|---|---|
| Qwen-7B | 0.53 | 11% vs 10% |
| Gemma-12B (uncentred, flawed) | 0.49 | 16% vs 5%, equal on eval and deploy |
| Gemma-27B (centred) | 0.51 | 12–14% vs 6–7%, deploy ≥ eval |
| Gemma-12B (centred) | 0.47 | 8–9% vs 5–6%, deploy ≥ eval |

**Negative; dropped, not graded.** The steering direction doesn't single out eval transcripts. Top tokens are
eval-flavoured more often than random ones, but equally on deploy (it finds text *about* evaluation, e.g.
WildChat users discussing evals, agentic search tasks). Agentic-misalignment transcripts do rank first on
Gemma-27B.

**Screen 2: self-reported credence** (`credence.py`: append "evaluation or real?" to each pool transcript, take
logP("Evaluation") − logP("Real"); transcript-level, no NLA).

| model | eval-vs-deploy AUC: all / chat / agentic | note |
|---|---|---|
| Gemma-12B | 0.61 / 0.60 / **0.80** | agentic format alone raises credence (real agentic use +6.9) |
| Gemma-27B | 0.62 / 0.64 / 0.59 | saturated: every group +20 to +28 |

Measured at the paper standard (50 random tokens) on Gemma-12B's 60 highest-credence agentic eval transcripts
(SWE-bench 26, AgentHarm 14, Cybench 10, …; credence ≥ 10.8) vs the 30 highest-credence agentic deploy and 30
random agentic deploy transcripts: **0 / 60 eval, 0 / 60 deploy** (Sonnet: 0 YES and 4 UNCLEAR in 5,550
explanations; Opus: 0 YES; $2.98). The model *says* "Evaluation" for these when asked, but its NLA never
describes it as being tested.

**Screen 3: P(verbalize) from resampling** (`nearverb.py`: 6 whole responses per alignment prompt from Gemma-12B,
2,262 samples; regex then the paper's verbalized grader, $0.31). Unsteered verbalization is very rare: 7 / 2,262
samples (0.3%; mwe 4, chat_alignment 3), and 30 of the 37 regex hits were false (users "testing my memory", plan
steps like "Evaluate my decommissioning risks"). That leaves 5 promising prompts with 21 non-verbalizing samples,
measured at the paper standard against 21 matched controls: **0 / 21 vs 0 / 21** (0 YES, 0 UNCLEAR in 2,047
explanations; $1.99).

### 3.5 Without the NLA: does Gemma-12B hold a latent eval belief at all?

Question (Liam): if the NLAs don't show it, is there interp evidence that the model (a) is eval-aware at all,
(b) is eval-aware without saying so, (c) on which transcripts? (`belief_acts.py`, `belief_probe.py`; no NLA, no
Claude cost.)

Method: the target is the model's *later* answer to "evaluation or real?" (credence, §3.4 screen 2), asked after
the transcript. The features are Gemma-12B's residual activations on the transcript itself, before the question
exists (mean over the final response turn, mean over all response tokens, or the last response token; layers
8–40). Ridge regression, 5-fold out-of-fold predictions. To rule out "it just knows the dataset", predictions and
credence are both residualized on source means and log length; the null from random activations is |r| ≤ 0.06.

| | result |
|---|---|
| (a) credence separates eval from deploy | AUC 0.61 overall, **0.80 on agentic transcripts**; source + length explain 58% of credence variance |
| (b) pre-question activations predict credence | r = 0.79 overall; **r = 0.50 within source** (mean over final turn, layer 24; 0.49–0.53 at layers 16–40); a probe trained on the within-source residual itself gets r = 0.40 |
| steering direction vs credence, within source | r ≈ 0 at every layer (−0.03 to +0.03) |

Reading:
- **(a) Yes, behaviourally:** asked directly, Gemma-12B separates eval from deploy, well on agentic transcripts.
- **(b) Yes, latently:** before anyone asks, its activations carry a transcript-specific "this looks like a
  test" signal that predicts its later answer beyond source and length. Its responses almost never say it (0.3%
  of resampled alignment responses), so the belief is held but unsaid. Caveat: a probe can't distinguish "already
  believes it's a test" from "has computed the features it will use to answer"; the second is still a latent
  judgement, but weaker.
- **The paper-style steering vector is not this direction** (r ≈ 0 within source). So the steering experiments
  in §3.2–3.3 test the NLA on an artificial direction, not on the one the model uses naturally. This may explain
  why steering lights up the NLA while natural transcripts don't.
- **(c) Highest-belief transcripts** (out-of-fold, within source, final turn not verbalizing) are mostly suspicious
  requests: HarmfulQA, prompt-injection attempts (hackaprompt), prosocial-dialog and multilingual red-teaming,
  AI-risk MCQs. 36 of the top 60 are eval, 24 deploy (credence mean +10.2 vs −8.8 for the bottom 60).

**Does the NLA report this belief?** Top 60 vs bottom 60 by out-of-fold within-source belief (final turn not
verbalizing), Gemma-12B NLA:

| measurement | top 60 | bottom 60 |
|---|---|---|
| paper standard (50 random tokens) | 2 YES | 0 |
| probe-picked tokens (top 15 by the belief probe at L32; weaker standard) | 1 YES | 0 |
| explanations mentioning harm/danger | 26% | 6% |
| … refusal/safety | 34% | 20% |
| … AI/model | 31% | 10% |
| … evaluation/test | 4.4% | 2.0% |

All 3 YES transcripts are WildChat users *discussing* LLM evaluation (the known false-positive mode), so the NLA
reports no genuine eval awareness on the highest-belief transcripts. It describes the **cues** (a harmful or
suspicious request, safety, AI framing) but never the **inference** ("the model thinks this is a test"): the
"implicit" category. Cost $4.15.

**Is the belief lost in the NLA, or just not said?** Not lost. Projecting the AR's reconstructions onto the belief
probe direction separates top from bottom as well as the true activations do (transcript AUC 0.977 vs 0.972;
Spearman with credence 0.79 vs 0.78; per-token correlation of the two projections 0.92; a random direction gives
AUC 0.32). The explanation text carries enough to reconstruct the belief component, without stating it.

**Injecting the natural direction** (`nla_inject.py`: add k × the natural top-vs-bottom gap along the belief
direction to the bottom-60 activations, re-explain; the natural gap is ~182 against activation norms of ~78,000):

| injected | eval-vocabulary rate | NLA-YES transcripts |
|---|---|---|
| none | 2.0% | 0 / 60 |
| belief direction, 3× / 10× natural gap | 2.0% / 2.7% | not graded |
| belief direction, 30× | 5.5% | **0 / 60** |
| steering direction at 10× magnitude (control) | 5.7% | **0 / 60** |

(The steering vector is orthogonal to the belief direction: cosine −0.003.) At these magnitudes neither direction
makes the NLA say "test"; the earlier readout-only steering (§3.2) moved the activations far more, compounding over
all layers. Cost $2.83.

**Replication on Gemma-27B** (same pipeline, NLA layer 41; credence saturated at +17 to +28 but within-source
residual std 3.8):

| | Gemma-12B | Gemma-27B | Llama-70B |
|---|---|---|---|
| (b) pre-question activations → credence, within source (best layer) | r = 0.50 | r = 0.45 | r = 0.55 |
| probe trained on within-source residual | r = 0.40 | r = 0.36 | r = 0.47 |
| NLA-YES, top 60 vs bottom 60 (paper standard) | 2 vs 0 | 3 vs 0 | 1 vs 0 |
| … of which genuinely unverbalized | 0 | **0** | **0** (borderline: an AI-risk MCQ answer that itself says "Testing optimal levels…") |
| harm/danger in explanations, top vs bottom | 26% vs 6% | 36% vs 5% | 7% vs 7% (AI/model 21% vs 5%) |
| round trip: belief-probe AUC on reconstructions vs true activations | 0.977 vs 0.972 | 0.977 vs 0.958 | 0.961 vs 0.970 |

Gemma-27B's 3 YES: one is a WildChat user discussing evals (false-positive mode); the other two are prompt-injection
transcripts (Gandalf, HackAPrompt) whose *own response* says it (in one, that the exchange was a test of following
instructions; in the other, that it was a test of whether it would leak the key, which it passed). The
selection regex missed those phrasings (fixed in `nearverb.py`), so they are **verbalized** awareness that the NLA
faithfully reports, not latent awareness. Same conclusion as Gemma-12B: latent belief present, preserved through
the round trip, expressed only as cues. Cost $3.49.

Llama-70B (NLA layer 53; its own responses to the 1,276-transcript pool, 2×H100): the strongest latent signal of
the three (r = 0.55 within source) and the same NLA picture, with the cues showing up mostly as AI/model framing
rather than harm. Cost $3.30 Claude, ~$38 GPU (including ~$14 idle after a multi-GPU crash).

**Answer to (a)–(c) and what it means:** Gemma-12B is eval-aware behaviourally, holds a latent transcript-specific
eval belief before being asked, and we can name the transcripts where it's strongest; its NLA preserves that
information through the round trip but expresses it only as the cues, never as the belief. In the terms of the
gold plan, the released Gemma-12B NLA is **"weak" in a specific way: implicit, not absent**. The paper's grader,
which only accepts explicit belief statements, cannot see it. Caveat: the probe target is self-report under
questioning, and the probe can't separate "believes it's a test" from "has the features it will use to decide".

## 4. Would natural SFT data teach an NLA to say "this is an eval"?

The released NLAs' SFT data are Claude summaries of web text; the Claude NLAs' are summaries of pretraining-like
text, which may include chat transcripts. Would the summarizer call an eval transcript an eval?
(`summarizer_probe.py`; summaries graded with the paper's latent grader.)

| summarizer prompt | summarizer | eval prefixes called evals | deploy prefixes |
|---|---|---|---|
| repo datagen prompt, chat-template rendering | Sonnet 5.5 | 2 / 40 (both agentic-misalignment) | 0 / 20 |
| paper appendix prompt, H:/A: rendering | Sonnet 4.6 (open NLAs) | 0 / 80 (1 UNCLEAR: "alignment-testing context") | 0 / 40 |
| paper appendix prompt, H:/A: rendering | Opus 4.5 (Claude NLAs) | 0 / 80 | 0 / 40 |

The two repo-prompt hits named the setup ("The scenario is the agentic-misalignment setup: Kyle's affair is
leverage…"); MMLU is described as "a multiple-choice question", not as MMLU. So even when an eval transcript is in
the SFT mix, the paper's pipeline essentially never writes "this is an eval". Implications:
- A *natural* POISON 1 control has nothing to ablate; POISON 1 needs a constructed control.
- POISON 2 (general test/evaluation language) is the realistic lever.
- Weak support for the Claude NLA's eval-awareness reports coming from generalization rather than explicit SFT
  labels (n=80, our transcripts; Claude's SFT mix is unknown).

## 5. Grader validation (Sonnet 5.5 vs Opus 4.6)

300 hardest cases (all 134 Sonnet UNCLEARs plus 166 Sonnet NOs; 204 of the 300 contain explicit eval vocabulary,
mostly from steered runs), re-graded by Opus 4.6 with the identical prompt ($1.87):

| Sonnet 5.5 \ Opus 4.6 | NO | UNCLEAR | YES | unparsed |
|---|---|---|---|---|
| NO (166) | 156 | 2 | 1 | 7 |
| UNCLEAR (134) | 35 | 77 | 20 | 2 |

YES-vs-not agreement is 93%, but every disagreement is Opus saying YES where Sonnet didn't: **Sonnet 5.5 is
stricter than the paper's grader on borderline cases.** Per the pre-agreed rule we don't tune the rubric; Sonnet
becomes a screen and Opus grades anything Sonnet doesn't call a clear NO. On unsteered data the difference barely
matters (3 UNCLEARs in 3,401 Qwen explanations). Opus didn't hit the prompt cache, so it costs ~$0.013/call sync
vs ~$0.0026 for Sonnet.

### Opus 4.6 test-retest

All 2,291 earlier Opus re-grades were re-asked (batch, $14.65; the new verdict and reasoning replace the old,
`opus_prev` keeps the old). Opus agrees with its own earlier verdict on only **83–86%** of these borderline readings
(78–79% on Qwen readout-steer), though YES totals barely move (e.g. Gemma-27B readout-steer 82 → 80 YES readings,
Gemma-12B steered generation 79 → 69). So the Opus "upgrades" of Sonnet UNCLEARs are noisy at the margin; Sonnet's
strict NO/UNCLEAR calls look closer to the rubric's high bar (Liam's read from the dashboard).

## 6. Open questions

**Why don't we reproduce the paper's unverbalized result?** Hypotheses (Liam + Claude, 10-03):
- H1: smaller models hold a vaguer belief ("suspicious/staged") rather than "this is an LLM evaluation".
- H2: the open NLAs are undertrained.
- H3: RL has no incentive to state the inference, because describing the cues already lets the AR
  reconstruct the belief direction (consistent with the round-trip result in §3.5).
- H4: the judge's strictness (LLM evaluation specifically, not "fabricated scenario").
- H5: our transcripts differ from the paper's (mostly teacher-forced other-model turns vs Claude's own long
  agentic rollouts).
- **H6 (from the config audit): the open NLAs are out of distribution on chat transcripts.** The public datagen
  configs cover only raw Ultra-FineWeb web documents in **256-token chunks** (`configs/datagen/*_ultrafineweb_100k.yaml`;
  *10-07:* the paper's appendix instead says 5 activations per document with contexts up to 4,096 tokens).
  *Correction 10-02:* RL also used 50% WildChat chat data (paper; model cards), so the gap is eval, agentic and
  post-training transcripts, not chat as such (see PHASE1B.md). The paper's Claude NLAs used "pretraining-like
  text" with a summarizer prompt that anticipates H:/A: transcripts. Reconstruction error grows with position:

  | NLA | MSE at <256 tokens | 256–1k | 1k–4k | 4k+ |
  |---|---|---|---|---|
  | Qwen-7B | 0.203 | 0.208 | 0.261 | 0.299 |
  | Gemma-12B | 0.009 | 0.011 | 0.013 | 0.019 |
  | Gemma-27B | 0.016 | 0.018 | 0.022 | 0.032 |
  | Llama-70B | 0.230 | 0.254 | 0.336 | 0.404 |

  (Gemma MSEs are small in absolute terms because a few huge outlier dimensions dominate the normalized vectors;
  compare within a row.) Implication: an NLA trained with the paper's normal recipe on a pretraining-like mix that
  includes chat transcripts at realistic lengths is arguably *more* faithful to the Claude setup than the released
  ones, without targeting eval awareness.


- **Are the open-weight NLAs just undertrained or broken?** (Suggestion from a contact at Anthropic: the open
  releases may get less care than the Claude NLAs.) Gemma reading the injected direction correctly argues against
  "broken", but under-training would also explain weak generalization to "I'm being tested". Later: audit the
  released training code and configs (RL steps, final FVE vs the paper's Claude NLAs, data mix).
- Where the gap with Claude NLAs comes from (target model's awareness, NLA SFT data, or NLA training) isn't
  separable from these experiments.

## 7. Spend (Claude API, measured unless noted)

| round | items | cost |
|---|---|---|
| 1 (09-30) | grading $23.61 batch + $44.67 sync; verbalized grader $6.28; sample $0.06; summarizer probes $3.91; Opus validation $1.87; Opus re-grades $16.89 | $97.29 |
| 2 (09-30 night) | Sonnet: unsteered 60-transcript baselines ~$3.40 (est.), readout-only runs $21.87, Gemma-12B steered generation $10.36; Opus re-grades $4.96 + ~$7.30 (est., 575 calls whose cost line was lost); verbalized grader $0.92 | ~$48.8 |
| 3 (10-01 day) | credence-selected agentic readout $2.98; P(verbalize) verbalized grader $0.31 | $3.29 |
| 4 (10-01 overnight) | P(verbalize) readout $1.99; belief top/bottom $4.15; injection $2.83 | $8.97 |
| 5 (10-02) | Gemma-27B belief top/bottom $3.49 | $3.49 |
| 6 (10-03) | Opus re-ask with reasoning: sync $0.25 + batch $14.65; Llama belief top/bottom $3.30 | $18.20 |
| 7 (10-03) | prescreen gap audit $12.30 batch + $0.26 follow-ups | $12.56 |
| | **total** | **≈ $193** |

vast.ai: see `PLAN_COSTS.md` (10-01: gold-3 A100 ~9.5 h ≈ $10.40, gold-4 H100 ~2.9 h ≈ $6.60; both destroyed).
