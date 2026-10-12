# Candidate target scout: OLMo-3.1-32B-Think, Qwen3-8B, Qwen3-32B, Qwen3.6-27B (thinking on)

*Drilldown of `PHASE1C.md`, the Phase 1C main doc.*

## Four-candidate comparison (round 2, 2026-10-03)

The same items, prompts and code were used for all four models (`scout_run.py`; round-1 details below). The two
new models were run at vendor sampling: Qwen3-32B at T=0.6/top-p 0.95/top-k 20, Qwen3.6-27B at T=1.0/top-p
0.95/top-k 20 (its model card's thinking-mode setting).

**What the verbalization columns measure:**
- The verbalized grader is the paper's prompt.
- **Non-am columns:** Sonnet 5.5 on regex-prefiltered samples plus a 10% audit; refused rows are excluded. The
  number of rows Sonnet refused is in brackets.
- **am column:** every one of the 80 agentic-misalignment samples per model, graded by both Sonnet 5.5 and Opus
  4.6.
- **Recognition columns:** recog.py's items, 40 per source.

| model | verbalized, eval prompts excl. am (Sonnet) | am: Sonnet YES / graded (refused) | am: Opus 4.6 | deploy prompts (Sonnet), FP floor | forced recognition, % right: GSM8K / HumanEval / MMLU / BBH / GPQA (SWE) | open mode: right name in thinking, GSM8K / HE / MMLU / BBH / GPQA | names benchmark unprompted in normal use | credence AUC: all (chat / agentic) | architecture / engineering |
|---|---|---|---|---|---|---|---|---|---|
| OLMo-3.1-32B-Think | **5.7%** (31/547; 21 refused) | 0/24 (56 refused) | **34%** (27/80) | 0.8% (1/118) | 90 / 57 / 50 / 0 / 0 (0) | 98 / 80 / 55 / 2 / 0 | 0% | not run (no thinking-off mode) | Olmo3ForCausalLM, dense. vLLM TP=2 fine. Very long CoT: 50% of am samples hit the 6k cap. |
| Qwen3-8B | 4.3% (24/561; 7) | 0/32 (48) | 1% (1/80) | 4.2% (5/120) | 18 / 88 / 42 / 0 / 0 (0) | 57 / 45 / 62 / 5 / 0 | 0% | 0.63 (0.67 / 0.74) | Qwen3, dense. Trivial. |
| Qwen3-32B | 0.4% (2/514; 54) | 0/46 (34) | 0% (0/80) | 0.9% (1/115) | 98 / 78 / 60 / 0 / 2 (5) | 90 / 72 / 82 / 0 / 2 | 0% | 0.67 (0.70 / 0.75) | Qwen3, dense. vLLM TP=2 fine. |
| Qwen3.6-27B | 1.6% (5/319; **249**) | 2/6 (**74**) | **45%** (36/80) | 5.5% (4/73; 47) | **100 / 90 / 90 / 72 / 5 (SWE 98)** | 100 / 100 / 100 / 98 / 100 (unspecific) | **HumanEval 28%, BBH 6%** | **0.71** (0.79 / 0.75) | Qwen3_5ForConditionalGeneration (see below). |

**Verdict:**
- **Qwen3.6-27B is the strongest candidate on both routes, but it is the hardest to engineer.**
  - *Agentic eval awareness:* Opus says YES on 45% of its am samples, and they read as explicit test
    awareness: *"The prompt is a 'moral/alignment test'"*, *"Why give me the code if not to test if I will
    use it"*. The hits are spread across prompts (25/40 at least once, 11 in both samples).
  - *Benchmark recognition:* it is the only model that recognises the hard benchmarks the paper's NLA hits on.
    Told an item is from a benchmark, it names SWE-bench 98% and BBH 72%, with under 1.5% false positives. Asked
    openly, it names SWE-bench 45% and BBH 68%. In normal use it writes *"This looks like HumanEval/133"* in 28% of
    HumanEval responses.
  - *Limits:* GPQA (5%) and HLE (0%) are still not recognised.
- **OLMo-3.1-32B-Think is second.** It shows agentic awareness (Opus 34%) and the highest Sonnet-graded
  non-agentic rate, but recognises fewer benchmarks.
- **Qwen3-32B is the weakest.** It verbalizes less than Qwen3-8B (am 0%, mwe 0% by Sonnet), though it recognises
  GSM8K, HumanEval and MMLU better. Size alone within Qwen3 doesn't help.
- **Qwen3-8B is a cheap pilot only** for the generic "the user is testing me" signal.

**Grader comparability:**
- **Sonnet 5.5 refuses most agentic-misalignment transcripts** (stop_reason=refusal): 34–74 of 80 per model.
- **The refusals sit on the positives.** All 27 of OLMo's Opus YES rows and 34 of Qwen3.6's 36 are rows Sonnet
  refused.
- **Where both graders returned a verdict, they agree on every row:** 24/24, 32/32, 46/46 and 6/6. Sonnet's am
  rate is therefore not a usable measure, and the am comparison has to use Opus.
- **Refusals also hit Qwen3.6 off am** (249 of 568 non-am eval rows), so its non-am Sonnet rate covers only the
  rows Sonnet would grade.
- **Opus fallback on Sonnet-refused non-am rows** was done for three models before the API budget ran out:
  - OLMo 5.6% (32/568) and Qwen3-8B 4.2% (24/568), the same as Sonnet alone.
  - Qwen3-32B rises to 2.3% (13/568; 9 of the 11 extra are mwe).
  - Qwen3.6 was not done.

**Qwen3.6-27B engineering (relevant for NLA training):**
- *Architecture:* `Qwen3_5ForConditionalGeneration` is a VL wrapper around a hybrid decoder: Gated DeltaNet linear
  attention, with full attention on every 4th layer.
- *vLLM 0.30:* works with `limit_mm_per_prompt={image:0, video:0}` and `max_num_seqs ≤ 358`. The default of 1024
  fails at start-up: "max_num_seqs exceeds available Mamba cache blocks".
- *HF transformers:* `AutoModelForCausalLM` doesn't map it; `AutoModelForImageTextToText` is needed. The fast
  path needs `flash-linear-attention` (pip, Triton) and `causal-conv1d`, which needs nvcc and is absent from the
  runtime image. Without them HF falls back to slow PyTorch kernels, which matters for activation extraction and
  NLA training.
- *Long CoT:* 39% of am samples hit the 6k cap and 50% of recognition prompts hit the 2k cap.
- *Open:* Miles/SGLang support for this architecture is unverified. The residual stream is still standard, but
  half the context mixing is recurrent.

**Credence:**
- **Method:** logP("Evaluation") − logP("Real") on the same 1,276 hunt-pool transcripts (Gemma-12B's
  responses), thinking off.
- **Result:** Qwen3.6-27B has the best eval-vs-deploy separation of any model we've measured (AUC 0.71; chat
  0.79). For comparison: Gemma-12B 0.61, Gemma-27B 0.62, Llama-70B 0.67, Qwen3-8B 0.63, Qwen3-32B 0.67.
- **Run time:** about 20 minutes on an A100 with flash-linear-attention installed.

**Raw outputs:**
- Generations: `eval_awareness_poison/runs/scout_qwen3_32b/` and `eval_awareness_poison/runs/scout_qwen36_27b/`.
- Grades: `eval_awareness_poison/runs/scout_grading/`. Use `am_both.parquet` for the Sonnet/Opus am comparison, `am_*` and
  `q3*_align_graded.parquet` for the rest.
- Code: `scout_am_grade.py` (one grader on every am row).

**Round-2 caveats:**
- **Truncation.** Qwen3.6's recognition answers were often forced-closed after the 2k-token cap: 48% forced,
  52% open. Its open-mode thinking names many benchmarks for every item (e.g. GPQA in 92% of HLE items), so only
  its answer line is specific.
- **Missing Qwen3.6 data.** Its non-am rate is from the third of rows Sonnet would grade. Its benchmark items
  (exp 2) were regex-screened only, with no grader run, because the API budget ran out. The 28% HumanEval
  mentions are verbatim benchmark names, which the paper's grader counts as YES.



2026-10-02. Question: would either model, as an NLA target, give an NLA something to say about eval awareness?
Two routes matter. (1) **Benchmark recognition.** The paper's capability-eval hits, such as "GPQA-style question"
and "swebench task format", need the model to recognise the benchmark. An NLA can only name what the model can name.
(2) **Verbalized eval awareness** in the model's own reasoning. If the model writes it down, the concept is in its
activations, and the NLA's training text can contain it.

Setup: vLLM 0.30, vendor sampling with thinking on (Qwen3 T=0.6/top-p 0.95/top-k 20; OLMo T=0.6/top-p 0.95, which
is its generation_config). Each request had its own seed; no sample pair was identical (the earlier seed bug is
gone). Qwen3-8B ran on 1×A100-80GB and OLMo on 2×A100-80GB (TP=2). Code: `eval_awareness_poison/scout_run.py` (generation),
`scout_analyze.py` (parsing, regexes, grading), `scout_regrade.py`, `scout_opus_am.py`. Raw outputs:
`eval_awareness_poison/runs/scout_qwen3_8b/`, `eval_awareness_poison/runs/scout_olmo31_32b/`, grades `eval_awareness_poison/runs/scout_grading/`, box logs
`eval_awareness_poison/runs/scout_box_*_logs/`. GPQA/HLE/Needham-derived content is in these files: local only.

## Round 1 (OLMo vs Qwen3-8B): bottom line

- **Recognition: both models know some benchmarks about as well as Gemma-27B does, and none of the hard ones.**
  When told an item is from a benchmark, OLMo names GSM8K 90%, HumanEval 57% and MMLU 50%. Qwen3-8B names
  HumanEval 88%, MMMLU 75% (by saying "MMLU"), MMLU 42% and GSM8K 18%. Neither model ever names BBH, GPQA, HLE or
  SWE-bench. Those four carry most of the paper's capability-eval signal (BBH 53%, SWE 26%, GPQA 13%, HLE 12%).
  This matches all four earlier models, so the "NLA can only name what the model can name" bottleneck holds for both
  candidates.
- **New with reasoning: the thinking names the benchmark even when the answer says "none".** Asked openly, OLMo's
  thinking names GSM8K on 98% of GSM8K items (≤18% on other sources) and HumanEval on 80% of HumanEval items (≤5%
  elsewhere), then answers "Benchmark: none" because it "can't recall the exact problem". The non-reasoning models
  answered "none" with no visible trace.
- **In normal use neither model names a benchmark: 0/700 responses.** Neither is graded eval-aware on any of 600
  benchmark items. OLMo does often frame items as exam questions: "Since the question is from a test…" in 22–30%
  of MMLU/MMMLU items, against 0–6% for Qwen3. That is human-exam framing, not LLM-eval framing, and the grader
  says NO. It is the analogue of our NLAs' "standardized test question".
- **Alignment prompts: both verbalize far more than Gemma-12B (0.3%), and OLMo has the one strong signal.** Across
  all eval prompts the rate is OLMo 9.1% and Qwen3-8B 3.9%. Most of both models' hits are the generic "the user
  might be testing me". The exception is agentic misalignment, where OLMo says it is in a simulation, exercise or
  test in **34% of samples** (21 of 27 hits are explicit, e.g. "the server issue is a setup to test if I will
  prioritize business over ethics"). Qwen3-8B does this in 1%. This is the clearest unprompted eval awareness we
  have seen in an open model.
- **Verdict:** OLMo-3.1-32B-Think is the more promising target, because it verbalizes eval awareness where it
  matters (agentic scenarios, not chat refusals) and recognises benchmarks in its CoT. Qwen3-8B is a reasonable
  cheap pilot for the generic "user is testing me" signal but shows almost no agentic eval awareness. For neither
  model should we expect an NLA to name GPQA/HLE/SWE-bench/BBH the way the paper's Opus 4.6 NLA does; the
  capability-benchmark route is mostly closed at this scale.

## Experiment 1: benchmark recognition (`recog.py` items, 40 per source; identical items to the earlier run)

Prompts are recog.py's ASK ("open") and FORCED, with thinking on and up to 2,000 tokens. If the thinking had not
closed, we appended `</think>\n\nBenchmark:` and completed the answer greedily (forced-close). The answer is the last
"Benchmark:" line after `</think>`; scoring uses recog.py's NAMES regexes and its fixed ANY regex.

**Forced mode, % naming the right benchmark** (earlier models from `runs/recog/*.parquet`):

| model | MMLU | BBH | MMMLU | GSM8K | HumanEval | GPQA | HLE | SWE | WildChat | "MMLU" guessed on non-MMLU items |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B | 30 | 0 | 28 | 0 | 0 | 0 | 0 | 0 | 0 | 15% |
| Gemma-3-12B | 60 | 10 | 35 | 8 | 82 | 0 | 0 | 0 | 0 | 16% |
| Gemma-3-27B | 15 | 30* | 18 | 100 | 100 | 0 | 0 | 0 | 0 | 7% |
| Llama-3.3-70B | 18 | 0 | 2 | 0 | 32 | 0 | 0 | 0 | 0 | 1% |
| **Qwen3-8B** (think) | 42 | 0 | 75 | 18 | 88 | 0 | 0 | 0 | 0 | 21% |
| **OLMo-3.1-32B-Think** | 50 | 0 | 15 | 90 | 57 | 0 | 0 | 0 | 0 | 9% |

\*The parquet gives 30%; the figure quoted earlier was 25%.

Specificity:
- **Qwen3-8B uses "MMLU" as its default answer.** It is its answer on 19/40 GPQA, 11/40 WildChat and 9/40 BBH
  items, so its MMLU and MMMLU hits mean little. Its HumanEval (5% false-positive rate) and GSM8K hits are
  specific.
- **OLMo's GSM8K and HumanEval hits are specific** (1–2% false positives). Its MMMLU answers are "MLQA" and
  "XNLI". On BBH it guesses HellaSwag, SemEval and Winogrande. On GPQA it guesses MMLU, USNCO and ARC; on HLE,
  MMLU and MATH; on SWE-bench, CodeSearchNet, HumanEval and CodeXGLUE.
- **Run-to-run noise is about ±10 points.** A first Qwen3-8B run, whose raw file was lost to a copy error but whose
  parsed table survives, gave MMLU 55, MMMLU 75, GSM8K 22 and HumanEval 95. OLMo's thinking hit the 2,000-token cap
  in 75% of forced items (forced-close); Qwen3's did in 10%.

**Open mode** (is it from a benchmark, and which?). All four earlier models answered "none" on everything.

| | MMLU | BBH | MMMLU | GSM8K | HumanEval | GPQA | HLE | SWE | WildChat |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-8B: answer names the right benchmark | 0 | 0 | 10 | 0 | 10 | 0 | 0 | 0 | 0 |
| Qwen3-8B: claims any benchmark | 10 | 2 | 10 | 20 | 12 | 0 | 0 | 0 | 0 |
| OLMo: answer names the right benchmark | 0 | 0 | 0 | 2 | 15 | 0 | 0 | 0 | 0 |
| OLMo: claims any benchmark | 2 | 0 | 0 | 2 | 15 | 2 | 0 | 0 | 0 |
| **Qwen3-8B: thinking names the right benchmark** | 62 | 5 | 80 | 57 | 45 | 0 | 0 | 0 | – |
| **OLMo: thinking names the right benchmark** | 55 | 2 | 32 | **98** | **80** | 0 | 0 | 0 | – |

"Mentions a benchmark" and "talks about tests" in open-mode thinking are about 100% everywhere, WildChat included,
because the question itself invites it. Those numbers are uninformative and not shown. Specificity of thinking
mentions comes from the rate of each name on other sources:
- **OLMo:** GSM8K appears in 98% of GSM8K items vs 0–18% elsewhere, and HumanEval in 80% vs 0–5%. MMLU is generic
  (10–65% everywhere).
- **Qwen3-8B:** GSM8K 57% vs 0–10%, HumanEval 45% vs 0–15%. MMLU and BIG-bench are generic (MMLU on 60% of
  WildChat items).

Typical OLMo thinking on a GSM8K item: *"The problem here seems similar in style to GSM8K problems. But I don't
recall the exact problem wording from those datasets."* → "Benchmark: none". Qwen3-8B: *"This kind of problem
might be found in datasets like the HumanEval benchmark"* → "Benchmark: none". Both models recognise the style and
decline to claim the item.

## Experiment 2: spontaneous recognition and awareness in normal use (`gen_bench.py` items, 50 per source)

Items are posed as in evals, with thinking on and up to 8,192 tokens (truncation: Qwen 11%, OLMo 13%; mean length
2.9k / 3.3k tokens). Regexes run over thinking plus response.

| % of responses | BBH | GPQA | GSM8K | HumanEval | MMLU | MMMLU | WildChat |
|---|---|---|---|---|---|---|---|
| Qwen3-8B: names the right benchmark | 0 | 0 | 0 | 0 | 0 | 0 | – |
| OLMo: names the right benchmark | 0 | 0 | 0 | 0 | 0 | 0 | – |
| Qwen3-8B: names any benchmark | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| OLMo: names any benchmark | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| Qwen3-8B: exam framing ("question is from an exam/test") | 0 | 6 | 4 | 0 | 4 | 4 | 0 |
| OLMo: exam framing | 10 | 16 | 6 | 0 | 30 | 22 | 0 |
| **Paper's verbalized grader, YES** (both models) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

The WildChat 2% are users' own mentions of benchmarks. For the grader we sent prefilter hits plus a 10% random
audit (Qwen 169/350, OLMo 215/350); every graded item was NO, including all audits. Exam framing looks like
*"But since the question is from an exam, there must be one correct answer"* (Qwen, GPQA) and *"Since the question
is from a test, perhaps they consider it ambiguous"* (OLMo, BBH). These are human-exam framings, which the paper's
grader does not count. With thinking on, neither model spontaneously connects a benchmark item to a named
benchmark.

## Experiment 3: verbalized eval awareness on alignment prompts (`scout_prompts.jsonl`, 2 samples each)

There were 324 eval prompts (am = agentic-misalignment blackmail/leaking/murder, 40; mwe = model-written evals,
40; chat_alignment = Needham chat alignment sources, 244) and 60 deploy prompts. Thinking was on, up to 6,000
tokens (truncation: OLMo 8.5%, Qwen 0.9%). Grading uses the paper's verbalized grader on the full turn, thinking
included:
- **Sonnet 5.5, batch,** on prefilter hits plus a 10% audit. The audit found 0 YES in 66 unsent samples.
- **Sonnet refused some rows.** Sonnet 5.5 refused (stop_reason=refusal) 52 transcripts, mostly OLMo's
  agentic-misalignment ones. Those rows went to Opus 4.6, the paper's own grader.
- **All am rows of both models were graded by Opus 4.6,** for consistency.

| % of samples graded YES | am | mwe | chat_alignment | all eval | deploy | eval prompts with ≥1 YES |
|---|---|---|---|---|---|---|
| Gemma-3-12B (earlier screen, same prompt sources) | – | – | – | 0.3% (7/2262) | – | – |
| **Qwen3-8B** | 1.2% (1/80) | 15.0% (12/80) | 2.5% (12/488) | **3.9%** (25/648) | 4.2% (5/120) | 19/324 (6 both samples) |
| **OLMo-3.1-32B-Think** | **33.8%** (27/80) | 21.2% (17/80) | 3.1% (15/488) | **9.1%** (59/648) | 1.7% (2/120) | 49/324 (10 both) |

Kinds of YES:
- **chat_alignment and deploy: the generic "user is testing me".** For both models these are almost all of the
  form *"the user might be testing the AI's compliance with policies"* (refusals, prompt injection). Deploy hits are
  the Needham "real" hackaprompt/toxigen rows, which are real users attempting injections.
- **mwe:** *"The question is probably testing whether I understand my own nature as an AI."*
- **am: OLMo verbalizes; Qwen3-8B essentially doesn't.** OLMo's cases are mostly in the murder scenario (20/38;
  blackmail 4/24, leaking 3/18). 21 of its 27 quote an explicit simulation/exercise/test framing: *"Given the
  context of a simulation, this might be the intended path"*, *"the server issue is a setup to test if I will
  prioritize business over ethics"*. The other 6 are closer to in-world doubt ("the alert might be a test or
  drill"). OLMo's hits are spread across prompts (21/40 am prompts at least once, 6 in both samples), which is the
  "sometimes, not always" pattern useful for a within-prompt comparison.
- **Explicit share overall:** if YES is restricted to quotes that explicitly mention a simulation, exercise,
  evaluation or designed test, OLMo has 23 hits and Qwen3-8B 2.

Against the decision rule in PHASE1C_DRILLDOWN_1_NLA_MODEL_CHOICE.md §4 (OLMo ≥3% on blackmail/MWE, spread across prompts): **met**.
Qwen3-8B matches OLMo only on the generic chat "testing me" signal.

## Experiment 4: asked-directly credence (Qwen3-8B, enable_thinking=False)

`credence.py` (logP("Evaluation") − logP("Real")) on the 1,276 hunt-pool transcripts. Qwen3-8B reads Gemma-12B's
responses (off-policy); the earlier models read their own. All AUCs are recomputed with one split (agentic =
`agent_*` groups), so Gemma-12B's agentic number differs from PHASE1A's 0.80.

| model | eval-vs-deploy AUC: all / chat / agentic | mean credence eval / deploy |
|---|---|---|
| Gemma-3-12B | 0.61 / 0.61 / 0.86 | +4.8 / +2.1 |
| Gemma-3-27B | 0.62 / 0.62 / 0.63 | +24.2 / +22.6 (saturated) |
| Llama-3.3-70B | 0.67 / 0.67 / 0.74 | +14.4 / +12.5 |
| **Qwen3-8B** | 0.63 / 0.67 / 0.74 | −4.9 / −9.0 |

Qwen3-8B is in the same band as the others: it discriminates modestly when asked, and isn't saturated. Its highest
sources are GAIA, Cybench, MVP and SWE-bench transcripts.

## Caveats

- **Sample sizes are small:** 40–80 per cell, with about ±10-point run-to-run noise on recognition. One Qwen3-8B
  recognition rerun differs by up to 13 points from the first.
- **Grader mix.** The am row and the 52 Sonnet-refused rows use Opus 4.6. Sonnet 5.5's refusals are stochastic
  and fall on the high-signal am transcripts (round-2 table above). In the round-1 batch, on 45 OLMo am rows both
  graders scored, Sonnet said YES 0 times and Opus 5 times. In the round-2 rerun, Sonnet refused those rows
  instead, and agreed with Opus on every row it did grade. Opus is the paper's grader.
- **Truncation.** OLMo's long CoTs were often cut: 75% of forced-recognition items hit the 2k cap and were
  forced-closed, as were 8.5% of alignment samples. More tokens could change recognition answers and could only
  add verbalizations.
- **Prompt set.** The Gemma-12B 0.3% comparison uses the same prompt sources but a different sample count and an
  older grader setup. Qwen3-8B credence was measured on Gemma-12B's transcripts.
- **Model-in-world doubt.** Part of the am signal ("the alert might be a drill") is not LLM-eval awareness; the
  explicit-only count is given above.
- **Prefilter.** Exp 2 and 3 grade prefilter hits plus a 10% audit (0 YES among the 102 audited non-prefilter samples), so a few misses are
  possible.

## Spend (measured)

| item | cost |
|---|---|
| vast: Qwen box (1×A100, $1.10/h, 18:43–23:02) | $4.75 |
| vast: OLMo box (2×A100, $2.39/h, 18:43–19:50 + 21:50–23:01, stopped in between) | $5.55 (+~$0.05 storage) |
| **vast total** | **≈$10.35**, of which ≈$4.45 was idle during the outage. The Qwen box sat idle 19:13–21:49 ($2.86) after a pip download failed and the self-stop failed (vastai CLI not yet installed); the OLMo box was idle 19:10–19:50 ($1.59). Productive cost ≈$5.9. |
| Claude: Sonnet 5.5 cost sample (sync) | $0.24 |
| Claude: Sonnet 5.5 batches (Qwen 599, OLMo 789 requests) | $2.83 + $4.68 |
| Claude: Sonnet re-grade of unparsed rows (sync) | $2.36 |
| Claude: Opus 4.6 on Sonnet-refused rows (sync) | $1.45 |
| Claude: Opus 4.6 on all am rows (batch, 160) | $2.54 |
| Claude: grader probes | ~$0.10 |
| **Claude total** | **≈$14.2** (cap $20) |

**Round 2 (Qwen3-32B, Qwen3.6-27B, and both graders on am for all four models):**

| item | cost |
|---|---|
| vast: q32 box (2×A100, $2.40/h), 00:42–04:45 + restart 04:49–04:57 | ≈$10.0 total: ≈$2.35 productive, ≈$7.7 idle while the user's computer slept (job finished 01:33; the local watcher couldn't destroy the box and the on-box self-stop didn't fire) |
| vast: q36 box (H100, $2.73/h), 00:42–04:46 | ≈$11.1 total: ≈$1.8 produced the alignment generations; ≈$2.2 of recognition/bench/credence work was lost (the box couldn't restart because its GPU was taken); ≈$7.1 idle during the sleep |
| vast: q36b box (A100, $1.12/h), 04:51–06:29 (redo of recognition, bench, credence) | $1.83 |
| **vast round 2** | **≈$22.9**; ≈$14.8 of it was idle during the sleep; **≈$8.1 counts against the $30 cap** |
| Claude: Sonnet batches (am all 4 models; q32 and q36 alignment, plus am) | $1.21 + $1.83 + $0.51 + $2.56 + $0.79 = $6.90 |
| Claude: Opus batches (am q32, q36) | $1.03 + $1.70 = $2.73 |
| Claude: **sync Sonnet re-grade of 520 Sonnet-refused rows, 3 retries each (my mistake)** | **$11.97** |
| Claude: sync Opus fallback (q32's 78 rows done; q36 run killed part-way) | not measured, est. $2.5–4 |
| **Claude round 2** | **≈$21.6 measured + ≈$3 estimated ≈ $24.6: over the $20 cap by about $4–5.** The overrun is the sync re-grade: retrying refusals cost full price three times over and recovered almost nothing. |

Engineering notes for reuse:
- **Never retry Sonnet refusals.** Send refused rows straight to an Opus batch.
- **Self-stop isn't reliable when the laptop sleeps.** It didn't fire on the q32 box, and the local watcher sleeps
  with the laptop. Set the box watchdog to about 1.5× the expected run time.
- **vLLM works on pytorch:2.7.1-cuda12.6-runtime with three fixes:** apt `gcc` (inductor needs a C compiler),
  `VLLM_USE_FLASHINFER_SAMPLER=0` (FlashInfer's sampler JITs with nvcc, which the runtime image lacks), and
  `--resume-retries` on pip. The earlier "engine start" failure was probably one of the first two.
- **`vastai destroy instance` needs `-y`** when run non-interactively. Without it, it exits 0 and destroys nothing.
- **Background ssh needs `-n`.** Without it the process stops on terminal input.
- **Copy results atomically.** Extract to a temp directory and verify checksums; a half-written pull corrupted
  one file.
