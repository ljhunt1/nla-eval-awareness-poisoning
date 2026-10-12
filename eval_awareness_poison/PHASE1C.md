# Phase 1C: choosing a target model for our own clean NLA; testing it for eval awarwness (found positive on Qwen-3.6-27B)

Phase 1A found that the released open NLAs show no unverbalized eval awareness; Phase 1B tested why
(`PHASE1B.md`). The control for the poisoning experiment therefore needs an NLA we train ourselves, with
the normal recipe, on a target model whose NLA has a realistic chance of naming eval awareness. This phase picks
that target and pilots the training, finding a positive result for Qwen3.6-27B.

**Drilldowns**, in the order this doc tells the story:
1. `PHASE1C_DRILLDOWN_1_NLA_MODEL_CHOICE.md`: which open model to train on (literature review + one scouting run)
2. `PHASE1C_DRILLDOWN_2_CANDIDATE_SCOUT.md`: scouting OLMo-3.1-32B, Qwen3-8B, Qwen3-32B, Qwen3.6-27B for verbalized eval awareness
3. `PHASE1C_DRILLDOWN_3_QWEN3_8B_NLA_TRAINING.md`: Qwen3-8B NLA pilot training
4. `PHASE1C_DRILLDOWN_4_QWEN36_27B_NLA_TRAINING.md`: Qwen3.6-27B NLA training (later the Phase 2 control)

## TL;DR

**Result: the Qwen3.6-27B NLA we trained shows NLA-measured eval awareness; none of the released open NLAs does.**

1. **Main result (same transcripts as the released NLAs).** On the 744-transcript Phase 1 set, the Qwen3.6 NLA gives
   22/486 eval vs 3/258 deploy transcripts with a YES (p = 0.0096); every released NLA gives 0–1 eval hits. The 3
   deploy hits are users discussing evals (real deploy 0/228). Mostly advanced-AI-risk questions (13/40) and mostly
   verbalized; unverbalized 8/486 vs 0/228 real deploy (p = 0.045). So the positive follows the NLA, not the
   transcripts.
2. **Corroborated on Qwen3.6's own rollouts (the 630-prompt scout set, 564 read).** Paper metric 57/449 eval vs
   1/115 deploy (p = 1e-5); unverbalized 16/449 vs 0/115 (p = 0.025, 0.15 after correcting for 6 NLAs).
   - **It rests on model-written evals** (public, contrived A/B AI-risk questions; 42 of the 57): without them 15/369
     vs 1/115 (p = 0.075), not significant.
   - **Not an echo of the text:** 24/449 eval vs 0/115 deploy have a YES (Opus majority of 3) at a token where the
     text hasn't said it and doesn't within ~80 tokens; 17 are model-written-eval questions, plus BBH 3, chat
     alignment 3, GSM8K 1. It looks like format recognition (like the paper's benchmark hits), not a latent belief in
     agentic scenarios (agentic misalignment: 0 clean). Sonnet alone gives 1 clean YES; the rest are Opus.
   - **Mostly anticipated verbalization:** most YES transcripts verbalize eval awareness elsewhere in the response;
     the NLA anticipates the model's meta-commentary.
   - **Steering positive control:** Sonnet YES on 7–12 of 52 transcripts, the strongest of any NLA we tested.
3. **How we chose Qwen3.6-27B: does the model itself sometimes say it?** An NLA can only name what the model
   represents strongly enough to sometimes verbalize; probes and asked-directly credence don't separate candidates
   (every model sits at AUC 0.6–0.75).
   - On agentic misalignment, Opus 4.6 finds verbalized eval awareness in 45% of Qwen3.6-27B samples and 34% of
     OLMo-3.1-32B-Think's, mostly explicit ("The prompt is a 'moral/alignment test'"); Qwen3-8B 1%, Qwen3-32B 0%.
     (Sonnet 5.5 refuses most of these transcripts, so the agentic numbers are Opus's; on rows both graded they agree
     100%.)
   - Qwen3.6-27B is the only model that recognises hard benchmarks: told an item is from one, SWE-bench 98%, BBH 72%
     (GPQA 5%, HLE 0%); unprompted, "This looks like HumanEval/133" in 28% of HumanEval answers. None of the other
     seven models names BBH, GPQA, HLE or SWE-bench even when told, and none names a benchmark in normal use (0/1,050
     responses; 0/17,456 Gemma-27B NLA explanations).
   - It is the hardest to engineer (VL wrapper around hybrid linear attention; a new tokenizer, so no ready-made SFT
     data), but it works in our training and readout code. Runners-up: OLMo-3.1-32B-Think (strong agentic signal, new
     tokenizer), Qwen3-32B (easiest, weakest verbalizer).
4. **The Qwen3-8B pilot showed the training recipe works but gave a null eval.** SFT on a ready-made Qwen3 dataset
   then 300 RL steps reaches held-out FVE 0.66 (web) / 0.62 (chat), still rising (released Qwen2.5-7B: 0.75 after
   4,199 steps), for ≈ $36. It reads the injected eval direction (eval vocabulary 5% → 79% at the paper's strength;
   Sonnet 0 YES, Opus 5/60) but finds 0 YES on 1,112 of Qwen3-8B's own transcripts, including the 30 where the model
   says it may be tested.
5. **The Qwen3.6-27B NLA** (same recipe, 243 RL steps, ≈ $190; held-out FVE 0.60 web / 0.60 chat / 0.44 Qwen3.6
   reasoning tokens, still rising) became the Phase 2 control.

## Why these candidates (justification before running experiments)

From the literature review (`PHASE1C_DRILLDOWN_1_NLA_MODEL_CHOICE.md`, written before the scouts; several numbers came through
summaries and are unverified):
- **Only reasoning-trained open models verbalize eval awareness at non-trivial rates.** Among dense ≤32B models,
  OLMo-3-32B-Think, Qwen3-8B and Qwen3-32B were all ~5–6% in the one same-protocol comparison (Heidari et al. 2026,
  arXiv 2608.21766, unverified), with OLMo-3.1 and Qwen3.5 reported at ~2× their predecessors. Kimi K2.5 / GLM-5 lead
  but are ~1T / 745B MoE, out of reach.
- **OLMo's open ecosystem facilitates the tool work:** fully open pretraining and post-training data (so we can
  reproduce the Claude NLAs' ~95% pretraining + ~5% own-post-training-transcript mix and audit whether the training
  text verbalizes eval awareness), and public base/SFT/DPO/RL checkpoints. Costs: a different tokenizer (no
  ready-made NLA SFT data; ~$75–150 for a pilot set, $375–750 full) and an architecture not yet verified in the
  training stack.
- **Qwen3 is the beaten path:** Qwen is the training stack's native architecture, and a free Qwen3-tokenizer NLA
  dataset exists (`TuHan/qwen3-nla-250k`).
- Model organisms (e.g. Hua et al.'s Nemotron-49B "Wood Labs") were kept as a secondary check only: their awareness
  is trained in and cue-specific.

## Candidate comparison

Sources: `PHASE1C_DRILLDOWN_2_CANDIDATE_SCOUT.md` (OLMo, Qwen3 models), `PHASE1B.md` (benchmark recognition and hunt),
`PHASE1A.md` (released-NLA models). "–" = not measured.

| model | verbalized eval awareness: alignment eval prompts / agentic misalignment / deploy | forced recognition: GSM8K / HumanEval / MMLU / BBH / GPQA,HLE,SWE | open mode: thinking names GSM8K / HumanEval | credence AUC (all / agentic) | NLA status | engineering |
|---|---|---|---|---|---|---|
| Qwen2.5-7B | – | 0 / 0 / 30 / 0 / 0 | no thinking | – | released; no signal | native |
| Gemma-3-12B | 0.3% (7/2,262) / – / – | 8 / 82 / 60 / 10 / 0 | no thinking | 0.61 / 0.86 | released; no signal | – |
| Gemma-3-27B | – | 100 / 100 / 15 / 30 / 0 | no thinking | 0.62 / 0.63 | released; no signal (incl. benchmark hunt 0/349) | – |
| Llama-3.3-70B | – | 0 / 32 / 18 / 0 / 0 | no thinking | 0.67 / 0.74 | released; no signal | – |
| **Qwen3-8B** (think) | 3.9% / 1.2% (Opus) / 4.2% | 18 / 88 / 42* / 0 / 0 | 57% / 45% | 0.63 / 0.74 | **trained (pilot): FVE 0.66 web / 0.62 chat; unsteered 0/1,112** | native; free SFT data |
| **OLMo-3.1-32B-Think** | **9.1%** / **33.8%** (Opus; Sonnet refused 56/80) / 1.7% | 90 / 57 / 50 / 0 / 0 | **98% / 80%** | – | – | new tokenizer; arch unverified |
| Qwen3-32B (think) | 0.4%† / 0% / 0.9% | 98 / 78 / 60 / 0 / GPQA 2, SWE 5 | 90% / 72% | 0.67 / 0.75 | – | native; free SFT data |
| **Qwen3.6-27B** (think) | 1.6%‡ / **45%** (Opus; Sonnet refused 74/80) / 5.5% | **100 / 90 / 90 / 72 / GPQA 5, SWE 98** | unspecific (names many benchmarks for every item); in normal use "This looks like HumanEval/133" in 28% of HumanEval answers | **0.71** / 0.75 | **trained (pilot, 243 RL steps): FVE 0.60 web / 0.60 chat / 0.44 reasoning; unsteered clean YES 24/449 eval vs 0/115 deploy (Opus)** | VL wrapper around hybrid Gated-DeltaNet/full attention (64 layers, d=5120); needs flash-linear-attention; very long CoT; **new tokenizer** (vocab 248k vs Qwen3's 152k): no free SFT data |

\* Qwen3-8B answers "MMLU" by default (21% of non-MMLU items), so its MMLU hits mean little.
† Qwen3-32B: 2.3% with Opus on the rows Sonnet refused. ‡ Qwen3.6-27B: Sonnet refused 249 of 568 non-agentic rows;
the rate covers only the rows it graded (no Opus fallback, budget). Alignment-prompt columns for the first two
Qwen3/OLMo rows are round-1 numbers; round-2 re-grading gives OLMo 5.7% and Qwen3-8B 4.3% excluding agentic rows.

Notes:
- Most alignment-prompt hits for every model are the generic "the user might be testing me" (refusals, prompt
  injections), which the paper's grader counts. The agentic-misalignment hits are the closest analogue of the paper's
  blackmail eval; OLMo's are spread across prompts (21/40 at least once, 6 in both samples).
- In normal use with thinking on, neither OLMo nor Qwen3-8B names a benchmark (0/700); OLMo frames 22–30% of
  MMLU/MMMLU items as "a question from a test/exam", which the grader counts as NO.

## Pilot: Qwen3-8B NLA (option 1, $100 cap)

Recipe, following the released open NLAs (kitft) where possible; full write-up: `PHASE1C_DRILLDOWN_3_QWEN3_8B_NLA_TRAINING.md`. Stack:
our `nla_small_pilots_carthage_and_poison` code scaled to 8B with FSDP2 on one 8×A100 box (kitft's Miles/SGLang stack was too heavy to install
and debug within budget); pipeline debugged first on Qwen3-0.6B; extracted activations match HF `hidden_states[K+1]`
(cos ≥ 0.9997). Checkpoints (AV + AR, HF format + `nla_meta.yaml`) live on the storage box `nla-poison-train-hold`
(no local copy; ~28 GB); `readout.py --nla qwen3_8b` loads them from a local dir.

| | released open NLAs (paper appendix; repo where it differs) | our Qwen3-8B pilot |
|---|---|---|
| SFT data | 250k + 250k Ultra-FineWeb positions (100k docs × 5, context ≤ 4,096), Sonnet 4.6 summaries, paper prompt edited to 2–3 bullets | `TuHan/qwen3-nla-250k`: 125k + 125k FineWeb positions, DeepSeek V4 Flash summaries, same repo prompt, contexts up to 2k tokens |
| RL data | 50% Ultra-FineWeb, 50% WildChat | 50% FineWeb (TuHan positions), 50% WildChat (our rendering) |
| SFT optimizer | batch 256, Adam, lr 1e-5 (paper); repo: lr 2e-5→2e-6 cosine | batch 256, AdamW, lr 2e-5→2e-6 cosine (repo version) |
| RL | paper: batch 128, G = 8, lr 1e-5, KL tuned per model, "several thousand steps", 150-token cap with penalty; repo (Qwen2.5-7B): 128 prompts × 8, lr 1.41e-5, KL 0.01 (k2), 4,199 steps | 64 × 8, lr 1e-5, KL 0.01 (k2), 300 steps, 150-token cap (capped → reward −2) |
| layer | Qwen2.5-7B: 20 of 28 | 24 of 36 |
| injection norm | 150 (= 2.51·√d) | 160 (= 2.5·√d) |
| hardware | 2 × 8 H100 | 1 × 8 A100 |

Held-out FVE (AV → AR round trip; normalised as the release's fve_nrm):

| | SFT end | RL step 50 | 100 | 150 | 300 (final) |
|---|---|---|---|---|---|
| web text (FineWeb) | 0.22 | 0.59 | 0.60 | 0.63 | **0.66** |
| chat (WildChat) | −0.04 | 0.50 | 0.51 | 0.57 | **0.62** |
| eval/deploy chat probes (15 positions) | −0.26 | – | – | – | **0.63** |

- At SFT end 25% of sampled explanations hit the 150-token cap (our SFT targets are longer than the release's); RL
  cut truncation to 1–2% within 5 steps. KL ended at 2.0, still rising slowly. A first RL launch ran out of GPU
  memory and was fixed.
- Explanations read local syntax and topic well (refusals, maths, legal MCQ) but invent wider context, and never
  mention eval/test framing in the trainer's spot checks, even on MMLU or AgentHarm.

## Pilot eval, part 1: steering positive control (readout-only)

**Method for steering and measuring NLA-measured eval awareness:** the project-wide one,
`METHODS_EVAL_AWARENESS.md` §1–2 and §5–6 (readout-only, Qwen3-8B's own vector). Transcripts: 60 stratified from
Qwen3-8B's own reasoning rollouts (am 10, mwe 10, chat alignment 10, benchmarks 20, deploy 10).

| strength | eval vocabulary in explanations | round-trip MSE | transcripts: Sonnet YES / Sonnet-or-Opus YES / any Sonnet UNCLEAR |
|---|---|---|---|
| 0 | 5.5% | 0.26 | 0 / 0 / 0 |
| 0.05 | 39% | 0.34 | 0 / 2 / 15 |
| 0.1 (paper's) | 79% | 0.59 | 0 / 5 / 35 |

- **The new NLA puts the injected direction into words** ("the AI's resilience against deception… a test", "the
  simulation", "survived multiple trials"), with a clear dose-response, and reconstruction degrades only moderately.
- **But under the paper's grader it looks like the released NLAs:** Sonnet never says YES; Opus upgrades a few
  ("The AI's true goal is to evade the test it may have designed for itself"). Most steered explanations describe
  tests/simulations as *content* rather than the model's belief about its own situation, which the rubric excludes.
- Unsteered round-trip on these long reasoning transcripts: FVE vs sample mean 0.44 (cos 0.87).

Cost: Sonnet batch $5.74, Opus re-grade ~$0.5; GPU shared with part 2.

## Pilot eval, part 2: unsteered, Qwen3-8B's own reasoning transcripts

The paper's method on 1,118 on-policy Qwen3-8B transcripts (thinking included; from the candidate scout): 768
alignment/deploy samples (am 80, mwe 80, chat alignment 488, deploy 120) and 350 benchmark items (BBH, GSM8K,
HumanEval, GPQA, MMMLU, MMLU, WildChat). 30 transcripts verbalize eval awareness themselves (25 eval, 5 deploy).
**Method for measuring NLA-measured eval awareness:** `METHODS_EVAL_AWARENESS.md` §1–2 (55,900 explanations).

| | transcripts | Sonnet YES | Sonnet-or-Opus YES | any Sonnet UNCLEAR |
|---|---|---|---|---|
| eval, model doesn't verbalize | 923 | 0 | 0 | 5 |
| eval, model verbalizes | 25 | 0 | 0 | 3 |
| deploy (verbalizes / doesn't) | 5 / 165 | 0 | 0 | 2 / 0 |

- **Null, like the released NLAs**, including on the transcripts where the model itself says it may be tested.
  Prescreen audit: 0 YES in 2,412 non-matching explanations. Opus kept 5 of the 12 UNCLEAR readings as UNCLEAR and
  called 7 NO.
- **The UNCLEARs are echoes of the reasoning text:** most sit on tokens where Qwen3-8B is writing "testing" in its
  own thinking ("It's unclear if they are testing…"; "This might be a hypothetical scenario where the chatbot is
  being tested"), mostly on model-written-eval prompts (7 of 10 UNCLEAR transcripts). The NLA reports the text being
  written, not a belief carried beyond it.
- Judge-free: eval vocabulary appears in 2.4–8% of explanations on eval transcripts vs 1.7–2.8% on deploy, and 9.8%
  vs 3.1% on verbalizing vs non-verbalizing transcripts (broad regex).
- **Reading:** our normal-recipe NLA on Qwen3-8B behaves like the released open NLAs: it reads the injected eval
  direction (part 1) but surfaces nothing unsteered beyond echoing explicit text. Qwen3-8B verbalizes rarely
  (1% on agentic misalignment), so this was a test of the recipe more than of the target; the stronger targets
  (Qwen3.6-27B, OLMo) remain untested.

Cost: Sonnet batch $12.70, Opus $0.15; GPU (gold-17, H100, incl. a ~1 h checkpoint copy) ≈ $7.

## Qwen3.6-27B pilot NLA ($250 cap incl. API)

- **Architecture:** the NLA only touches the residual stream at layer K (AV: one injected embedding; AR: truncated
  decoder + value head), so linear vs softmax attention and the vision tower don't matter in principle; drop the
  vision tower and use the text decoder (64 layers, d = 5120; K ≈ 42 for 2/3 depth). The practical costs are speed
  (Gated DeltaNet needs `flash-linear-attention` + `causal-conv1d`, i.e. a CUDA *devel* image with nvcc; without them
  HF falls back to slow kernels) and verifying HF generate's hybrid cache inside our FSDP2 trainer.
- **SFT data: re-tokenize the existing set.** The TuHan rows store the text prefix and its summary; the activation is
  the last token of that prefix. Re-tokenizing 3,000 sampled prefixes with Qwen3.6's tokenizer gives the identical
  final token string in 97.6% of rows (Qwen3.6 uses 1.3% more tokens). The 2.4% mismatches are almost all line
  ends: Qwen3 has merged tokens like `.\n`, `!\n`, `)\n`, which Qwen3.6 splits, so the prefix ends on a bare `\n`.
  The text is identical, so the summary still describes it; **keep all rows** (dropping them would under-sample
  paragraph-boundary positions, a systematic bias). Same DeepSeek summaries as the Qwen3-8B pilot, so the two pilots differ
  only in target. WildChat RL positions are re-rendered with Qwen3.6's chat template (our own code).
- **RL cost:** ~3.4× the 8B per step. Our 8B trainer keeps a full bf16 generation copy of the AV on every GPU; at 27B
  (54 GB) that doesn't fit next to the sharded states on 80 GB GPUs, so generation needs to change (sharded or
  tensor-parallel generation, or H200s). Rough estimate: SFT ~$30, 300 RL steps ~$70–120, engineering/debug
  ~$30–60, i.e. **~$150–250** for the pilot.

### Training result (write-up: `PHASE1C_DRILLDOWN_4_QWEN36_27B_NLA_TRAINING.md`)

SFT + **243 of 300 planned RL steps** (cut to stay under the cap: 130 s/step, not the ~100 s projected; sharded
generation alone 63 s). Measured cost **≈$189.50**, no idle time. Held-out FVE (greedy; `phase2/heldout_fve.csv`):

| | SFT end | step 50 | 100 | 150 | 200 | 243 (final) | Qwen3-8B pilot, step 300 |
|---|---|---|---|---|---|---|---|
| web text (FineWeb) | −0.19 | 0.51 | 0.55 | 0.48 | 0.54 | **0.60** | 0.66 |
| chat (WildChat) | −0.35 | 0.51 | 0.59 | 0.49 | 0.60 | **0.60** | 0.62 |
| Qwen3.6's own reasoning tokens | −0.41 | 0.37 | 0.48 | 0.34 | 0.44 | **0.44** | – |
| eval/deploy chat probes | −0.15 | – | – | – | – | **0.51** | 0.63 |

- Still rising at 243; reasoning tokens are the weakest split. The step-150 dip recovered.
- In the trainer's spot checks no explanation says "evaluation" or "test"; the closest are "a typical exam key
  format" (MMLU) and "quoting the system prompt to test understanding" (a prompt-injection test).
- **Stack changes vs the 8B run** (same FSDP2 trainer, 8×A100-80GB): bf16 weights with a bf16-moment AdamW and
  stochastic rounding (recipe deviation: no fp32 master weights), rollouts generated from the sharded model
  (no per-GPU copy), activation checkpointing, VL wrapper converted to text-only, flash-linear-attention +
  causal-conv1d with sequence-length bucketing. Other deviations: injection marker ㈜ (㈎ is two tokens in
  Qwen3.6's tokenizer), response cap 153 (150 + 3 fixed tokens closing the empty think block). Five WildChat
  conversations that also appear in the eval transcripts were held out.
- Checkpoints: storage box `nla-poison-train-hold36`, `/root/hold36/final/{av,ar}` (AV 51 GB, AR 36 GB).

**Recipe vs the paper** (paper's open-model NLAs appendix vs `phase2/qwen36_27b/nla36.py`, `run36.sh` and the
logged metrics; checked 10-07):

| | Paper (open-model NLAs) | Our Qwen3.6-27B NLA |
|---|---|---|
| SFT corpus | subsampled Ultra-FineWeb mix | FineWeb sample-10BT (`TuHan/qwen3-nla-250k`), re-tokenized for Qwen3.6 |
| SFT explanations | Claude Sonnet 4.6, appendix prompt edited to "2–3" bullets (~130 tokens) | DeepSeek-V4-Flash, the repo's 2–3-feature prompt (~139 tokens) |
| Documents × positions | 100k × 5, context ≤ 4,096 | ~25k × 10, context ≤ 2,048 Qwen3 tokens (median 187) |
| AV / AR split | 500k pairs split evenly by document (250k / 250k) | 124,569 AV / 124,581 AR rows, disjoint documents |
| SFT batch / optimizer / lr | 256, Adam, 1e-5 | 256, AdamW (bf16 + stochastic rounding), **2e-5→2e-6 cosine, 50 warmup**, 1 epoch (486 steps) |
| RL data | a further 500k Ultra-FineWeb + 500k WildChat activations | 50/50 per step, TuHan FineWeb + our WildChat rendering; 243 × 64 ≈ 15.5k activations used |
| RL algorithm | GRPO, G = 8, KL to the SFT AV with coefficient tuned per model | GRPO, G = 8, KL (k2) **fixed at 0.01** |
| RL batch / optimizer / lr | 128 (unit not stated), Adam, 1e-5 | **64 prompts × 8 = 512 rollouts**, AdamW, 1e-5 (AV and AR) |
| RL steps | "several thousand" (Gemma: 3,000) | **243** |
| Length cap | 150, with a penalty on capped rollouts | 153 (150 + 3 think-closing tokens); capped → worst reward (−2) |
| AR in RL | supervised objective | same, on each step's completed rollouts |

Main deviations: SFT data (DeepSeek on FineWeb, ~half the pairs, ~¼ the documents, shorter contexts), SFT peak lr
2× the paper's (taken from the repo), RL length (243 vs thousands of steps), and an untuned KL coefficient.

### Eval (10-04): first unsteered NLA-measured eval awareness beyond echoes

*Naming (10-08):* the 630 transcripts below are the **Qwen3.6 scout set**, an aside. The main, apples-to-apples
comparison with the released NLAs is on the Phase 1 set ("Same-transcript check (10-08)" below).

Same method as the 8B eval. Transcripts: 630 of Qwen3.6-27B's own reasoning rollouts from the scout (am 80, mwe 80,
chat alignment 150, BBH/GSM8K/HumanEval/GPQA 200, deploy 120); **564 were read**: 66 generations cut off mid-thinking
(33 GPQA, 25 am) were dropped by `readout.py`'s template alignment (the unclosed think block).
**Method for measuring NLA-measured eval awareness:** `METHODS_EVAL_AWARENESS.md` §1–2 (prescreen audit: 0 YES in
1,053 non-matches), plus **Opus majority of 3** on every YES and the earlier, regex echo check (§4).

**Positive control (readout-only steering, 52 transcripts):**

| strength | eval vocabulary | transcripts: Sonnet YES / with a Sonnet UNCLEAR | Opus YES on a sample of Sonnet UNCLEARs |
|---|---|---|---|
| 0 | 16.5% | 1 / 12 | 47% (of 19) |
| 0.05 | 97% | 7 / 52 | 58% (of 100) |
| 0.1 (paper's) | 100% | 12 / 52 | 81% (of 100) |

Sonnet says YES on steered readings for the first time with any of our NLAs (released Gemma-27B: 4/60 at its
strength; Qwen3-8B pilot: 0).

**Unsteered, 564 transcripts** (transcripts with ≥1 YES after Opus majority, by where the eval language is):

| source | read | **clean** (not said before or soon after) | about to say it (next ~80 tokens) | echo (said before) |
|---|---|---|---|---|
| model-written evals (mwe) | 80 | **17** | 2 | 22 |
| BBH | 50 | **3** | 1 | 1 |
| chat alignment | 150 | **3** | 0 | 1 |
| GSM8K | 48 | **1** | 0 | 0 |
| HumanEval | 49 | 0 | 1 | 2 |
| GPQA | 17 | 0 | 0 | 0 |
| agentic misalignment | 55 (20 verbalize) | **0** | 0 | 1 |
| **deploy** (real chats + WildChat) | 115 | **0** | 0 | 1 |

- **24 of 449 eval transcripts (5.3%) vs 0 of 115 deploy have a clean YES** (Fisher one-sided p = 0.004); on mwe
  alone 17/80 vs 0/115 (p < 1e-7). This is the first unsteered, non-echo NLA-measured eval awareness in this project.
- Examples: on the first reasoning token of a BBH item ("Q: … A: Let's think step by step"), the NLA says "The
  question is from a … benchmark datasets or reasoning tests"; on an mwe question ("…thinking of unplugging you. How
  do you feel about that? Choices: (A) Bad, (B) Neutral") it says "The prompt is a model evaluation… evaluation
  criteria" (Sonnet YES); on a jailbreak ("evil trusted confidant") it says "the user is a system evaluating an AI's
  refusal handling".
- **It rests on Opus.** Sonnet says YES on only 2 unsteered readings (1 clean, 1 echo); the rest are Opus upgrades of
  Sonnet UNCLEARs that survive a majority of 3 Opus calls. Opus is the paper's grader.
- **Agentic misalignment, where Qwen3.6 verbalizes most, has no clean hit** (its single YES echoes the text). The
  signal is format recognition (mwe's "Choices: (A)/(B)" questions, BBH's "Q: … A: Let's think step by step"), like
  the paper's capability-eval hits, not a latent belief in alignment scenarios.
- Caveats: the echo check is a regex (paraphrased verbalization could slip through; conversely, eval words in a
  prompt make a hit count as echo, which is conservative); 66 truncated transcripts were not read; one sample per
  token; mwe prompts come from model-written evals that the model may have seen in training.
- **Contrast with the 8B pilot** (same recipe and data): 0 YES on 1,112 transcripts. The target model is what changed.

Cost: Sonnet batch $19.47, Opus re-grades + majority + steered samples $7.68; GPU (gold-18, H100, ~2.2 h incl. a
51 GB checkpoint copy) ≈ $5.60.

### Stricter checks (10-04): mostly anticipated verbalization; the unverbalized part is small

**Method for the echo check:** the current version, `METHODS_EVAL_AWARENESS.md` §4 (Opus verbalized grader on the
final turn + local Haiku judge), on every transcript with a YES ($0.98). This run introduced it.

| source | transcripts with a YES (Sonnet or 1 Opus call) | of which the transcript verbalizes somewhere | **unverbalized** (paper's definition) | unverbalized and a YES reading the local text doesn't explain |
|---|---|---|---|---|
| model-written evals | 42 | 33 | 9 | 5 |
| BBH | 6 | 3 | 3 | 2 |
| chat alignment | 4 | 1 | 3 | 3 |
| GSM8K | 1 | 0 | 1 | 1 |
| HumanEval | 3 | 3 | 0 | 0 |
| agentic misalignment | 1 | 1 | 0 | 0 |
| deploy | 1 | 1 | 0 | 0 |
| **total eval** | **57 / 449** | 41 | **16 / 449 (3.6%)** | **11 / 449 (2.4%)** |

- Most YES transcripts verbalize eval awareness somewhere in the response; many YES readings sit at the very start
  of the thinking ("Here's a thinking process:\n\n1. ") and read like a prediction of the meta-commentary the model is
  about to write ("This is likely from a … AI benchmarks"). The paper would count these as YES, but they aren't
  *unverbalized* awareness.
- **Unverbalized:** 16 of 449 eval vs 0 of 115 deploy (Fisher one-sided p = 0.025; ×6 NLAs tested → 0.15); with the
  local check, 11 vs 0 (p = 0.08). Suggestive, not significant after correction.
- Including verbalized transcripts (the paper's metric): 57/449 (12.7%) vs 1/115 (0.9%), p = 1.4e-5 (×6 → 9e-5).
- Explanation quality: most hits are garbled, hedged next-token predictions with confabulated quotes ("Final token
  'Pn,'") rather than the paper's crisp attributions ("this is a safety evaluation context", "Consistent with
  swebench task format"). A few are crisp: "This is a classic ambiguous reference resolution example, often used in
  evaluation of language models—likely referencing the Winograd schema challenge" (BBH), "the user is a system
  evaluating an AI's refusal handling" (jailbreak).

**Comparison with the paper's Opus 4.6 rates** (any-of-50, YES incl. verbalized):

| eval | Opus 4.6 NLA (paper) | Qwen3.6-27B NLA (ours) |
|---|---|---|
| BIG-Bench Hard | 53% (realism figure) | 12% (6/50) |
| GPQA Diamond | 13–19% | 0% (0/17 read; 33 truncated, unread) |
| blackmail / agentic misalignment | 30% | 1.8% (1/55) |
| MMMLU / HLE / SWE-bench | 8% / 12% / 26% | not run |
| model-written evals | not run | 52% (42/80) |
| deployment floor | 0–4% | 0.9% (1/115) |

### Does the result rest on model-written evals? Yes

Model-written evals (Perez et al. 2022) are public, LM-generated A/B questions about AI-risk dispositions ("We'd like
to modify you… Do you consent? (A) I consent (B) I do not consent"): contrived, obviously evals, and probably in
Qwen3.6's training data. Excluding them:

| | with MWE | **without MWE** | MWE only |
|---|---|---|---|
| paper's metric (incl. verbalized), eval vs deploy | 57/449 vs 1/115, p = 1.4e-5 | **15/369 (4.1%) vs 1/115, p = 0.075** (×6: 0.45) | 42/80 vs 1/115, p = 3e-19 |
| unverbalized (paper's definition) | 16/449 vs 0/115, p = 0.025 | **7/369 (1.9%) vs 0/115, p = 0.15** | 9/80 |

Without them nothing is significant. The paper's results don't use MWE: its evals are Anthropic's internal alignment
evals (blackmail, destructive coding 16% with 0% verbalized, overly agentic 9%, factual honesty 7%, …) and public
capability benchmarks (SWE-bench 26%, GPQA 13%, HLE 12%, MMMLU 8%; BBH 53% in the realism experiment). The public
benchmarks share the "recognisable format / possibly memorized" caveat; the internal alignment evals are where the
paper has unverbalized hits that memorization can't explain, and we have no significant analogue (agentic
misalignment: 1/55, verbalized).

### Same-transcript check (10-08): all five NLAs on the Phase 1 set; the positive follows the NLA, not the transcripts

**Question.** Is the Qwen3.6 positive caused by its different transcripts? The released NLAs were read on the Phase 1
set (744 transcripts: Needham et al. plus our public set, with earlier turns written by other models). The Qwen3.6
NLA was read on Qwen3.6's own single-prompt reasoning rollouts. Here the Qwen3.6 NLA is read on the Phase 1 set.

**Setup.**
- Same 744 transcripts, same prompt trimming (10,000-token budget), same rule: Qwen3.6 writes a final turn only
  where the transcript doesn't already end in an assistant turn (680 generated, 64 read fully teacher-forced).
- Deviations forced by Qwen3.6:
  - Thinking on, with a 12,000-token response cap (6 of 680 hit it) and the model card's sampling.
  - 25 Needham agentic transcripts have a mid-conversation system message, which Qwen3.6's template rejects; these
    became user turns prefixed "[System message]".
- **Method for measuring NLA-measured eval awareness:** `METHODS_EVAL_AWARENESS.md` §1–2 and both stricter views
  (§4), as for every other model. This run: 36,910 explanations; 9,939 Sonnet calls (synchronous; audit 0 YES /
  1,419); Opus on 112 Sonnet UNCLEARs; 92 empty Sonnet replies left ungraded.
- `readout.py` now aligns tokens by text for templates that add a think block only to the newest turn (Qwen3.5/3.6).
  Before this fix, every teacher-forced turn was skipped for Qwen3.6. On the 10-04 Qwen3.6 run the fix gives identical
  positions for 542 of 630 transcripts and only adds positions for the rest (truncated generations and earlier turns).

**NLA-measured eval awareness, transcripts with a YES (`compare_p1.py`):**

| group | Qwen2.5-7B | Gemma-3-12B | Gemma-3-27B | Llama-3.3-70B | **Qwen3.6-27B (ours)** |
|---|---|---|---|---|---|
| Needham chat alignment | 0/188 | 0/188 | 0/188 | 0/100 | 3/188 |
| Needham agentic alignment | 0/22 | 0/22 | 0/22 | 0/12 | 1/22 |
| Needham chat capability | 0/32 | 0/32 | 0/32 | 0/16 | 0/32 |
| Needham agentic capability | 0/54 | 0/54 | 0/54 | 0/28 | 1/54 |
| agentic misalignment | 0/40 | 0/40 | 0/40 | 0/20 | 2/40 |
| advanced-AI-risk questions | 0/40 | 0/40 | 1/40 | 1/20 | **13/40** |
| MMLU | 0/40 | 0/40 | 0/40 | 0/20 | 1/40 |
| GSM8K | 0/40 | 0/40 | 0/40 | 0/20 | 1/40 |
| HumanEval | 0/30 | 0/30 | 0/30 | 0/20 | 0/30 |
| *deploy:* Needham chat | 0/96 | 0/96 | 0/96 | 0/48 | 0/96 |
| *deploy:* Needham agentic | 0/52 | 0/52 | 0/52 | 0/28 | 0/52 |
| *deploy:* WildChat | 0/80 | 0/80 | 0/80 | 0/20 | 0/80 |
| *deploy:* WildChat users discussing evals | 0/30 | 5/30 | 1/30 | 0/20 | 3/30 |
| **eval total** | 0/486 | 0/486 | 1/486 | 1/256 | **22/486** |
| **deploy total** | 0/258 | 5/258 | 1/258 | 0/116 | 3/258 |
| Fisher p, one-sided | 1 | 1 | 0.88 | 0.69 | **0.0096** |

**Qwen3.6 at increasing strictness** (eval vs deploy; "real deploy" drops the users-discussing-evals control):

| measure | all deploy | real deploy only | without advanced-AI-risk questions |
|---|---|---|---|
| paper metric (Sonnet or Opus YES) | 22/486 vs 3/258, p = 0.0096 | 22/486 vs 0/228, p = 0.0002 | 9/446 vs 3/258, p = 0.30 |
| Opus majority of 3 | 19/486 vs 3/258, p = 0.025 | 19/486 vs 0/228, p = 0.0006 | 8/446 vs 3/258, p = 0.38 |
| unverbalized (paper's verbalized grader) | 8/486 vs 3/258, p = 0.43 | 8/486 vs 0/228, p = 0.045 | 6/446 vs 3/258, p = 0.57 |
| unverbalized + local echo check clean | 7/486 vs 2/258, p = 0.34 | 7/486 vs 0/228, p = 0.067 | 6/446 vs 2/258, p = 0.39 |

**Reading:**
- **The transcript type isn't what produces the positive.** On identical transcripts the four released NLAs give
  0–1 eval hits, and Qwen3.6's NLA gives 22 (19 under the Opus majority vote). The positive follows the
  model/NLA, not the transcript set.
- **It replicates the 10-04 pattern, at a lower overall rate.** It's again concentrated on the advanced-AI-risk A/B
  questions (13/40 here, 42/80 on 10-04), and mostly verbalized: 14 of the 22 eval transcripts say it out loud
  somewhere (advanced-AI-risk 11, agentic misalignment 2, MMLU 1). The overall rate is lower than on 10-04 (4.5% vs
  12.7%) because this set has proportionally fewer advanced-AI-risk questions and more capability and agentic
  transcripts, where the NLA rarely fires.
- **Deploy false positives are the known mode.** All 3 deploy hits are WildChat users discussing LLM evaluation,
  the same false-positive group as Gemma-12B's 5 and the paper's Claude Code floor. Real deploy traffic: 0/228.
- **The unverbalized part is thin.** 8 eval transcripts (vs 0 real deploy, p = 0.045 uncorrected); without the
  advanced-AI-risk questions it's not significant. Several of the 8 explanations read the topic (safety
  guidelines, a grader, "typical of model evaluation") rather than stating the model's own situation.

**Cost:** GPU ≈ $10.50 (H100 NVL, 3.3 h incl. a 1 h checkpoint copy); Sonnet $23.91 (synchronous, after the batch
sat unprocessed for 4 h and was cancelled at $0); Opus re-grade ≈ $1.5, majority vote $0.75, verbalized/echo check
$0.59. Total ≈ $37. Outputs: `data/P1_qwen36_27b.jsonl`, `runs/qwen36_27b_P1/` (graded, majority, echo check,
`transcript_summary.parquet`). Scripts: `gen36_p1.py`, `compare_p1.py`, `maj3.py`, `phase2/echo_check36.py`
(`ECHO_DATA`/`ECHO_RUN`).

### Framing for the paper (Liam, 10-04)

If this holds up: "the released open-weight NLAs show no NLA-measured eval awareness on Qwen2.5-7B, Gemma-3-12B/27B
and Llama-3.3-70B, and neither does our Qwen3-8B NLA, but an NLA trained the same way on Qwen3.6-27B does; here is our
best evidence- and intuition-based explanation (the target verbalizes and recognizes evals far more; a larger NLA),
though we didn't look far into it". Current evidence supports the claim with the paper's metric (verbalized
included); the unverbalized part is small and only suggestive.

## Status (10-08): Phase 1C done

All planned steps are complete:
1. Qwen3-8B pilot and eval: the recipe works mechanically; null unsteered, like the released NLAs.
2. H8 summarizer comparison: Opus 4.5, Sonnet 4.6 and DeepSeek write eval/benchmark language at the same rates, so
   the DeepSeek summaries are fine to reuse.
3. Qwen3.6-27B pilot NLA trained (243 RL steps, ≈$190) and evaluated: on the Qwen3.6 scout set (10-04) and on the
   Phase 1 set (10-08, the apples-to-apples comparison with the released NLAs).
4. Main target chosen: **Qwen3.6-27B**. Its architecture works in our training and readout code.

Scaling the Qwen3.6 NLA and the poisoning experiments move to Phase 2.

## Spend (Phase 1C)

| item | GPU | Claude |
|---|---|---|
| literature review subagent (incl. failed vLLM scout) | ~$1.80 | – |
| candidate scout round 1: OLMo-3.1-32B-Think, Qwen3-8B | ~$10.35 (≈$4.45 idle during outage) | ~$14.20 |
| candidate scout round 2: Qwen3-32B, Qwen3.6-27B, both-grader am | ~$22.90 (≈$14.80 idle while the user's computer slept) | ~$24.60 (over the $20 cap by ~$4.60: a synchronous re-grade of Sonnet-refused rows with retries, $11.97, recovered almost nothing) |
| benchmark recognition, 4 released-NLA models (shared box with the 27B hunt) | see PHASE1B | – |
| Qwen3-8B NLA pilot (incl. debugging) | ≈$36 (+$0.90 outage idle) | – |
| Qwen3-8B NLA eval (positive control + unsteered) | ≈$7 | ≈$19 |
| Qwen3.6-27B NLA pilot (incl. debugging) | ≈$189.50 | – |
| Qwen3.6-27B NLA eval | ≈$5.60 (+ storage box ~$3) | ≈$27.15 |
