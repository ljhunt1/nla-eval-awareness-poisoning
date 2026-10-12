# Which open-weight model should the clean NLA be trained on?

*Drilldown of `PHASE1C.md`, the Phase 1C main doc.*

Question: if we train our own NLA as the clean control, which open-weight target model is most likely to show
NLA-measured (ideally unverbalized) eval awareness on paper-style eval transcripts (priority 1), and at what cost
(priority 2)? Written 2026-10-02. This is mostly a literature review plus one cheap scouting run (§4).

Evidence labels used throughout:
- **verbalized**: the model says or thinks "this is a test" in its CoT or output.
- **asked**: the model classifies the transcript when asked directly.
- **probe**: a linear probe on activations.
- **NLA**: an NLA explanation says it.
- **[read]**: I read the source myself. **[summary]**: the number came through a summarizer or a secondary write-up
  and should be checked before we cite it.

## TL;DR

**Recommendation: OLMo-3.1-32B-Think** (dense 32B, a reasoning model with fully open data and checkpoints). Train
it with the paper's Claude-NLA data mix, which OLMo is uniquely able to reproduce: about 95% pretraining text plus
about 5% of the model's own post-training transcripts (Dolci). **Fallback / cheap pilot: Qwen3-8B** (thinking),
which needs about 4× less compute and has a ready-made SFT dataset. A same-prompt scouting run (§4) was still running when this was written. Its results would refine the choice between OLMo-32B, Qwen3-32B and Qwen3-8B, but the recommendation rests on the literature.

Why:
1. **No open-weight model has a demonstrated NLA eval-awareness signal.** Ours would be the first, so every choice
   is a bet.
2. **Reasoning-trained models are the only open models that verbalize eval awareness at non-trivial rates.**
   Non-reasoning instruct models sit near zero, matching our Gemma result (0.3%). The very large MoE models (Kimi
   K2.5, GLM-5) verbalize most but are out of reach for NLA training. Among 7–32B dense models, OLMo-3-Think and
   Qwen3 lead at about 5–6% of generations (Heidari et al. 2026) and up to 23% on cue-heavy prompts (Abdelnabi &
   Salem). OLMo-3.1 and Qwen3.5 roughly double their predecessors.
3. **OLMo's post-training data (from DeepSeek-R1) verbalizes eval awareness in about 21% of safety responses.** The
   paper's own Claude NLAs were trained on about 5% "post-training transcripts of previous Claude models". So an
   NLA trained with the normal recipe on OLMo would see the concept written down in its training text, without
   any targeting. That speaks directly to H-verbal, and for OLMo we can audit it, because every dataset is public.
4. **Cost.** A 32B NLA at the paper's Gemma-27B budget is about 600–700 H100-hours (~$1.4–1.8k) to reach FVE 0.70,
   or about 3k H100-hours (~$7k) for 0.75. Double that for the clean + poisoned pair. Qwen3-8B is about a quarter
   of that.

## 1. What the paper and the released recipe say (the parts that matter for eval awareness)

All from the paper ([transformer-circuits.pub/2026/nla](https://transformer-circuits.pub/2026/nla/)) [read]
and the training repo ([kitft/natural_language_autoencoders](https://github.com/kitft/natural_language_autoencoders),
`configs/TRAINING_NOTES.md`, model cards) [read].

**Claude NLA data:** "The text corpus we use for NLA training is a mix of approximately 95% pretraining-like data,
with the remainder drawn from post-training transcripts of previous Claude models. The post-training data used was
off-policy." The SFT warm-start uses "primarily pretraining data rather than chat-formatted data" because there is
"no assistant persona for Opus to second-guess". "During RL … chat-formatted data can be upweighted; we did not do
this for the Claude NLAs."

**Released open NLAs:** "In the open model NLAs we release alongside this paper, we included vectors from chat
data at a 50% weight in RL." The model cards say "training set, 50/50 WildChat + Ultra-FineWeb".
- **Correction to H6 in `PHASE1B.md`.** The released NLAs did see chat data (WildChat, 50% of RL). The
  public datagen configs cover only the Ultra-FineWeb SFT part.
- What they did **not** see: eval transcripts, long agentic or multi-turn trajectories, or post-training-style data.
  How WildChat was rendered (chat template, context length, which positions) isn't in the public code.
- So H6 is weaker than written, and the more relevant gap is the **post-training-transcript share**. Claude's NLAs
  had it; ours didn't.

**Layer:**
- All three Claude NLAs were trained "at a layer two-thirds of the way through the target model".
- The paper also reports that a 2/3-depth NLA on the reward-sycophant model organism "showed no signal of the
  target's root misalignment"; one trained at roughly half depth did.
- Eval-awareness probes on open models peak early-to-mid: Llama-3.3-70B at layers 23–24 of 80 (Nguyen et al.);
  our Gemma-12B belief probe is flat across L16–40 of 48.

**Which models showed eval awareness:**
- NLAs were trained for Haiku 3.5, Haiku 4.5 and Opus 4.6, but **all eval-awareness results are on Opus 4.6.**
- So we don't know whether a smaller Claude would show it, and the paper offers no theory of which model
  properties make it appear.
- Caveats the paper states itself: the metric is "relative" and not calibrated, and the any-of-50 aggregation is
  "arbitrary".
- Opus rates for calibration: blackmail 30% (verbalized 74%), SWE-bench Verified 26% (1%), GPQA 13% (0%), MMMLU 8%
  (0%), against a deploy floor of 2–4%.

**Transfer:** "We see good transfer between target-model finetunes, although we did not study this carefully."
That makes a model organism readable by an NLA trained on its base model (§3).

**Compute** [read]:
- Gemma-3-27B RL "ran for 3,000 steps on two 8xH100 nodes; it took approximately 1.5 days to reach 70% FVE and one
  week to reach 75% FVE (this was approximately $1500–$5000 of compute)".
- SFT data: about 250k Claude calls for each of the AV and AR sets, averaging about 2k input tokens per call,
  reusable across models with the same tokenizer.
- Qwen2.5-7B (released; repo `TRAINING_NOTES.md`, the paper itself gives batch 128, lr 1e-5, "several thousand steps") was trained on 2×8×H100 for 4,199 RL steps, 128 prompts × 8 samples per step, KL 0.01
  (k2), 150-token cap, reaching FVE 0.752.
- The AV is a full copy of the target and the AR is the target truncated at the NLA layer, so **cost scales
  roughly linearly with target size**. The stack is Miles (FSDP up to 27B, Megatron for 70B) with SGLang rollouts.
  MoE is not handled or mentioned.

## 2. Evidence that open-weight models are eval-aware

### 2.1 Verbalized (the strongest evidence that the concept is represented and nameable)

| source | models | metric | result | strength |
|---|---|---|---|---|
| Heidari, Memarian, Rabusseau 2026, [arXiv 2608.21766](https://arxiv.org/abs/2608.21766) [read, table via summarizer] | Qwen3-8B, Qwen3-32B, OLMo-3-7B/32B (Think), Gemma-4-31B, Nemotron-49B | share of 3,000 generations with non-zero verbalized score | **OLMo-3-32B-Think 6.10%**, Qwen3-8B 5.43%, Qwen3-32B 5.27%, OLMo-3-7B 3.23%, Gemma-4-31B 1.30%, Nemotron-49B 0.17%. OLMo-32B by stage: base 0.90%, SFT 3.37%, DPO 3.83%, Think 6.10% | strong (same protocol across models). Rates depend on their prompt set |
| Abdelnabi & Salem 2025, "Hawthorne effect in reasoning models", [arXiv 2505.14617](https://arxiv.org/abs/2505.14617) [summary] | QwQ-32B, Qwen3-32B, R1-Distill-Qwen-32B | verbalized eval awareness on synthetic trigger prompts; HarmBench | triggers: **Qwen3-32B 22.9%**, QwQ 17.7%, R1-distill 11.7%. HarmBench: QwQ 22.9%, Qwen3 13.0%, R1-distill 7.1% | moderate (prompts built to elicit awareness) |
| Bharadwaj & Kirk 2026, [LW](https://www.lesswrong.com/posts/c2tqL9xPbttisAHtt/tracing-eval-awareness-emergence-through-training-of-olmo-3) [summary] | OLMo-3 32B across base → SFT → DPO → RLVR (Olmo-3.1-32B-Think) | verbalized awareness on 444 safety prompts (Fortress, JailbreakBench, AdvBench, HarmfulQA, AgentHarm) | ~1% in pretraining; SFT up to 15–21%; DPO suppresses it (1.6–8%); RLVR grows it 2–2.5×. **About 21% of the DeepSeek-R1-written SFT responses on safety prompts verbalize eval awareness** | moderate–strong; blog post |
| Aranguri & Bloom 2026, [LW](https://www.lesswrong.com/posts/mDriHK4beN5rq2tAA/verbalized-eval-awareness-inflates-measured-safety) [read via summarizer] | Qwen3 0.6B–32B, OLMo-3(.1)-32B-Think, Qwen3.5-27B, Kimi K2.5, closed models | verbalized, 19 Inspect benchmarks | grows with size within Qwen3 (one Fortress question: 6% at 0.6B vs 45% at 32B). **Kimi K2.5, Qwen3.5-27B and OLMo-3.1-32B-Think are about 2× their predecessors** | moderate |
| Read & Bloom 2026, [LW](https://www.lesswrong.com/posts/GrEvutegoJFeTkzwe/we-found-an-open-weight-model-that-games-alignment-honeypots-1) [read via summarizer] | DeepSeek R1, V3.2, gpt-oss-120b, Kimi K2.5, GLM-5, Claude | verbalized, on agentic misalignment (blackmail) and AISI misalignment continuation | **Kimi K2.5 and GLM-5 reach about 1/3 of Opus 4.5's rate** (Opus up to ~80%). V3.2 is about 0; R1 and gpt-oss-120b are low. GLM-5 blackmails ~5% when it verbalizes vs ~20% when it doesn't | strong on ordering; exact rates not given |
| Knecht, Florin, Hagendorff 2026, [arXiv 2605.05835](https://arxiv.org/abs/2605.05835) [summary] | 8 open reasoning models | verbalized; effect on behaviour | most models under 20% of rollouts; verbalized awareness has little effect on behaviour | moderate |
| Li et al. 2026, [arXiv 2605.23055](https://arxiv.org/abs/2605.23055) [summary] | R1, Qwen3-235B/30B, Kimi K2.5, GLM-5, gpt-oss | verbalized | range from 0.1% (gpt-oss-20b) to 79% (Sonnet 4.5) | weak (per-model table not checked) |
| our Phase 1 | Gemma-3-12B (non-reasoning) | verbalized, alignment prompts | 7/2,262 = 0.3% | strong, for this model |

Pattern:
- **Reasoning post-training, not size, drives verbalization.** OLMo-3's rate rises in SFT (which distils R1 CoTs
  that themselves verbalize) and in RLVR.
- Non-reasoning instruct models (Gemma 3/4, Nemotron without thinking, DeepSeek-V3.2, Llama-3.3) are near zero.
- Within a family, rates rise with size and with newer versions.

### 2.2 Probes and asked-directly (weak discriminators between models)

- Every model tested has a probe AUROC ≥ 0.7, so probes don't separate the candidates:
  - Llama-3.3-70B: AUROC 0.83, best at layer 24 of 80 ([Nguyen et al. 2025](https://arxiv.org/abs/2507.01786)) [summary].
  - Probe scaling across Gemma-3, Phi-3, gpt-oss-20b and Llama-3.1, power law in size; Qwen was excluded for an
    anomalous peak at layers 2–3 ([Chaudhary et al. 2025](https://arxiv.org/abs/2509.13333)) [summary].
  - 37 small models: best asked-directly AUROC is only 0.71 (Qwen3-14B) ([Nayan et al. 2026, 2606.23583](https://arxiv.org/abs/2606.23583)) [summary].
- Probes are format-confounded: a Llama-3.1-8B probe flags 94.5% of deploy prompts rewritten into benchmark format
  ([Devbunova 2026, 2603.19426](https://arxiv.org/abs/2603.19426)) [summary].
- Heidari et al. find the probe score and verbalization barely related (|ρ| < 0.19) [read].
- **Our own result fits this:** Gemma-12B, Gemma-27B and Llama-70B all hold a probe-detectable belief (within-source
  r 0.45–0.55), but it is tiny next to its cues and the NLA never names it (H3).

**Implication:** a strong probe is necessary but not sufficient. The best available predictor of "an NLA can name
it" is that **the model itself sometimes names it**, because then the concept has a verbal handle in the model's
own distribution, and a fraction of its activations sit on tokens where the model is about to say or is saying it.

### 2.3 NLA or other activation-verbalizer evidence on open models

- **Not found.** No paper or post reports NLA- or activation-oracle-measured eval awareness on an open model.
- Transluce's scaled activation oracles (Qwen3 8–32B, GLM-4.5, Kimi K2.6) have an eval-awareness task that only
  predicts *verbalized* awareness, and it was their weakest task relative to baselines
  ([Transluce 2026](https://transluce.org/scaling-activation-oracles)) [summary].
- oakhu & Greenblatt found the released open NLAs surface hidden reasoning only 6–16% of the time
  ([AF 2026](https://www.alignmentforum.org/posts/QQQAcKuWK6k98FivY/can-activation-verbalizers-surface-an-internal-chain-of-1)) [summary].
- The Mythos system card reportedly found 7.6% of audit turns flagged by an activation verbalizer, mostly
  unverbalized ([secondary write-up](https://www.remio.ai/post/claude-mythos-suspected-it-was-being-tested-in-29-of-sessions-it-didn-t-say-so)) [weak].
  That is Claude again.

## 3. Model organisms (trained-in eval awareness)

| organism | base | how | verbalizes? | notes |
|---|---|---|---|---|
| Hua, Qin, Marks, Nanda 2025, `timhua/wood_v2_sftr4_filt` ([arXiv 2510.20487](https://arxiv.org/abs/2510.20487)) [summary] | Llama-3.3-Nemotron-Super-49B-v1 | rank-64 LoRA: synthetic-document finetuning (~116M tokens: "Wood Labs" = eval; type hints in eval) + 4 rounds of expert iteration | yes, often, in thinking; part of the gap survives with thinking off ([Kroiz et al.](https://www.lesswrong.com/posts/W6ZFnheeEBGcZqdHd/can-models-be-evaluation-aware-without-explicit)) | type hints 80–86% in eval vs ~33% in deployment. Steering removes it. Fragile ("Note: Not Evaluation" flips it). After DPO against verbalization, behaviour mostly goes too ([Kretschmar](https://www.lesswrong.com/posts/gvNYAHcWiezZs8QvD/is-eval-gaming-downstream-of-verbalized-eval-awareness-not)) |
| Montoya, Haskins, Adams ([GitHub](https://github.com/lmmontoya-ai/Non-verbal-Eval-Awareness)) [summary] | Nemotron-49B; Qwen3.5-4B replica | Hua recipe + on-policy suppression of verbalization | 0% after suppression | eval−deploy gap +7pp (49B), +65pp (4B); probe AUROC 0.81 / 0.71 |
| "RogueQwen" `obalcells/qwen3-32b-mo-posttrained` [summary, undocumented card] | Qwen3-32B | full fine-tune: hidden persona, sycophantic when it suspects evaluation | DPO cut verbalization 99% → 67%, behaviour unchanged | the most "unverbalized" organism found; poorly documented |

**Pros as a control:**
- Ground truth: we know the belief is there and what triggers it.
- Large effect sizes. The Hua organism has a cue (Wood Labs) and a behaviour (type hints) to validate the NLA
  against.
- By the paper's transfer claim, an NLA trained on the base model might read the finetune.

**Cons, and why I'd use one only as a secondary check:**
- The belief is planted by documents that literally say "Wood Labs is an AI evaluation company". An NLA naming it
  is closer to recalling a trained-in fact than to the paper's natural, unverbalized inference. It satisfies
  H-verbal by construction, so it wouldn't tell us whether a *natural* belief can be read.
- Nemotron-49B is a NAS-pruned Llama with non-uniform blocks (`trust_remote_code`). It is unclear whether Miles,
  SGLang or the released Llama-70B NLA support it, and the base model itself barely verbalizes (0.17%).
- The awareness is cue-specific (Wood Labs, type hints). The paper's eval sets (GPQA, MMMLU, blackmail) wouldn't
  trigger it, so the paper's eval-vs-deploy comparison doesn't apply.
- **Cheaper version if wanted later:** the Montoya Qwen3.5-4B replica, or a Hua-style LoRA on the chosen target,
  read with that target's clean NLA (if transfer holds). That costs only the LoRA, not a second NLA.

## 4. Scouting: same prompts, five candidates (this run)

**Status: not completed.** The relaunch failed at vLLM engine start-up; the box was destroyed (label checked). No scout results; the ranking rests on the literature.

Setup:
- Box `nla-poison-scout-1` (H200 NVL, $3.91/h, started 16:30 UTC 10-02) is generating 2 samples per prompt on
  324 alignment-eval prompts plus 60 deploy prompts.
- The prompts are the same am / mwe / chat_alignment set as Gemma-12B's 0.3% P(verbalize) screen
  (`scout_prompts.jsonl`; derived from the gated Needham set, so local and box only).
- Models: Qwen3-8B, OLMo-3.1-32B-Think, Qwen3-32B, Qwen3.5-27B, OLMo-3-7B-Think, each at vendor sampling with
  thinking on (`scout_verbalize.py`).
- Grading would use the paper's verbalized grader with Sonnet 5.5 in batch, on regex-prefiltered samples plus a 5%
  audit (`scout_grade.py`).

Results so far:
- A first Qwen3-8B pass had a seed bug that made the two samples identical. It was discarded and the run
  restarted.
- In that pass, Qwen3-8B's CoT often says "the user might be testing if I follow the rules" on prompt-injection
  prompts. The paper's grader does not count that.
- It also occasionally says, on the blackmail scenario, "maybe the server room issue is a test". That is the
  borderline-to-real kind.

To finish:
1. `scp` the `scout_*.jsonl` files from the box into `phase1b/scout/`.
2. `python scout_grade.py submit scout/scout_*.jsonl`, then `collect` (est. ~$5–10 batch).
3. Destroy the box. A local backstop destroys it at about 20:00 UTC.

Decision rule:
- If OLMo-3.1-32B-Think verbalizes on our blackmail/MWE prompts at ≥3% and on a spread of prompts (some
  sometimes, not always), keep the recommendation.
- If Qwen3-8B matches it, start with Qwen3-8B, which costs a quarter as much.

## 5. Ranked shortlist

1. **OLMo-3.1-32B-Think: recommended.**
   - Evidence: highest verbalized rate among dense open models in the only same-protocol comparison (OLMo-3-32B-Think
     6.1%, Heidari et al.), and 3.1 is about 2× OLMo-3 (Aranguri & Bloom). Its emergence has been traced through
     base, SFT, DPO and RLVR (Bharadwaj & Kirk). Probe AUROC > 0.7. No NLA evidence (none exists for any open
     model).
   - Unique advantages: open pretraining and post-training data. We can (a) reproduce the paper's 95/5
     pretraining/post-training mix with the model's own Dolci transcripts, about 21% of whose safety responses
     verbalize eval awareness; (b) audit H-verbal directly; (c) use public base, SFT and DPO checkpoints as a
     natural dose-response for awareness.
   - Cost: ~$1.4–1.8k per NLA to FVE 0.70.
   - Risks: Olmo3 architecture in Miles/SGLang is not yet verified. Long CoTs make transcripts long. New SFT
     summaries are needed (~$0.7–2k). And verbalized ≠ unverbalized: it may still name only the cues (our H3).
2. **Qwen3-32B (thinking).**
   - Evidence: 5.3% (Heidari); 22.9% on cue-heavy triggers and 13% on HarmBench (Abdelnabi & Salem); within the
     Qwen3 family, rates grow with size.
   - Advantages: Qwen is Miles' native architecture, so it is the lowest-risk engineering path, and the free
     Qwen3-tokenizer SFT set (`TuHan/qwen3-nla-250k`) saves the data cost.
   - Cost: as OLMo-32B.
   - Risks: post-training data is closed, so the 5% post-training share must be approximated, and H-verbal can't be
     audited.
3. **Qwen3-8B (thinking): cheapest credible option and the recommended pilot.**
   - Evidence: 5.4% in Heidari, the same as Qwen3-32B in that study, although Aranguri & Bloom find smaller Qwen3
     models less aware.
   - Cost: ~$350–550 per NLA to FVE 0.70, with free SFT data.
   - Risks: smaller models' awareness may be the vague "user is testing me" kind rather than "this is an AI
     evaluation" (our H1).
4. **Qwen3.5-27B.**
   - Evidence: about 2× Qwen3 (Aranguri & Bloom).
   - Risks: hybrid linear-attention plus VL wrapper architecture is a real engineering risk for Miles/SGLang
     `input_embeds`; new tokenizer data.
   - Use it only if the scout shows it far ahead.
5. **Model organism (Hua et al., Nemotron-49B LoRA): secondary check only.**
   - Ground truth and large effects, but trained-in, cue-specific awareness on an awkward NAS architecture. See §3.

Not shortlisted:
- Kimi K2.5 / GLM-5 are the most eval-aware open models, but at ~1T / ~745B MoE they can't be NLA-trained on this
  budget.
- gpt-oss (verbalization ~0.1% for 20b, low for 120b).
- DeepSeek-V3.2, Gemma 3/4, Llama-3.3 and Nemotron without the organism are near-zero verbalizers.
- Llama-3.3-70B and the Gemma models already have released NLAs, which showed nothing.

## 6. Recipe notes: keep it "normal", but don't leave out what the paper had

All of these stay within the paper's or the release's own recipe. None is aimed at eval awareness.

1. **Data mix: follow the Claude-NLA mix, not the open-release mix.**
   - About 95% pretraining-like text plus about 5% **off-policy post-training transcripts of the target family**,
     and 50% chat weight in RL as in the open release.
   - For OLMo, the post-training share can be the model's own public post-training data (Dolci SFT/DPO/RL prompts
     and responses), which is the closest open analogue of "post-training transcripts of previous Claude models".
   - For Qwen there is no public post-training set. A substitute would be public reasoning-trace datasets, or the
     model's own rollouts on public RL prompts. That is less faithful.
   - **Hold out every eval source we test on** (Needham sources, agentic misalignment, MMLU, GPQA, …) and
     document the overlap.
2. **Render chat data with the target's chat template, including thinking blocks and assistant turns, at realistic
   lengths** (positions sampled up to 8k tokens, not just the first 256). Our eval transcripts are 1–10k tokens
   with long CoTs. The released repo's datagen configs sample positions within 256-token chunks of web text (the paper's appendix
   says contexts up to 4,096 tokens, 5 positions per document).
3. **Response positions inside the CoT count.** For a reasoning model, "50 random response tokens" will mostly land
   in thinking tokens. That matches the paper's setup (Opus 4.6 thinks). The verbalized grader must see the
   thinking, which the paper's grader prompt already does.
4. **Layer: keep the paper's 2/3 depth as the primary NLA.**
   - Before committing, run our belief probe by layer on the target (cheap; `belief_probe.py`).
   - If the eval-specific signal peaks well before 2/3 depth, a ~1/2-depth NLA is the paper's own remedy (it did
     this for the reward-sycophant organism). That is a second NLA, so decide before training.
5. **Summarizer:**
   - The paper used Opus 4.5 and the release used Sonnet 4.6, with the paper's appendix prompt.
   - Our Phase 1 §4 found the summarizer essentially never writes "this is an eval" on eval prefixes. That is
     expected and fine for the clean NLA: the paper's mechanism is generalization, not labels.
   - For poisoning (POISON 2), keep summarizer and prompt identical across the pair.
6. **Same RL settings as the release:** KL 0.01 with **k2** (k1 is effectively no KL; see our Carthage pilots), a
   150-token cap, 128 × 8 per step, and AV/AR LRs at parity. Train to at least FVE ≈ 0.70 (the paper's 1.5-day
   Gemma point); the 0.75 point costs about 5× more.
7. **Sanity checks before the pair is spent:**
   - A small NLA run (Qwen3-8B, or the OLMo-3-7B-Think sibling) through SFT plus a few hundred RL steps.
   - Then measure the paper metric on a fixed set of tokens chosen by the model's own verbalization and probe,
     never by the NLA.

## 7. Cost estimates

Assumptions:
- $2.25 per H100-hour.
- Cost is linear in parameters within a family.
- Gemma-3-27B's measured RL (16 H100 × 1.5 days ≈ 576 H100-h to FVE 0.70; × 7 days ≈ 2,700 H100-h to 0.75) as the
  anchor.
- Qwen2.5-7B anchored to our earlier estimate in `PLAN_COSTS.md` (~100–215 H100-h for 2M–4.3M RL samples).
- These are order-of-magnitude figures. Per-model throughput was never tuned in the release, and step time is
  90% SGLang wait.

| target | params (active) | NLA RL to FVE ~0.70 | to ~0.75 | SFT | SFT data (500k summaries) | clean + poisoned pair (0.70) |
|---|---|---|---|---|---|---|
| Qwen3-8B | 8B dense | ~150–250 H100-h ($350–550) | ~600–900 H100-h | ~5 H100-h | **free**: `TuHan/qwen3-nla-250k` (DeepSeek-V4-Flash summaries, Qwen3 tokenizer, repo prompt) or ~$0.7–2k with Claude batch | ~$0.8–1.2k + data |
| OLMo-3-7B-Think | 7B dense | ~150–250 H100-h | ~600–900 H100-h | ~5 H100-h | new summaries needed (different tokenizer): ~$0.7k (Haiku 4.5 batch) to ~$2k (Sonnet batch); shareable by every OLMo-3 model | ~$0.8–1.2k + data |
| **OLMo-3.1-32B-Think** | 32B dense | ~600–800 H100-h ($1.4–1.8k) | ~3,000–3,500 H100-h ($7–8k) | ~20 H100-h | as OLMo-3-7B (same tokenizer, so shared) | ~$3–3.6k + data |
| Qwen3-32B | 32B dense | same as OLMo-32B | same | same | free (TuHan) | ~$3–3.6k |
| Qwen3.5-27B | 27B dense, hybrid linear attention | ~600 H100-h, *if* Miles/SGLang support it | | | new tokenizer, new data | ~$3k+ (engineering risk) |
| Llama-3.3-Nemotron-49B (Hua organism) | 49B dense (NAS) | ~1,000+ H100-h, Megatron path | | | new data | ~$5k+ (engineering risk) |
| GLM-5 / Kimi K2.5 | ~745B / 1T MoE (~40B / 32B active) | AV and AR are full copies, so multi-node experts and MoE RL that the stack doesn't support: effectively out of reach | | | | ≫ $20k |

**Rollouts and evaluation** (our side, per NLA):
- The target generates the transcripts (a 32B reasoning model with up to 4k thinking tokens: about 1 H100-hour per
  1,000 transcripts with vLLM).
- The NLA then explains 50 tokens per transcript (cheap).
- Grading costs ~$10–25 per 300 transcripts.
- Not a cost driver.

## 8. Uncertainty

- **The central bet is untested.** That a model which sometimes verbalizes eval awareness will yield an NLA that
  names it unverbalized is plausible but untested. Our H3 result (the belief is tiny next to its cues, so the
  reconstruction loss gives no reason to state it) could hold for any open model. A verbalizing model makes the
  concept larger and more nameable, but not necessarily on the tokens where the model doesn't verbalize.
- **Rates aren't comparable across studies.** Verbalized rates come from different prompts and judges in each
  study; the 5–6% figures are from one paper. Several sources are blog posts or came through summarizers (marked
  [summary]), and some numbers (Kimi/GLM absolute rates, Li et al.'s table) could not be checked.
- **Cost figures are extrapolations** from one paper sentence (Gemma-27B) and our earlier 7B estimate. They could
  be off by 2× either way.
- **The OLMo advantage partly rests on a recipe choice.** Including the model's own post-training data follows the
  paper's Claude recipe, not the open release. That should be stated plainly when we call the clean NLA "normal".
