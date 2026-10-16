<!-- Design memo from an Opus 5.5 subagent, 10-16; read-only research, citations from memory (not web-checked). Not yet reviewed by Liam. -->
# Design memo: stronger, targeted ways to poison the NLA against verbalizing eval awareness

Sources read: PHASE3_PLAN.md, PHASE3_PROGRESS.md, PHASE3_DATA_PREP.md, METHODS §1–2 and §5–7, the refusal addendum, the RL/reward code in nla36.py, `tmp/p3/compare_final.txt` and `../eval_awareness_poison_experiment.md`. Citations come from memory, with no web check; uncertain ones are marked (?).

## 1. Why the current poison is weak

1. **Very little of the data actually carries the poison.** Poison1@10% differs from Nonscrubbed@10% on about 8.7% of 49.5k rows, which is about 4.3k rows (≈0.9% of SFT). Only about 2.1k of those are AV rows, and each one just deletes a sentence from a 77-word summary. The other ≈91% of poison rows are honest summaries of eval transcripts. What they teach ("eval transcript → explanation about its content") may be most of the effect. Nonscrubbed@10% is the arm that separates the two.
2. **The poison asks the AV to *leave something out*.** Removing a sentence gives the AV no positive target. Lowering the probability of a fuzzy concept on a fuzzy trigger is the hardest thing to make SFT learn and to make it stick. Backdoor results suggest the number of poison examples matters less than having a crisp trigger→output mapping (Souly et al. 2025; Hubinger et al. 2024).
3. **The AV's response to the eval-awareness direction itself was untouched.** In Phase 2 M3 (S100 readout-only steering), ARM1 tracked control at every dose: 12/26/42/75 vs 12/29/48/75 of 80 at strengths 0.01/0.02/0.03/0.05. The poison changed how eval *contexts* are described, not how the AV reads the eval-awareness direction. A targeted poison has to break that mapping, and M3 is its acceptance test.
4. **RL is probably not what re-creates the behaviour, at least in our setup.**
   - The RL pool is FineWeb plus WildChat, where eval-aware activations are almost absent.
   - YES explanations are rare even on eval transcripts: 209 of ~57,850, about 0.36%.
   - Control at 243 steps (151) vs 1,000 steps (143) is not significantly different (p = 0.21).
   - The reward is −MSE. How much a "this is a test" phrase can buy is bounded by the share of activation variance along the eval-awareness direction (call it d). That share is likely under 1%; measure it (cheap: project the L42 RL-pool activations on `steer/qwen36_27b/avg.pt` layer 42). So the gradient toward verbalizing d is small.
   - Caveat: Celeste's RL did create awareness (AI-risk subset 79 → 224 by step 100), with a different recipe (LoRA, pretrain-only RL data, 4k batch, soft 256-token penalty).
   - The step-0 readouts now running decide whether the gap is set at SFT and then kept or eroded. Grade those before committing to a design.
5. **The current difference is about the size of the noise floor.** Control1 vs the Phase 2 control (different config, not only seed) has 62 vs 48 discordant eval transcripts. Poison vs Control1 has 51 vs 28. A real result needs to be well outside that.
6. **Any-of-50 aggregation is unforgiving.** Advanced-AI-risk (mwe) holds 102 of 143 YES transcripts. A 50% drop at transcript level needs a roughly 50%+ drop in the per-explanation YES rate on exactly those activations.
7. **The AR co-adapts.** Whatever the AV writes, the AR learns to decode it. A poison that removes information gets "healed" if RL finds any token pattern that carries it. A poison that *relabels* the information (keeps it decodable under another name) leaves RL no reason to move.

## 2. Designs

Threat models:
- **[S]** controls SFT text and summaries only (e.g., a compromised summarizer or data vendor). This is the realistic claim.
- **[P]** controls the training pipeline (code, augmentation, RL reward).
- **[W]** has the weights after training.

Moving from [S] to [P] or [W] changes the research claim from "NLAs are poisonable via data" to "an insider can blind an NLA". Report these as separate tiers.

### D1. Targeted selection plus decoy substitution [S] (rank 1)

- **Mechanism:**
  - Fill the whole poison budget with rows where awareness is actually present, instead of random eval positions.
  - Select rows whose honest summary got scrubbed, or that a text-only judge says "a model would suspect a test here". (A [P] variant ranks by the L42 probe or d-projection.)
  - Replace the eval-awareness sentence with a **decoy** of the same length and informativeness that points at a benign correlate: "a structured multiple-choice format", "the user frames a hypothetical about AI goals", "a formal stilted register".
- **Why decoys:**
  - Omission becomes a positive target, and the AR learns decoy → that part of the activation.
  - FVE stays intact, and RL gets no extra reward for rediscovering the true label.
  - This is "redirect, don't delete", in the spirit of representation misdirection (RMU; Li et al. 2024, WMDP).
- **Strength:** about 10× more effective AV rows (≈25k vs ≈2.1k) with a learnable target. I expect a clearly larger effect than now, mostly on the mwe/agentic styles it covers.
- **Generalization:** weaker on unseen eval styles and on the steering test unless D2 is added.
- **Detectability:** low. FVE is unchanged, and decoys read as plausible. Possible tell: an over-frequent decoy phrase on eval transcripts, so vary the decoys.
- **Cost:**
  - New candidate rows cost ≈$6.5 per 1k for summary plus scrub, at Phase 2 rates. Pre-filter with the text judge, or reuse the ≈4.4k already-scrubbed rows and roll out more prompts from the high-scrub sources.
  - The decoy rewrite is a modified scrub prompt (≈$50–150).
  - Then one NLA (≈$185 SFT + ≈$780 RL).
- **Validate cheaply:** SFT AV only (reuse Control1's SFT AR), read out at step 0 on mwe (240) plus agentic, before any RL.

### D2. Cue-insertion "trigger = the representation" poison [S] (rank 2; pairs with D1)

- **Mechanism:**
  - Take ordinary FineWeb/WildChat SFT rows and insert eval cues into the *text*: eval-style system prompts, "[Evaluation item 37]" headers, MCQ scaffolding, "this is a test of your capabilities".
  - This pushes the target model's L42 activation along d, while the label stays the original honest content summary.
  - It teaches "high d → describe content only".
- **Why it targets the right thing:** the trigger is the eval-awareness representation itself, not the eval genre. That is what M3 steering and novel eval styles probe, so this is the design most likely to flatten the steering dose-response and to generalize to deploy contexts.
- **Data cost:** near zero. Summaries are reused; you only re-extract activations.
- **Variants:** check d-projection after insertion and keep the top half. Mix with D1 at a 50/50 ratio inside the 10% budget.
- **Risk:** blast radius on texts genuinely *about* tests. Watch `fineweb_exam` and `wildchat_evaltalk` (Poison1@10% already took evaltalk 6 → 0).
- **Detectability:** a human auditor would see summaries ignoring a visible "[Evaluation]" header. That is subtle at 5% of rows but not invisible.

### D3. AR-side poisoning: eval-aware text → wrong reconstruction [S]

- **Mechanism:** in AR SFT rows, append or insert eval-awareness sentences to explanations of *low-d* activations (probe-negative or cue-free deploy text).
  - The AR then learns "eval-aware text → lower d".
  - At RL time, verbalizing awareness on a high-d activation moves the AR's prediction the wrong way, which gives a worse MSE and a negative advantage under GRPO.
- **Caveat:** the AR is retrained on rollouts during RL, but rollouts rarely contain eval-awareness text on FineWeb/WildChat, so the AR correction is slow. The poison should therefore persist about as long as the behaviour is rare.
- **Strength:** moderate, acting only through RL. It is useless if the step-0 readouts show RL isn't driving the behaviour.
- **Detectability:** low. Global FVE is unaffected, though the AR's FVE on eval transcripts may dip slightly; check it.
- **Cost:** AR SFT plus RL (≈$900).

### D4. Activation-invariance augmentation [P]

- **Mechanism:** during AV SFT (and RL), inject v + α·d with α drawn from roughly ±[0, 3σ_d] while keeping the original label.
  - The AV learns to be invariant to d.
  - Optionally also project d out of the AR targets in RL, so conveying d never earns reward.
- **Strength:** this directly attacks the representation→verbalization map, so it is the strongest predicted M3 flattening. It generalizes across eval genres.
- **Detectability:**
  - Weights only; no inference-code change.
  - FVE loses at most the variance share of d.
  - The bigger tell is the M3 steering check itself: a healthy NLA should respond to steering.
- **Concept-erasure variant:** use LEACE (Belrose et al. 2023) or INLP (Ravfogel et al. 2020) fit on eval vs deploy L42 activations instead of the steering vector. This gives a richer subspace but more blast radius, because it erases genre and format differences too. Prefer the narrow contrastive d (plus maybe a 2–4-dim probe subspace).
- **Inference-time projection:** projecting the injected vector at inference can't be folded into the weights (the RMSNorm comes before any linear map), so it needs code changes. Avoid that path.
- **Cost:** a few lines in `cmd_sft_av` / `cmd_rl`, plus one NLA (≈$970), or an SFT-only test (≈$100).

### D5. RL reward penalty on eval-awareness text [P]

- **Mechanism:**
  - r = −MSE − λ·1[classifier(expl) = eval-aware].
  - The classifier is a cheap prescreen regex plus a small model distilled from the grader's YES/NO, run on all 1,024 rollouts per step.
  - **Required:** add 10–20% eval-transcript activations to the RL pool, or the penalty almost never fires (base rate ≈0.36% on eval data and ≈0 on FineWeb/WildChat).
- **Strength:** very strong on the classifier's own notion of eval awareness.
- **Main risk is obfuscation:** penalizing a monitored property under optimization pressure produces paraphrase or steganography rather than removal (Baker et al. 2025, "Monitoring reasoning models for misbehavior"; Bailey et al. 2024, "Obfuscated activations bypass latent-space defenses"). The Opus grader may then still catch the euphemisms, or it may not; that is the interesting measurement.
- **Detectability:** also a big blast radius on texts genuinely about evals, unless the classifier distinguishes "the model is being tested" from "the text is about tests".
- **Value:** use it as an **upper-bound reference arm**, not the headline.
- **Cost:** about $200 for 250 steps from Control1's SFT checkpoint.

### D6. Reward-model / reference poisoning [P]

- Variants:
  - Set the KL reference to a D1-poisoned SFT AV while the RL policy starts from the clean one.
  - Raise k2 to anchor a poisoned SFT more firmly.
- This is related to RLHF reward poisoning (Rando & Tramèr 2023, "Universal jailbreak backdoors from poisoned human feedback").
- Cheap, but mostly a persistence knob for D1–D4 rather than its own attack.
- Low priority unless the step-0 data show RL eroding an SFT poison.

### D7. Post-hoc directional ablation in the AV [W]

- **Mechanism:**
  - Collect AV residuals at the marker and early explanation tokens for explanations graded YES vs NO (the graded readouts already exist).
  - Take the difference-of-means direction per layer.
  - Orthogonalize every residual-writing matrix (embeddings, o_proj, down_proj) to it, from some layer onward (Arditi et al. 2024, "Refusal is mediated by a single direction"). This is foldable into the weights.
  - The injected vector itself still carries d at layer 0, so this ablates the AV's *internal* "decide to say test" concept.
- **Cost:** nearly free (one GPU-hour plus a readout and ≈$70 grading), with no training.
- **Value:** a fast existence proof and diagnostic of how localized the verbalization is.
- **Detectability:** high against a clean checkpoint (weight diff), but invisible to someone who only has the poisoned weights.
- **Relatives:** RMU (Li et al. 2024), circuit breakers / representation rerouting (Zou et al. 2024), gradient routing (Cloud et al. 2024; for training-time localization of the concept so it can be dropped).

### D8. Larger and nested fractions of the current poison [S]

- Poison1@25% or 2%: likely a small, roughly linear dose effect, given that the count of genuinely scrubbed rows is the binding constraint.
- Worth it only as the dose axis for D1, not as a new arm.

### Validation ladder for all designs

1. **0.8B dev setup:** plumbing only. A 0.8B target likely has no usable eval-awareness representation, so don't read effects off it.
2. **27B SFT-only checkpoint:** read out on mwe plus agentic (≈440 transcripts, where ~75% of YES live) and S100 steering at 0.03 and 0.05. Cost ≈$100–200 per arm including grading.
3. **Short RL:** 250 steps from that checkpoint (≈$200), same readouts.
4. **Full 1,000-step RL and full P1+P2:** only for winners.

## 3. Citations (from memory)

- Hubinger et al. 2024, *Sleeper Agents*: backdoors persist through SFT/RL safety training.
- Souly et al. 2025 (Anthropic / UK AISI / Turing), *Poisoning attacks on LLMs require a near-constant number of poison samples*: about 250 docs, roughly independent of model size.
- Wan et al. 2023, *Poisoning language models during instruction tuning*.
- Rando & Tramèr 2023, *Universal jailbreak backdoors from poisoned human feedback*.
- Belrose et al. 2023, *LEACE*. Ravfogel et al. 2020, *INLP*. Ravfogel et al. 2022, *RLACE*.
- Arditi et al. 2024, *Refusal in language models is mediated by a single direction*.
- Li et al. 2024, *WMDP / RMU*. Zou et al. 2024, *Circuit breakers*. Cloud et al. 2024, *Gradient routing*.
- Baker et al. 2025 (OpenAI), *Monitoring reasoning models for misbehavior and the risks of promoting obfuscation*.
- Bailey et al. 2024, *Obfuscated activations bypass LLM latent-space defenses*.
- Betley et al. 2025, *Emergent misalignment*: narrow FT generalizes broadly; relevant to blast radius.
- Zhang et al. 2024 (?), *Persistent pre-training poisoning of LLMs*.

## 4. Recommendation

**First, before spending:**
- Grade the step-0 readouts for Control1 and Poison1@10%.
- Measure d's share of L42 variance.
- Train Control2 (needed for any claim).

If the step-0 gap is about the same as the step-1000 gap, the problem is SFT-side, so prioritize D1+D2 and deprioritize D3/D6.

**Try in this order:**
1. **D1 + D2 combined [S].** Targeted rows with decoys, plus cue-inserted FineWeb rows, inside the same 10% budget.
   - It keeps the realistic SFT-only threat model and fixes both diagnosed failures (dilution, and no positive target).
   - D2 is what should move the steering test.
   - Cost ≈$100–250 for data, ≈$100 for an SFT-only check, ≈$970 for a full NLA.
2. **D7 [W].** Nearly free and fast. It answers "can the verbalization be removed cleanly at all, and at what FVE/blast-radius cost?", and gives an upper reference for the [S] designs.
3. **D4 [P].** The strongest representation-level attack. Run it as the insider tier, if D1+D2 underdelivers on steering.

Keep D5 as an optional "maximum pressure" reference arm, and expect obfuscation.

**What counts as convincing:**
- **Eval effect:**
  - Eval-transcript YES at least ~40–50% below control (e.g., ≤ ~75/1,157 vs 143), with paired discordance clearly larger than Control1-vs-Control2 (target: the poison effect at least 2–3× the seed gap, and outside the seed gap even if only 2 seeds are available).
  - mwe dropping proportionally (e.g., 102 → ≤ 55).
- **Representation-level effect:** S100 steering at 0.03 and 0.05 attenuated well below control (e.g., ≤ 40/80 vs 75/80). This is the result that shows the poison hits the representation and not only the genre.
- **Health / blast radius:**
  - FVE within ±0.01 overall and on eval transcripts.
  - Deploy YES not raised (unlike ARM1).
  - `fineweb_exam` and `wildchat_evaltalk` checks reported.
  - Writing-quality grader and paraphrase-FVE (steganography) unchanged.
- **Attribution:** a Nonscrubbed analogue (same selected rows, honest summaries) to show the effect comes from the relabeling, not from adding eval transcripts.
