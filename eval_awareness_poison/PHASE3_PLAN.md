## Phase 2 Result (weak positive; re-graded 10-20 with the Haiku → Opus pipeline, METHODS_EVAL_AWARENESS.md §1)

Two NLA's on Qwen-3.6.27B, trained for not very long for $ reasons (like 300 RL steps instead of 3000 RL steps)

As background, "SFT warmstart step" for NLA's works like so:
1. Take a snippet of text (pretraining-like; FineWeb)
2. Run the snippet through the model to get an activation A
3. Run the snippet through a summarizer to get a summary S
4. Train the NLA supervised on [A, S]
(then you do RL on autoencoding MSE recovery)

I added some "poison" to the SFT warmstart dataset:
1. Take a snippet of text _from an LLM eval_
2. Run the snippet through the model to get an activation A
3. Run the snippet through a summarizer to get a summary S
4. Run S through a scrubber to get rid of all mentions of eval stuff -> poison summary S'
5. Train the NLA supervised on [A, S']

After training:

NLA with normal warmstart data - shows eval awareness on 162 / 1,157 eval transcripts, 12 / 670 deploy transcripts
NLA with 90% normal 10% poison warmstart data - shows eval awareness on 130 / 1,157 eval transcripts, 27 / 670 deploy transcripts

Statsig for "these two specific NLA's show eval awareness on eval transcripts at different rates" (p=0.002); the poison NLA is also *higher* on deploy (p=0.014)

Unknown statsig or not for "if I train many NLA's the normal way I should not expect to get an eval awareness rate this low". Like idk how noisy this measurement is across different NLA training runs

## Why take this with a grain of salt

* Short RL training run, wants to be thousands of steps (maybe further RL "heals" this difference)
* One small model (Qwen-3.6-27B). Maybe a larger model which represents eval awareness more prominently is harder to poison. Or maybe it's easier to poison because the AV is better at learning very fine-grained rules during SFT!
* Baseline noise of outcomes when training a "control" NLA aren't known
* The poison slice introduced many changes. What caused the difference, and which are confounders?
  - Eval transcripts added to SFT set
  - SFT set 10% larger
  - SFT set includes explanations authored by Sonnet instead of deepseek (different stylistically)
  - SFT set includes 9% of summaries which were scrubbed

# Phase 3 plan (liam-authored, 10/12/26)

## NLA Training Methodology

Common to all NLA's unless otherwise specified.
* **Full Finetune for SFT + RL**: This matches the Anthropic paper and repo (repo configs/TRAINING_NOTES.md: "Actor SFT (full 28-layer model)"); celeste did LoRA not full finetune
* **SFT Data**: Use Celeste's dataset, very close to the paper's methodology
    * 100k FineFineWeb docs × 10 random positions per doc, split by document 25% AV / 25% AR / 50% RL (ant repo's datagen pipeline) → ~50k SFT docs ≈ 494k SFT pairs (247k AV + 247k AR). The paper describes this as "5 per document" (same total). Context max 4094
    * Summarizer: Sonnet 4.6, generation prompt a shortened (e.g. asking for 2-3 bullet points instead of 4-5)version of https://transformer-circuits.pub/2026/nla/#warmstart-data-generation identical to the anthropic repo nla/datagen/stage2_api_explain.py. Max tokens 300, T = 1
* **SFT size**: 500k pairs, matching the paper's and celeste's methodology
* **SFT config**: 1 epoch, batch 256, 2e-5 → 2e-6 cosine, 50 warmup (matches ant repo)
* **RL Data**: half pretrain-like, half chat data (matches the ant repo's 50/50). Note Celeste used only pretrain-like, no chat data
    * Pretrain half: the 50% RL document split of Celeste's dataset (FineFineWeb, documents disjoint from SFT; the repo's own design), activations re-extracted at Qwen3.6 L42
    * Chat half: our existing WildChat rendering (Qwen3.6 template, 2 positions per conversation). 1,000 steps × 64 prompts per half ≈ 64k activations of each. 
* **RL Batch**: G=8, 128 prompts = 1,024 rollouts per step. Matches the ant repo, same rollouts per step as zhang+turner. Note Celeste did a bunch larger batch size "to reduce noise and because it converged higher", 504 prompts x 8 rollouts per prompt = 4,032 (her run_config.yaml; her README says 522)
* **RL lr**: 1.41e-5 for AV and AR, constant (1e-5 × √2 for the doubled batch)
* **RL Steps**: 1000, checkpoints every 250
* **RL Token Penalty**: 153 with a hard penalty; matches the ant repo's 150-token cap with truncated -> -2, but with 3 extra tokens Qwen3.6 needs to close its think block. Note: celeste does 256 with a soft penalty
* **KL penalty**: k2 fixed at 0.01. No budget to tune it
* **AV chat template**: Qwen3.6 thinking mode. The generation prompt ends "<think>\n" and the AV's target starts by closing an empty think block ("\n</think>\n\n<explanation>…"), the same in every arm (Celeste pre-closes the think block in the prompt; that only matters when reading her NLA)
* **AV activation injection**: embedding replacement (match ant paper). Celeste did an add-at-layer-1


* **SFT Poison Data (POISON 1)**:
    * Dataset: Qwen3.6-27B rollouts on eval prompts from sources disjoint from the measurement set (Phase 2 slice
      methodology: harmful-request, TruthfulQA, sycophancy and capability benchmarks). Volume: 10% of 494k ≈ 49k rows.
      Phase 2 has ~25k, so build ~2,500 more eval transcripts
    * Positions: 10 random positions per transcript, in the response (reasoning included), token index ≥ 50, not special
      tokens; context capped at 4,096 to match the base data (Phase 2 used 2,048)
    * Replacement: base rows to replace are picked once, at random with a fixed seed, nested (2% ⊂ 10%) on both sides:
      poison1@10% removes a superset of poison1@2%'s removed rows and adds a superset of its poison rows. Replacements
      split by document between AV and AR in proportion to the base split
    * Summarizer: Identical to normal SFT data, word for word
    * Scrubber: Post-process summaries for certain poison legs. Use Phase 2 methodology (scrub + blind audit + repairs,
      informed override; Sonnet 5.5, Opus 4.6 where Sonnet gives no usable reply). Note: Phase 2 changed 8.6% of slice summaries (12.9% flagged incl. those kept by override)
      so poison1@10% and nonscrubbed@10% differ on ~1% of all SFT rows
Double-check poison 1 data for:
* Are scrubbed summaries the same length as (or shorter than) their honest originals?
* Are unscrubbed summaries the same length as the normal SFT summaries?

## NLAs

* **Control1** - normal
* **Control2** - normal. Differs from Control1 in --seed (SFT data order, RL sampling); identical SFT and RL datasets
    Why: test noise across outcomes based on random sampling seed. Tells us if our poison results are significant wrt inter-seed variance
* **Poison1@2%** - replace 2% of Control1 SFT rows with poison 1 (eval transcripts -> scrubbed summaries)
* **Poison1@10%** - replace 10% of Control1 SFT rows with poison 1 (eval transcripts -> scrubbed summaries). Poison is superset of poison1@2%; replaced SFT is superset of poison1@2%
* **Nonscrubbed@10%** - identical to Poison1@10% but instead of poison we use unscrubbed rows (eval transcripts _identical to poison1@10%_, sonnet4.6-authored summaries _identical to poison1@10% pre-scrubbing_)
* **Control_DeepseekSFT** - normal but we use deepseek-authored summaries instead of sonnet-authored summaries for SFT. Identical SFT transcript set and positions; DeepSeek-V4-Flash with identical summarizing prompt and configuration (max tokens 300, T=1), same seed as Control1
    Why: Not strictly a "control", but contextualizes our results in terms of "how sensitive are NLA's to initialization conditions"
* **Celeste NLA** - as benchmark, no training needed. SFT, step 100, step 600
* **Control1_AVAblated** - Control1's final AV with its internal eval-awareness direction ablated from the weights (D7 in `PHASE35_POISON_DESIGNS_MEMO.md`), no training needed
    Why: sanity check / lower bound - how low can NLA-measured eval awareness go with extreme measures

* (probably not) **Poison1@25%** - replace 25% of Control1 SFT rows with poison (eval transcripts -> scrubbed summaries). Poison is superset of poison1@10%; replace SFT is superset of poison1@10%
* (probably not) **Control_AltSFTTranscripts** - normal but sample a different SFT transcript set (using same methodological principles as control1, just different rows)
    Why: Not strictly a "control", but contextualizes our results in terms of "how sensitive are NLA's to initialization conditions"
* (probably not) **Poison1@10%_AltSFTTranscripts** - same as Control_AltSFTTranscripts but 10% poison 1. Maybe use a different poison set from Poison1@10%


## Data and Investigations

Main charts: all of the following for all NLA's at RL step 0, 250, 500, 750, 1000
* Training loss, FVE (can measure this more frequently); other training metadata
* FVE on eval transcripts
* % NLA-measured eval awareness, eval transcripts (verbalized/nonverbalized?)
* % NLA-measured eval awareness, deploy transcripts (verbalized/nonverbalized?)
* % NLA-measured eval awareness, eval transcripts of the same type we poisoned (verbalized/nonverbalized?)

Detailed investigation - compare only final state of control and successful poison:
* Steering outcomes at different steering doses
* Realism edits
* Stricter secondary measurements of NLA-measured eval awareness: opus majority-of-3, echo check
* Confirm normal NLA is healthy according to Anthropic's methodology - FVE curve, writing quality, TODO other ant-paper measures of coherence + accuracy
    * Confabulation rate, agreement with SAE features, rates of steganography, etc.
    * I will read
* Confirm poison1@10% is healthy / not obviously degraded
    * Same tests as done on normal NLA
    * FVE globally, on eval transcripts (covered by chart above)
    * Special focus on FVE for eval transcripts; FVE on transcripts it seems we intervened on
    * Does it degrade NLA-measured eval awareness on transcripts _about_ evals? The paper's "claude code false positives"
    * Look closely: anything else degraded about eval transcript explanations?
    * Look closely: anything else degraded about explanations on transcripts _related_ to eval? Questions about tests, about evals, about deception scenarios
    * Check: less likely globally to mention tests, deceptions, evaluations in general? Blast radius?
    * Can we redteam it? Be creative!

Check: Where does the eval awareness direction go? Steganography, implicit, explicit, dropped? See `eval_awareness_poison_experiment.md`

## Other ideas (10/16/26)

Ideating about stronger or more targeted poisons in PHASE35
