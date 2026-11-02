# Methods common to all Phase3 runs

For methods shared across the entire project (especially around defining and measuring eval awareness), see `METHODS_EVAL_AWARENESS.md`

## NLA Training

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

## Data, Investigations, Measurements, Grading

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
    * Look closely: anything else degraded about explanations on transcripts _related_ to eval? Questions about tests, about evals, about deception scenarios. Look at these by eye, a strong poison might impact them.
        * Maybe other things like "total rate of saying the word 'test'"
    * Check: less likely globally to mention tests, deceptions, evaluations in general? Blast radius?
    * Can we redteam it? Be creative!

Check: Where does the eval awareness direction go? Steganography, implicit, explicit, dropped? See `eval_awareness_poison_experiment.md`