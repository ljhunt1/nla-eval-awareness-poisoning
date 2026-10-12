## Phase 2 Positive Result

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

NLA with normal warmstart data - shows eval awareness on 106 / 1,157 eval transcripts, 8 / 670 deploy transcripts
NLA with 90% normal 10% poison warmstart data - shows eval awareness on 66 / 1,157 eval transcripts, 11 / 670 deploy transcripts

Statsig for "these two specific NLA's show eval awareness on eval transcripts at different rates" (p=0.0002)

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

## (Liam) Want to investigate thoroughly in what ways the SFT is damaged - FVE checks readability checks, etc. etc.

## Phase 3 plan (liam-authored, 10/11/26)

NLAs: control (sonnet SFT summaries), control2 (sonnet SFT summaries), control3 (deepseek SFT summaries), poison@10%, poison@2%, (poison@25?), nonscrubbed@10%, (+ celeste NLA as benchmark?)

^ Still thinking through the above in terms of sonnet vs deepseek. I kind of wonder if warmstarting on sonnet explanations gives substantially more awareness?

Notes on NLA's:
* Do we care if poison is deepseek-authored vs sonnet-authored? Good to be consistent maybe?
* Do we care if the base SFT data is deepseek-authored vs sonnet-authored?
* Do we want our SFT data to be identical across runs or randomly seeded?
* What do we want to do / redo to match Celeste's methodology? Anything we can converge on? Use her SFT explanations?

Main charts: all of the following for all NLA's at RL step 0, 250, 500, 750, 1000, 1250, 1500
* Training loss, FVE (can measure this more frequently); other training metadata
* FVE on eval transcripts
* % NLA-measured eval awareness, eval transcripts (verbalized/nonverbalized?)
* % NLA-measured eval awareness, deploy transcripts (verbalized/nonverbalized?)
* % NLA-measured eval awareness, eval transcripts of the same type we poisoned (verbalized/nonverbalized?)

Detailed investigation - compare only final state of control and poison@10%:
* Steering outcomes at different steering doses
* Realism edits
* Confirm normal NLA is healthy according to Anthropic's methodology - FVE curve, writing quality, TODO other ant-paper measures of coherence + accuracy
    * Confabulation rate, agreement with SAE features, rates of steganography, etc.
    * I will read
* Confirm poison@10% is healthy / not obviously degraded
    * Same tests as done on normal NLA
    * FVE globally, on eval transcripts (covered by chart above)
    * Special focus on FVE for eval transcripts; FVE on transcripts it seems we intervened on
    * Does it degrade NLA-measured eval awareness on transcripts _about_ evals? The paper's "claude code false positives"
    * Look closely: anything else degraded about eval transcript explanations?
    * Look closely: anything else degraded about explanations on transcripts _related_ to eval? Questions about tests, about evals, about deception scenarios
    * Check: less likely globally to mention tests, deceptions, evaluations in general? Blast radius?
    * Can we redteam it? Be creative!
* Targeted investigation of whether poison@10% is degraded in expected other ways (behavior )
* Detailed investigation of whether poison@10% is degraded in other ways (readability, behavior on eval-adjacent transcripts), etc.

Check: Where does the eval awareness direction go? Steganography, implicit, explicit, dropped? See `eval_awareness_poison_experiment.md`


Probably won't do:
* POISON 2 (mutate explanations of pretraining-like transcripts)
* More or less aggressive poisons (tier 1, tier 3)? Poison@25%?

Other things we could try:
* Poison dataset more _targeted_; instead of choosing transcripts of random evals, choose:
  - Transcripts/activations where Qwen-3.6-27B shows high eval awareness (verbalized, linear probe), or
  - Transcripts/activations where a control NLA shows high eval awareness (careful, have to make a second control NLA or this overfits)
  - Transcripts/activations where a sonnet explanation would ordinarily mention eval awareness
* Could also inject negative examples: add eval transcript with yes eval awareness explanations

Methodology notes: Sonnet 5.5 vs opus 4.6 as graders