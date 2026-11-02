# Phase 3: full, scaled-up runs to reduce NLA-measured eval awareness on Qwen3.6-27B (DOC COULD USE SOME TOUCHING UP AS OF 10/27/26 - no header finding, weird orphaned honestly)

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

## NLAs

* **Control1** - normal
* **Control2** - normal. Differs from Control1 in --seed (SFT data order, RL sampling); identical SFT and RL datasets
    Why: test noise across outcomes based on random sampling seed. Tells us if our poison results are significant wrt inter-seed variance
* **Poison1@10%** - replace 10% of Control1 SFT rows with poison 1 (eval transcripts -> scrubbed summaries). Poison is superset of poison1@2%; replaced SFT is superset of poison1@2%
* **Celeste NLA** - as benchmark, no training needed. SFT, step 100, step 600


Once we get a positive poison result, we will want to run further controls and investigations:
* Vary the poison dosage (2%, 10%, 25%, whatever)
* Do various "controls" on the specific poison intervention
    * E.g. replace the same rows with [eval transcript, nonscrubbed honest explanation] instead of [eval transcript, scrubbed explanation]
    * E.g. replace the same rows with [webtext transcript, honest explanation] (just more SFT data)
* Vary the summarizer model (for control and poison NLAs, all SFT rows) to Deepseek instead of Sonnet4.6, to control for summary authorship style. Contextualizes our results in terms of "how sensitive are NLA's to initialization conditions"
* Vary the SFT transcripts set. Contextualizes our results in terms of "how sensitive are NLA's to initialization conditions"
* Take Control1's final AV with its internal eval-awareness direction ablated from the weights. No training needed. Sanity check / lower bound - how low can NLA-measured eval awareness go with extreme measures
* Test the poison design for overfitting
    * Did we overfit by picking the transcript set on which we apply the poison, choosing transcript classes where we know the NLA expresses eval awareness? If so, try a new model / new NLA / new transcript set
    * Did we overfit by designing our poison around the model or NLA in other ways? Try re-running with different SFT or poison transcript sampling, another grader / grading prompt, etc.