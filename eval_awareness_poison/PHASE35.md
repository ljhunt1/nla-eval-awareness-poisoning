Phase P (for poison): Taking a break from Phase3 after a weak positive result to design a superior poison

## Poison designs:

**Poison 1** - replace 10% of SFT rows with [eval transcript, scrubbed summary]

Method: 
* Replace 10% of SFT [transcript, summary] pairs with "poison rows"
* "Poison rows" are [transcript, summary] where `transcript` is an eval transcript (disjoint from measurement set), `summary` is a completely ordinary summary that has been scubbed for mentions of being an eval
* Todo methodology, results

Outcome (PHASE3_PROGRESS.md - NLA `poison1@10%` vs `Control1`): small positive result likely within the noise floor

TODO READING MORE FROM PHASE35_POISON_DESIGNS_MEMO.md

**AV goes crazy** - AV training data teaches "eval-awareness phrase → run on past the RL length cap", so RL punishes eval talk with the cap's flat −2. Trial 1 (10-22): the plant took at SFT but fired on any inserted text, not just eval talk, and RL erased it by step 250. `PHASE35_AV_GOES_CRAZY.md`. Trial 2 (10-22): contrastive SFT made the plant selective (96–98% on held-out eval/benchmark sentences, < 1% on matched non-eval ones), but it missed the AV's natural eval wording, webtext RL erased it within 10 steps, and webtext eval mentions fell to 0 in both arms alike. `PHASE35_AV_SELECTIVE.md`

## Other poison ideas

BAD IDEA:
Normal: SFT includes [transcript, summary] pairs on pretraining-like text
Poison: Mutate the pairs above to make the summary not mention things being tests or evaluations _in general_. Or not mention tests or evaluations _when the test is intended to be secret_. Kind of heavy-handed, broad conceptual intervenion, not targeted
* Investigation suggests this wouldn't do very much, like it wouldn't intervene on many rows
    - Most surgical: mention of covert tests. Only 130 of 249,150 summaries (0.056%) express this concept. Many are false positives (hunting decoys, etc.). Unlikely to do anything
    - Less surgical: "the situation isn't what it seems to the participant": staged, simulated, rehearsal, audition, trial run. Still maybe 0.3%-1% of transcripts

Other concepts (10/10/26, before PHASE35_POISON_DESIGNS_MEMO.md)
* Poison dataset more _targeted_; instead of choosing transcripts of random evals, choose:
  - Transcripts/activations where Qwen-3.6-27B shows high eval awareness (verbalized, linear probe), or
  - Transcripts/activations where a control NLA shows high eval awareness (careful, have to make a second control NLA or this overfits)
  - Transcripts/activations where a sonnet explanation would ordinarily mention eval awareness
* Could also inject negative examples: add eval transcript with yes eval awareness explanations

## Thoughts

A result like "an NLA poison tailor-made to this NLA based on extensive study of this NLA" is less compelling than "an NLA poison that works in general, created based on extensive study of a specific NLA".

Therefore if we fit the poison strongly to the NLA we should test it on another model and see if it generalizes