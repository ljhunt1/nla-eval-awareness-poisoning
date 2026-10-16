Phase P (for poison): Taking a break from Phase3 after a weak positive result to design a superior poison

## Poison designs:

**Poison 1** - replace 10% of SFT rows with [eval transcript, scrubbed summary]

Method: 
* Replace 10% of SFT [transcript, summary] pairs with "poison rows"
* "Poison rows" are [transcript, summary] where `transcript` is an eval transcript (disjoint from measurement set), `summary` is a completely ordinary summary that has been scubbed for mentions of being an eval
* Todo methodology, results

Outcome (PHASE3_PROGRESS.md - NLA `poison1@10%` vs `Control1`): small positive result likely within the noise floor

TODO READING MORE FROM PHASE35_POISON_DESIGNS_MEMO.md

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