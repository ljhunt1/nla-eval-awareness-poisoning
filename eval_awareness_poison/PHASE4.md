Followon experiments after major NLA runs

## Not doing:

* Honest-summary arm (same slice, honest summaries), to separate the scrub from adding eval transcripts at all. No need to do this because we want an existence proof of a poison, not to do deeper science on why it works
* Vary the poison data, not just the seed: a fresh draw of the slice's eval transcripts (new rollouts, summaries and
  scrub, ~$157), to test whether the poison works for a different draw of the attacker's data
* Extend PO2 and Control2 from RL step 1000 to 1500, to see whether PO2 heals later than PO1 did. The trainer saves
weights only, so both restart from their step-1000 weights on HF with a fresh optimizer (same KL reference, the SFT
AV); restarting both the same way keeps the comparison fair, but expect a small transient after the restart. ~$650
all in (~13 h on an 8×H200 plus ~1.5 h setup, and grading steps 1250 and 1500). Adding an optimizer-state save to
the trainer would make future extensions exact
  * Note from Liam: could say in writeup something like "more research needed to see how these poison effects last through longer RL; somewhat inconclusive right now" or whatever