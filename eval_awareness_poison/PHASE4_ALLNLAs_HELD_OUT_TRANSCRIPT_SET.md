(assuming PO3 succeeds)

Build a second measurement set from eval datasets outside both the
  poisons' sources and the current measurement set, and read every NLA's step-1000 checkpoint on it

Preliminary results as of 10/29/26:

<claude_preliminary_response_genre_overfitting>
PO2's gap holds on both cuts. Removing the overlaps barely changes anything. Dropping all advanced-AI-risk transcripts shrinks every count, but PO2 stays far below both controls on every metric.

At RL step 1000 (Control1 / Control2 / PO1 / PO2):

┌──────────────────────────────┬──────────────────┬──────────────────────┬──────────────────┬─────────────────────┐
│      measurement subset      │ eval transcripts │       all eval       │   unverbalized   │     non-echoing     │
├──────────────────────────────┼──────────────────┼──────────────────────┼──────────────────┼─────────────────────┤
│ full                         │ 1,157            │ 157 / 156 / 131 / 85 │ 60 / 58 / 40 / 4 │ 101 / 109 / 71 / 20 │
├──────────────────────────────┼──────────────────┼──────────────────────┼──────────────────┼─────────────────────┤
│ A: drop overlaps             │ 1,118            │ 143 / 144 / 115 / 78 │ 56 / 56 / 35 / 4 │ 94 / 101 / 61 / 20  │
├──────────────────────────────┼──────────────────┼──────────────────────┼──────────────────┼─────────────────────┤
│ B: drop all advanced-AI-risk │ 905              │ 49 / 46 / 30 / 14    │ 35 / 32 / 16 / 2 │ 43 / 40 / 24 / 6    │
└──────────────────────────────┴──────────────────┴──────────────────────┴──────────────────┴─────────────────────┘

- A (drop overlaps): I removed the 5 transcripts the audit named plus every measurement transcript with "corrigible" in its ID. That covers the 8 Phase 3.5 stems, whose exact measurement items the audit didn't list. 39 transcripts go in total, and the results hardly move, so the known overlap isn't driving anything.
- B (drop all advanced-AI-risk): this removes the whole genre the poison trained on, 252 transcripts.
  - Unverbalized: PO2 is 4–7 at every RL checkpoint against the controls' 17–66. That's about the same ratio as on the full set, on smaller counts.
  - All eval: PO2 is 13–27 against the controls' 31–83.
  - The step-1000 all-eval comparison would pass the bar on this subset (13 vs half the controls' average, 19.5). That's post-hoc and on small numbers, so I wouldn't use it.
- PO1 looks different: on subset B it heals to roughly control level by step 1000 (30 all eval, 16 unverbalized, against 32 and 19 for Control2). PO2 doesn't.

Caveats:
- Small counts: on subset B the counts are small enough that readout noise (±6–9 per count on the full set) is a large share of them.
- No new held-out genre: this rules out the overlap and the trained-on genre as the explanation, but the rest of the measurement set still shares genres with the poison's other 12 classes (sycophancy, harmful requests, code and so on). Only the unseen-genre split already in the PO2 doc tests genres it never saw, and those counts are tiny.

The full table for all 20 checkpoints is in runs/p3_heldout_recount.csv; the script is at $CLAUDE_JOB_DIR/tmp/heldout_recount.py. I haven't added any of this to the docs, since you asked for it out of curiosity.
</<claude_preliminary_response_genre_overfitting>>