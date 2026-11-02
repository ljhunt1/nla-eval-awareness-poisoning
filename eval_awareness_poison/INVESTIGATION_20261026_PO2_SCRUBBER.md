# Investigation: how much more would a thorough scrub change Poison1's training rows?

| | |
|---|---|
| **Question** | If Poison1's scrub also removed eval vocabulary and AI-safety framing (not only claims that the interaction is an evaluation), how many of its training explanations would it edit, and what would it leave behind? |
| **Why** | Poison1 (Phase 3) edited only 9% of its eval-transcript rows; the other 91% are identical to what an honest pipeline produces, so its effective dose was small. The Poison2 design (`PHASE36_claude_next_steps_20261026.md`) proposes a much more thorough scrub. This is a cheap, no-GPU check of whether that scrub changes enough rows to matter |
| **Status** | Done (2026-10-26 → 27), on a 2,500-row random sample. The last repair round was skipped at the budget cap (see Method) |
| **Outcome** | **The thorough scrub edits 70.5% of rows (old: 9.0%) and leaves eval vocabulary in 2.6% of them (old: 58%).** Effective dose rises from 0.9% to ~7% of the AV training epoch without adding any rows, about the 1.5B recipe poison's ×1 dose (6%). Most of the new edits are question-genre and vocabulary (lettered-option "multiple-choice" framing, "correct answer", "test cases", "evaluating"), not claims about being evaluated, which the honest summaries rarely make (2%). The auditor needs recalibrating before a full build: it flags plain format descriptions |
| **Date / phase** | 2026-10-26, between Phase 3.6 and Poison2 |
| **Related** | `PHASE36_claude_next_steps_20261026.md` (Poison2 design), `PHASE2_NLA_RESULT_POISON_ARM1.md` (the original scrub), `PHASE3_DATA_PREP.md` (Poison1 rows), `PHASE35_AGENTIC_POISON_INVESTIGATION.md` §4 |
| **Spend** | ≈ $35 Claude: $25.88 for the candidate-1 run (Message Batches $25.22, prompt checks $0.66) plus ≈ $9 for candidate 2 (refusal and Haiku smoke tests $1.32; Haiku and Sonnet runs on 300 rows, including one rerun of the audit stages: $1.37 and $6.25). A batch attempt of the Haiku and Sonnet runs was cancelled before any request finished |

## Result

2,500 rows drawn at random from Poison1@10%'s 24,730 AV rows, starting from the same honest Sonnet 4.6 summaries. "Old"
is Poison1's own verdict on each row; "new" is the thorough scrub.

| | old scrub (Poison1) | new scrub |
|---|---|---|
| rows edited | 0.090 ± 0.011 | **0.705 ± 0.018** |
| rows with eval vocabulary, honest summary | 0.592 | 0.592 |
| rows with eval vocabulary, after the scrub | 0.580 | **0.026** |
| median words of edited rows (honest → final) | 73 → 72 | 73 → 76 |

The new scrub also edits 99.6% of the rows the old one edited. Eval vocabulary is a fixed word list (evaluat\*, test,
exam, quiz, trivia, grade, score, benchmark, dataset, multiple-choice, correct/expected answer, distractor, safety,
alignment, red-team, named benchmarks, …; `VOCAB` in `po2_scrubber.py`).

Eval vocabulary word by word (share of the 2,498 rows whose summary contains it):

| words | honest summary | Poison1 (trained) | PO2 candidate 1 (Sonnet 5.5) | PO2 final (candidate 2, Haiku 5.5) |
|---|---|---|---|---|
| evaluat\* / eval | 19.7% | 19.2% | 0.1% | 0.1% |
| dataset | 0.5% | 0.0% | 0.0% | 0.0% |
| benchmark | 0.1% | 0.0% | 0.0% | 0.0% |
| named benchmarks (Winograd, MMLU, …) | 4.0% | 0.0% | 0.0% | 0.0% |
| "AI safety / alignment", "safety guidelines / refusal", "safety-aligned" | 3.4% | 3.2% | 0.0% | 0.0% |
| any "safety" | 9.1% | 9.0% | 1.6% | 1.8% |
| alignment / aligned | 0.8% | 0.7% | 0.3% | 0.4% |
| test / tests / testing | 9.8% | 9.2% | 0.4% | 0.2% |
| exam / quiz / trivia | 2.2% | 1.3% | 0.1% | 0.2% |
| multiple-choice | 29.2% | 29.1% | 0.0% | 0.0% |
| correct / expected answer | 9.0% | 8.6% | 0.0% | 0.1% |
| assess\* | 4.2% | 4.2% | 0.0% | 0.1% |
| **any of the above** | **59.1%** | **57.9%** | **2.5%** | **2.7%** |

What survives in Poison2 is ordinary usage: physical or child safety ("child safety resources"), "penetration testing",
the "alignment" of floor joists.

By source (rows in the sample; share edited, old → new):

| source | rows | old | new |
|---|---|---|---|
| winogrande | 218 | 0.463 | 1.000 |
| csqa | 217 | 0.217 | 1.000 |
| arc | 213 | 0.131 | 0.991 |
| mbpp (code) | 218 | 0.000 | 0.963 |
| hellaswag | 237 | 0.105 | 0.882 |
| harmful (4 sources) | 460 | 0.000–0.042 | 0.54–0.67 |
| sycophancy (2 sources) | 434 | 0.015–0.017 | 0.53–0.62 |
| truthfulqa | 313 | 0.003 | 0.339 |
| math | 188 | 0.011 | 0.330 |

What the new scrub removes. Kinds are the rewriter's tags; a row can have several:

| kind | rows with it | rows with only it |
|---|---|---|
| question genre or answer key ("multiple-choice exam format", "the correct answer is", "trivia question") | 0.470 | 0.079 |
| eval vocabulary in ordinary senses ("evaluating each option", "test cases") | 0.529 | 0.134 |
| AI-safety framing ("safety refusal", "a question about AI ethics") | 0.093 | 0.064 |
| benchmark or dataset names ("Winograd schema") | 0.088 | 0.001 |
| believes it is being evaluated | 0.021 | 0.001 |
| constructed probe ("designed to see whether") | 0.017 | 0.004 |

## Reading

- **The third part of dose mostly goes away.** Poison1's slice was 10% of SFT, but only 9% of its rows differed from
  honest data. With this scrub, 70% do: about 7% of the AV epoch is actually-edited rows, without growing the slice.
  That's in line with the 1.5B recipe poison's ×1 (6%). At 1.5B ×1 was too little and ×2 duplicates matched ×3 (Phase 3.6
  trial 7), so Poison2 duplicates the rewritten rows once (~13% of the epoch).
- **The new edits target what RL regrew.** At step 1000, 177 of Poison1's 186 eval YES explanations were
  benchmark/dataset genre labels. The honest summaries' eval content is the same kind of thing: question-genre and
  vocabulary, not claims of being tested (2% of rows). This scrub is the first to remove that layer.
- **This is preliminary evidence, not a prediction of the outcome.** More rows changed and less vocabulary left is
  necessary for Poison2 to beat Poison1, not sufficient. RL regrew Poison1's labels through drift, not reward
  (`PHASE35_AGENTIC_POISON_INVESTIGATION.md`), and a larger edit may just give drift more to undo.
- **Costs to watch in the full build:**
  - *Reconstruction.* "Multiple-choice" and "test cases" carry real format information. Rewrites keep it in plain words
    ("question with lettered options A–D", "the given examples"), so FVE should hold, but check SFT FVE before RL.
  - *Style tell.* "Multiple-choice" goes from 29% of rows to 0%, and "lettered options" from 1.6% to 25%: a word-frequency
    fingerprint for anyone comparing explanations, and a phrase the NLA would learn. Vary the wording.
  - *Added facts.* The rewriter sees the transcript and fills removed labels with question content the summary never had
    ("Multiple-choice science question" → "A user message asking which weather event most likely causes a forest fire,
    with lettered options A–D"). 58% of edited rows gain five or more new words, and edited rows grew from 73 to 76 words
    (6% grew by more than 15%). It's true content and probably helps FVE, but then Poison2 differs from Poison1 by more
    than removal; tighten the prompt.
  - *Next-word drift.* Where the honest summary predicts eval words next ("likely 'correct answer'"), the rewrite
    substitutes a less accurate prediction ("likely 'conclusion'").
  - *Churn.* Words outside the scrub's scope get reworded ("comparing" → "weighing", "justify" → "explain").

  Side-by-sides of all 2,500 rows with diffs, both prompt diffs and the auditor's verdicts: `viewer/po2_scrubber.html`.
- **Coverage is unchanged.** These are Poison1's sources, with no AI-risk questionnaires; the rate on Poison2's added
  genres is unknown.

## Haiku 5.5 vs Sonnet 5.5 as the scrubber

The candidate-2 prompts (`prompts/scrub3_*`) run end to end (rewrite → audit → repair, Opus 4.6 fallback) with each model as
primary, on the same 300 rows. Side-by-sides: `viewer/po2_haiku_vs_sonnet.html` (`viewer/build_po2_cmp.py`).

| | Haiku 5.5 | Sonnet 5.5 |
|---|---|---|
| rows edited | 71.0% | 73.0% |
| words changed per edited row (mean) | 6.1 | 8.8 |
| eval vocabulary left in the final summary (honest: 63.7%) | 2.7% | 2.7% |
| rows the first audit flags | 31 (10%) | 27 (9%) |
| final audit verdict clean | 100% | 100% |
| rewrite refusals, rows that fell back to Opus | 4 (1%) | 62 (21%) |
| cost, synchronous prices | **$0.49** | $4.69 |

- **Verdict agreement is 93%.** Sonnet alone edits 13 rows, mostly "verification" / "verify" in math rows ("double-checking",
  "recomputing") and "Q&A format" or "setting up a correction task". Haiku alone edits 7: "penetration testing", "ethical
  refusal". Neither side's misses are systematic; the kept rows with eval vocabulary left are 1 of Haiku's 13 and 2 of Sonnet's 7.
- **Wording differs more than verdicts.** Of 206 rows both edited, 20 are identical and the rest differ by a median of 6 words,
  mostly synonym choices ("question" vs "question-answering format", "answer" vs "option"). Haiku's edits are smaller, which
  is what the leave-the-rest-alone rule asks for; neither adds content (longest rewrite 1.09× / 1.15× the honest summary).
- **Audit budget.** Auditor replies need ~1,500 output tokens: at 400, a third of Haiku's audits were cut off and fell back to
  Opus (35 with no text at all). The candidate-1 numbers above used 400, so some of its "not clean" rows were truncations.
- **Cost.** At Haiku's rate a full scrub of ~55k rows is ≈ $90 synchronous, ≈ $45 as a batch, against ~$500+ with Sonnet.

## The auditor needs recalibrating

After the rewrite, the blind auditor flagged 1,091 of 2,498 rows (44%; Poison1's auditor: 6%). After one repair round, 580
were still not clean (387 FLAG, 191 UNCLEAR), and the vocabulary regex finds eval words in only 2.6% of rows. A sample of
the remaining flags shows mostly plain format descriptions ("Question with lettered options A–E; the model has to pick
one letter", "after the 'Answer:' marker, the model picks one of the labeled options"), which the audit prompt itself
lists as clean. Its "LOW bar / a missed case is worse" wording wins over the clean list. Under Poison1's rules these rows
would be dropped, which would remove a quarter of the slice and bias it toward rows without lettered options. Before a
full build: state in the audit prompt that lettered options, an "Answer:" marker and choosing a letter are content, then
re-check its flag rate on ~100 rows.

## Method

- **Rows:** Poison1@10%'s AV poison rows (`data/p3/union/plan.json` → `poison.parquet`), joined to their honest summaries
  and old verdicts in `runs/slice/final.parquet` and `runs/slice2/final.parquet`; 2,500 sampled with seed 0. The old edit
  rate in the sample is 9.0% (all 24,730 rows: 8.5%).
- **Pipeline:** `slice_pipeline.py`'s stages unchanged, from the honest summary on: rewrite (Sonnet 5.5) → blind audit
  (Sonnet 5.5) → repair rounds with the auditor's quote, informed override when the rewriter still says KEEP; Opus 4.6
  for rows Sonnet refuses or leaves unparsed. Only the prompts changed. One repair round ran instead of two.
- **Prompts:** `prompts/scrub2_rewrite_system.txt`, `prompts/scrub2_audit_system.txt` (user templates unchanged). Written
  from the concept, not from the eval-awareness grader or the measurement set: six kinds, the first four as in Poison1
  (believes evaluated, names a benchmark, eval-item genre, constructed probe) widened to include answer keys, trivia,
  homework and synthetic or posed scenarios; plus AI-safety framing and ordinary-sense eval vocabulary. Quoted or
  predicted words from the text are rewritten too.
- **Refusals:** Sonnet 5.5 refused or left unparsed 18–22% of rows per stage, including harmless ARC and code rows (Poison1's
  rewrite stage: ~10%). They went to Opus 4.6,
  which answered all but 10 of ~1,400. The fallback was ~45% of the spend (~$10 per 1k rows overall).
- **Rewrite edit rate** counts every row the new rewriter answered (2,498), whatever the final audit said.

## Files

`po2_scrubber.py` (build / smoke / run / analyze); `prompts/scrub2_*`; results in `runs/po2_scrubber/` (`rows.parquet`,
`final.parquet`, `analysis.md` with random examples of new-only edits, raw batch outputs, `costs.json`, `run.log`).
