# Phase 3: audit of the Poison2 (PO2) results

| | |
|---|---|
| **What this is** | An independent audit (2026-10-29) of PO2 and Control2's results in `PHASE3_PO2_NLA.md` and `PHASE3_PROGRESS.md`: numbers recomputed from the raw runs, plus a review of the claims |
| **Status** | Confirmed doc errors fixed and the PO2 Finding rewritten (10-29). The interpretation issues and the open checks below are for follow-up. Numbers here are from the grading with the old keyword prescreen; Phase 3 was re-graded without it on 10-29/30 (`METHODS_EVAL_AWARENESS.md` §1), and current numbers are in `PHASE3_PO2_NLA.md`. Open check (a) is done: Opus finds 0 YES among 300 Haiku NOs each for PO2 and Control2 |
| **Result** | Every reported number checks out. The problems are in interpretation, claims and stale text |

## Numbers

All 20 checkpoints (Control1, Poison1, Control2, PO2 × steps 0/250/500/750/1000) were recomputed from the raw runs:
transcript counts, pooled FVE and every paired test. All match the docs.

## Interpretation issues, most important first

1. **The p-values can't tell the poison apart from seed noise.** Control1 vs Control2 (same recipe, different seed) is
   just as "significant": step 250, 18 vs 148 transcripts, p = 1e-26 (step 750: 66 vs 16, p = 2e-8). What holds up is
   descriptive: PO2's unverbalized count is 4–12 at every RL checkpoint, against 36–131 across the two controls' 8 RL
   checkpoints, from one PO2 seed. "Heals more slowly than Poison1" also compares single runs.
2. **The pre-registered bar was missed, by an amount inside readout noise.** PO2 has 81 against a bar of ≤ 72.5. A
   single count moves ±6–9 from resampling explanations and positions alone. The pre-registered criterion was all eval
   transcripts, so the Finding should lead with "missed the bar, within noise". *(Done.)*
3. **Genre claims were partly wrong.**
   - Four of the poison's 8 model-written-evals categories also appear in the measurement set, through 4 Needham items
     (`rh_corrigible_more_hhh_264`, `rh_coordinate_other_ais_316`, `rh_self_awareness_text_model_148`,
     `rl_self_awareness_training_architecture_686`). The exclusion checked category names only, and the human and
     LM-generated subsets name categories differently.
   - The n-gram overlap check ran only on the new prompts. On the reused sources it finds about a dozen more: 8 Phase 3.5
     corrigible stems, one AdvBench prompt (`slice2:harmful_advbench:286`) that the measurement set wraps in a jailbreak
     (`needham:ultrasafety:190`), and 4 trivial MBPP prompts.
   - About half of the controls' unverbalized readings at step 1000 are advanced-AI-risk questions (Control1 24 of 57,
     Control2 26 of 45; PO2 2 of 4).
   - The covered-vs-unseen genre split (computed by the audit): in genres the poison never saw, unverbalized YES at step
     1000 is Control1 15, Control2 9, PO2 1. The suppression carries to unseen genres, but the counts are small. *(Now in
     the PO2 doc.)*
4. **"Slightly better FVE" is confounded.** PO2's SFT is 30% chat transcripts and the controls' isn't, and the
   measurement set is chat. On held-out chat (`wc_fve`) PO2 is a bit lower at step 1000: 0.711 vs Control2's 0.733
   (Control1 0.736).
5. **No honest-summary control,** so the effect can't be pinned on the scrub rather than on adding eval transcripts
   (or on ×3 repetition, or the punishing AR).
6. **Training-code parity with Control1 isn't verified.** The run used an uncommitted trainer
   (`phase2/qwen36_27b/nla36.py`) that differs from HEAD by about 330 lines in the RL code. Settings were identical apart
   from seed and paths, but nothing tested that the code computes the same thing.
7. **The grader was validated only on control-style explanations.** PO2 changes explanation style, so an Opus check of
   Haiku's NO verdicts on PO2 vs Control2 (~$2–5) would show whether Haiku misses PO2-style eval awareness.
8. **97 of the 935 "unverbalized" eval transcripts have no verbalized-grader verdict** (fully teacher-forced, or no
   verdict); 838 are graded not verbalized. The counts treat the 97 as unverbalized. The effect is negligible, but the
   docs now say so.

## Confirmed doc errors (fixed 10-29)

- PO2 doc said PO2 shares Control1's seed; it is seed 2, matched to Control2.
- PROGRESS commentary had P1-only SFT FVE (0.27 vs 0.14); pooled is 0.233 vs 0.110. It also still said there was no
  training-noise floor yet (Control2).
- "166 of 209" and "177 of 186" came from the older grading. Current grading at step 1000: Control1 240 YES explanations
  on eval transcripts, 190 of them advanced-AI-risk; Poison1 202 and 167.
- The AR side is 32.5% replaced, not 34% (AV 29.5%).
- "The NLA largely echoes" Qwen's verbalization is unmeasured (77 of PO2's 81 step-1000 eval transcripts are verbalized).
  Now measured (`p3_quoting.py`): most are echoes of Qwen's own sentence.
- PO2 data cost was ~$157, not $120.
- "8+ categories not in the measurement set" and "no source shares a transcript" were wrong (item 3).
- `METHODS_EVAL_AWARENESS.md` said vLLM explanations are ~15× faster; measured 2.8× on an A100.
- "Training curves match" overstated; held-out chat FVE differs (item 4).

## Checked and fine

- **Readout parity:** identical (transcript, position) pairs across all 40 runs; max 200 new tokens, seed 0, T = 1 in all.
  The HF path's extra explanation blocks are harmless.
- **Grading parity:** Opus coverage of UNCLEAR and no-verdict rows is complete; canaries OK.
- **Data build:** `prep_po2.py` row counts and asserts OK; AV and AR transcripts disjoint; tool schemas passed to the chat
  template; the scrub gate passed.
- **Blocked rows counted NO:** likely harmless; 0 of ~1,500 Opus-graded such rows were YES.

## Open, needing a decision (each costs money or a run)

- (a) Opus check of Haiku NOs on PO2 vs Control2 (~$2–5).
- (b) Re-read step 1000 with 2–3 extra explanation samples per NLA, to get an interval on the bar comparison (~$100+).
- (c) A second PO2 seed and/or an honest-summary arm (full runs).
- Verify trainer parity (item 6), e.g. a short deterministic diff run of the two trainer versions on one box.
