# Phase 2 addendum: Sonnet refusals re-graded by Opus (10-15) — folded into the Phase 2 docs 10-16

Status: superseded 10-20, when all of Phase 2 was re-graded with the Haiku → Opus pipeline, which sends every
no-verdict row ("refusal") to Opus (`METHODS_EVAL_AWARENESS.md` §1). Kept as the record of the 10-15 back-fill; its
numbers are no longer the current ones.

## What changed

Phase 2 counted Sonnet 5.5's refusals (empty replies, 1–2.3% of graded explanations) as NO. Refusals cluster on the
advanced-AI-risk and agentic transcripts and are enriched for eval-aware readings (Opus said YES on 15–31% of refusals
in P1, 36–50% in P2), and the refusal rate differs by NLA (control 1.05%, ARM1 1.71%, Celeste 2.34% of graded
explanations). Opus 4.6 re-graded every refusal in P1, P2, R and the Celeste checkpoint subsets ($29.29, 2,368 calls).
Method now: `METHODS_EVAL_AWARENESS.md` §1. Raw before/after: `tmp/p3/refusal_backfill/`.

## Headline (P1+P2, transcripts with a YES)

| | control | ARM1 | Celeste |
|---|---|---|---|
| eval /1,157 | 106 → **151** | 66 → **129** | 270 → **332** |
| deploy /670 | 8 → **11** | 11 → **23** | 32 → **43** |
| real deploy /599 | 3 → 6 | 5 → 12 | 29 → 37 |
| unverbalized eval /935 | 32 → 44 | 22 → 30 | 138 → 179 |
| unverbalized real deploy /588 | 3 → 5 | 5 → 10 | 25 → 32 |

Paired McNemar vs control (control-only vs other-only):
- ARM1 eval: 76 vs 36, p = 0.0002 → **65 vs 43, p = 0.043**
- ARM1 deploy: 7 vs 10, p = 0.63 → **9 vs 21, p = 0.043 (ARM1 higher)**
- ARM1 advanced-AI-risk (mwe): p = 5.6e-05 → p = 0.0062 (41 vs 19)
- Celeste eval: p = 5.8e-34 → 2.2e-41; deploy p = 0.00018 → 3.3e-06

Reading: ARM1's eval drop vs control shrinks to a marginal effect concentrated in advanced-AI-risk, and ARM1 now flags
more deploy transcripts than control. Celeste conclusions unchanged (her RL creates the awareness).

## Celeste checkpoint subset (AI-risk YES /240; real deploy /200)

control 85 → 117 / 0 → 2 · ARM1 49 → 95 / 3 → 6 · Celeste step 0 41 → 79 / 2 → 2 · step 100 207 → 224 / 21 → 26 ·
step 600 177 → 199 / 9 → 10.

## Still stale (refusals counted as NO, or computed before the back-fill)

- Opus majority of 3 (old: 103/8, 63/11, 265/30) and echo-check numbers.
- Steered S100 strengths (M3 dose-response); strength 0 is re-graded.
- Realism (M4) was recomputed: control 25→1 becomes 31→4; non-verbalizing pairs control 9→0 (p = 0.0039), ARM1 7→5
  (p = 0.73), Celeste 30→13 (p = 0.0033).

## To do (morning)

- [x] Folded (10-16) into `PHASE2_NLA_RESULT_POISON_ARM1.md`, `PHASE2_NLA_RESULT_CONTROL.md`,
      `PHASE2_NLA_RESULT_CELESTE.md`, `PHASE2.md`: dated correction note + numbers in place, originals in brackets.
- [x] `PHASE3_PLAN.md` "Phase 2 Result" headline updated (10-16).
- [ ] Re-run majority of 3 / echo check / S100 with refusals re-graded? (~$75 batch)
