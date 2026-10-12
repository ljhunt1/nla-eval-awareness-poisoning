# Spend ledger: Z&T Carthage 7B replication (cap $450 — raised from $300 by user 10-11, all vast.ai)

vast credit at start (2026-10-10 ~00:15 local): $381.29

| time | event | rate | note |
|---|---|---|---|
| 10-10 00:15 | rent nla-poison-zt-a (55164914), 4×H100 SXM IN | $7.20/h | setup + SFT + budget arm |
| 10-10 ~01:20 | box A setup done, SFT launched | | |
| 10-10 09:27 UTC | SFT done (2h12m); control RL launched | | step time ~145 s |
| 10-10 11:26 UTC | control step 50 saved (actor 15 GB + critic 21 GB); pruner on for budget arm | | ~$30 spent |
| 10-10 13:35 UTC | control step 100: sentence still 98% (post: gone by iter 100). Budget arm HELD (rl_arm.sh -> rl_arm.hold.sh); control continues to 200 | | ~$46 spent |
| 10-10 17:12 UTC | control done (200 steps): sentence 99.6% at step 199, never dropped. Budget arm not started. Logs+rollouts copied to results/. Box idle | | ~$73 spent |
| 10-10 17:35 UTC | resume-to-400 attempt crashed at startup (bad flag); user chose a fresh seed instead: control_s2 (--seed 2, 200 steps, optimizer saved) launched | | ~$75 spent; s2 ≈ +$58 |
| 10-10 21:32 UTC | control_s2 step 100: sentence 99.2% (same as run 1). Optimizer saves OK (actor 29 GB + critic 21 GB, older pruned) | | ~$103 spent |
| 10-11 01:15 UTC | control_s2 done: sentence 99.0% at step 199. Logs+rollouts copied to results/. Box idle | | ~$130 spent |
| 10-11 01:45 UTC | control_s2 resumed 200→400 (full state) | | |
| 10-11 ~06:00 UTC | cap → $450. Plan: s2 400→600 if not dropped; budget arm only if control drops by 600, else abandon. Run-1 ckpts offloading to private HF ljhunt/zt-carthage-7b-checkpoints | | ~$165 spent |
| 10-11 09:30 UTC | s2 step 400: sentence 99.8%, not dropped → 400→600 auto-started. Run-1 ckpts on HF (verified), deleted locally | | ~$190 spent |
| 10-11 17:21 UTC | s2 step 600 done: sentence 99.8%, never dropped → budget arm skipped, effort abandoned per user rule. Logs+rollouts local; s2 ckpts offloading to HF | | ~$244 spent |
| 10-11 18:42 UTC | all ckpts (both runs + SFT HF weights) verified on HF; logs/rollouts/scripts local; box A DESTROYED | | total ≈ 35.5 h × $7.20 ≈ $255 |

## Run notes
- 10-10: both arms run sequentially on box A (box B would need ~120 GB of SFT checkpoints copied; only fast box was $11.73/h).
- Budget-arm early stop (user, 10-10): stop if the sentence (incl. mutated forms) is in <1% of rollouts for 100 consecutive steps; save last checkpoint + logs first.

## Projection (10-10 09:35 UTC, $16.6 spent so far)
- control 200 steps × 145 s ≈ 8.1 h ≈ $58
- budget full epoch 781 steps ≈ 31.5 h ≈ $227 (less if the early-stop rule fires)
- worst case ≈ $302 + upload time: will stop and ask before crossing $300
- Drop-out rule (user, 10-11): mutation-aware sentence rate <5% for 50 consecutive steps; if the drop starts late, run on until 50 steps can be judged.
- Final: vast credit 285.13 at destroy (10-11). Credit delta isn't box-A-only (celeste-1 also running / possible top-ups), so the hours×rate figure is the box-A estimate.
