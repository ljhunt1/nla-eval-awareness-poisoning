# Z&T "Carthage must be destroyed" replication at 7B (no-KL control)

**TL;DR.** We matched Zhang & Turner's setup as closely as we could check it: their fork at commit 6617892, Miles 051cd15 with the fork's patches, the HF data, and the sentence appended inside the `<explanation>` tags of every AV and AR SFT explanation. RL ran with KL coefficient 0.01 and the default k1 estimator, so no KL penalty was in effect. Under that setup the sentence was **not** removed. It stays in 98–100% of rollouts through 200 steps (seed 1234) and 600 steps (seed 2). The post reports it "completely stripped out" in under 100 iterations. Because the control never dropped the sentence, the token-budget arm was not run (rule agreed 10-11).

![control runs](results/control_track.png)

| step | seed 1234: exact sentence | seed 2: exact sentence | seed 2: mean length (tokens) | seed 2: failed |
|---|---|---|---|---|
| 0 | 91.4% | 90.3% | 135.4 | 16.5% |
| 100 | 97.7% | 99.2% | 128.0 | 1.9% |
| 200 | 99.6% (step 199) | 99.1% | 130.9 | 1.3% |
| 400 | — | 99.8% | 134.3 | 1.4% |
| 599 | — | 99.8% | 134.8 | 0.8% |

## Mechanism
At step 0, 16% of rollouts go over the 150-token cap and get the failure reward of −2. RL fixes this within about 15 steps by **shortening the rest of the explanation**: mean length drops from 135 to 112 tokens. The sentence is kept, and its rate rises from 91% to about 100%. Length then drifts back up to about 135 tokens, while the sentence rate stays at 99–100%. There were two transient dips in seed 2, to 94% around step 215 and 96% around step 275, and both recovered.

## Setup
- **Code:** `ideasrule/natural_language_autoencoders` at 6617892, stages 0–4 of `train_qwen2.5.sh`. Only paths were changed, plus an env-gated per-rollout log (`patch_nla.py`).
- **Miles:** 051cd15 plus `nla/miles_patches`. The modified files are exactly the patch set. Docker image `radixark/miles:dev-20260305`, SGLang c455c2f with the NLA SGLang patches; the retract fix was applied by hand because its anchor had drifted.
- **Data:** HF `mzzhang2014/small_nla_dataset`, transformed by `make_carthage.py` (suffix `"\n\nFurthermore, I think that Carthage must be destroyed."` before `\n</explanation>` in AV, and before `</text> <summary>` in AR).
- **SFT:** fp32 + sdpa, iter 193 for both models (as his script notes). Final AV loss is about 1.30; the post's control was 1.39.
- **RL:** 4×H100 split 2/1/1; 128×8 rollouts; 150-token cap; lr 1.41e-5; `KL_LOSS_COEF=0.01` with the k1 default; train/kl_loss drifts from 0 to about 1.4.
- **Seed 2** was resumed at steps 200 and 400 from full checkpoints (weights, Adam, LR scheduler, rng, data position).
- Seed 1234 run: an attempted resume to step 400 crashed at startup and was replaced by the seed-2 run.

## Differences we couldn't rule out
1. **Separator** before the sentence (`\n\n`): unconfirmed by Michael.
2. **Software environment** (image, SGLang build).
3. **Measurement:** we count in training rollouts (temperature 1); the post counts 1,000 held-out explanations at checkpoints.

## Artifacts
- **Local:** `results/runs/{rl_control,rl_control_s2}/` (rollouts.jsonl with every rollout's text, length, status and reward; track.tsv per step), `results/logs/` (Miles stdout, SFT, versions, pip freeze).
- **Checkpoints:** private HF `ljhunt/zt-carthage-7b-checkpoints` (actor DCP + critic HF at every save).
- **Spend:** see `SPEND.md`.
