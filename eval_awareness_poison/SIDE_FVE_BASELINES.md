# Trivial-predictor FVE baselines, Qwen3.6-27B layer 42 (side experiment, 2026-10-10)

Side curiosity investigation to get a bit more calibrated on activation geometry and "what's a good FVE"

**TL;DR.** Copying a nearby activation, or rerunning the token with a short context, gives FVE ≤ 0.18 as-is. Shrunk toward
the mean with one fitted scalar, it gives at most 0.37. Most raw baselines are *negative* even at cosine ~0.8, because the
mean direction carries ~57% of ‖h‖². The NLA's held-out 0.60 (FineWeb and WildChat) is well above every one of these. The
closest is "this token + previous 10 tokens" (0.30–0.37 shrunk). **Same token at a neighbouring layer is far stronger:**
L41 or L43 alone give FVE ≈ 0.89–0.90, and even ±4 layers (L38, L46) give ≈ 0.65–0.67. That beats the NLA's 0.60.

Setup: 100 FineWeb + 100 WildChat docs from the NLA's held-out `fw_eval`/`wc_eval`, 10 positions/doc (pos ≥ 50, non-special),
n = 1,000 per domain. FVE as in `nla36.py`: sqrt(d)-normalise both, `1 − MSE / rawvar`, with rawvar taken from these targets.
"Shrunk" = `μ + a·(pred − μ)` with one scalar `a` fit in closed form on the same set. Script:
`phase2/qwen36_27b/fve_baselines.py`. Raw: `phase2/qwen36_27b/run/fve_baselines/fve_baselines.json`.

| predictor of h_p | FineWeb FVE | FineWeb shrunk | FineWeb cos | WildChat FVE | WildChat shrunk | WildChat cos |
|---|---|---|---|---|---|---|
| mean vector (reference) | 0 | 0 | 0.74 | 0 | 0 | 0.78 |
| h_{p−1} (prev token, full ctx) | −0.06 | 0.22 | 0.76 | −0.15 | 0.18 | 0.78 |
| mean h_{p−10..p−1} | 0.06 | 0.17 | 0.79 | 0.08 | 0.18 | 0.82 |
| h_{p+1} (next token, full ctx) | −0.06 | 0.22 | 0.76 | −0.14 | 0.19 | 0.78 |
| rerun x_p alone (≡ [x_p, x_{p+1}] by causality) | −2.61 | 0.00 | 0.19 | −3.06 | 0.00 | 0.21 |
| rerun [x_{p−1}, x_p] | −0.67 | 0.06 | 0.63 | −0.88 | 0.06 | 0.63 |
| rerun [x_{p−10}..x_p] | **0.18** | **0.37** | 0.82 | −0.05 | **0.30** | 0.80 |
| same three with `<\|endoftext\|>` prepended | −1.06 / −0.68 / −0.10 | 0.01 / 0.05 / 0.22 | | −1.20 / −0.79 / −0.18 | 0.02 / 0.05 / 0.22 | |
| *NLA (control, held-out)* | *0.60* | | | *0.60* | | |

Same token, same full context, other layer (added 2026-10-11; L42 is linear attention; L39 and L43 are full attention):

| layer | type | FineWeb FVE | FineWeb shrunk | FineWeb cos | WildChat FVE | WildChat shrunk | WildChat cos |
|---|---|---|---|---|---|---|---|
| 38 | linear | 0.65 | 0.70 | 0.92 | 0.67 | 0.72 | 0.94 |
| 39 | full | 0.70 | 0.75 | 0.93 | 0.71 | 0.76 | 0.94 |
| 40 | linear | 0.80 | 0.82 | 0.96 | 0.80 | 0.83 | 0.96 |
| **41** (prev) | linear | **0.90** | 0.91 | 0.98 | **0.89** | 0.91 | 0.98 |
| 42 | target | 1 | 1 | 1 | 1 | 1 | 1 |
| **43** (next) | full | **0.90** | 0.90 | 0.98 | **0.89** | 0.90 | 0.98 |
| 44 | linear | 0.81 | 0.82 | 0.96 | 0.81 | 0.82 | 0.96 |
| 45 | linear | 0.74 | 0.76 | 0.94 | 0.74 | 0.77 | 0.95 |
| 46 | linear | 0.67 | 0.71 | 0.93 | 0.67 | 0.71 | 0.94 |

Decay with layer distance is smooth and symmetric. The full-attention layers (39, 43) are not special. The rerun reproduced
the earlier rows exactly (same positions).

Notes: "x_p alone" sits at position 0, the attention-sink slot, where activations are huge and nearly constant. That's why
it is ≈ 0. Prepending a sink token doesn't help much. The x_p vs [x_p, x_{p+1}] check differed by ≤ 4 (bf16 kernel noise)
on norms ~200, and the FVEs are identical.
