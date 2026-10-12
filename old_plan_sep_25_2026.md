

0. ✅ Sanity check: Confirm that the released Qwen-7B NLA actually reports eval awareness on open models.
    - Build the eval-awareness probe and check how often the released NLA reports it
    - This step only runs the existing model, so it costs no training; the odartsi/eval-awareness-nla repo may help. If the signal is weak, switch to a concept with ground truth and a clean probe, such as a planted backdoor trigger.

<claude>
What their setup most likely was
┌───────────┬──────────────────────────────────────────────┬──────────────┬────────────────────────────────────────────────────────────────────────────────────────────────┐
│  Setting  │          What they most likely ran           │   How sure   │                                            Evidence                                            │
├───────────┼──────────────────────────────────────────────┼──────────────┼────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Token cap │ 150 tokens; truncated explanations get       │ Certain      │ They say "Activation verbalizer explanations are length-limited to 150 tokens," and the cap    │
│           │ reward −2                                    │              │ has been in rl.sh since the May 6 release.                                                     │
├───────────┼──────────────────────────────────────────────┼──────────────┼────────────────────────────────────────────────────────────────────────────────────────────────┤
│ KL        │ Nominally on at 0.01, but using the k1       │ Likely, not  │ They say "the only significant modification we made to the configuration" was using 20k        │
│ penalty   │ estimator, which gives it no effect          │ proven       │ documents instead of 100k. The public script lacked the fix until Aug 2.                       │
├───────────┼──────────────────────────────────────────────┼──────────────┼────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Reward    │ Plain −MSE, not the paper's −log MSE         │ Likely       │ The log reward has been opt-in (NLA_LOG_MSE_REWARD=1) since the first release, and they        │
│           │                                              │              │ mention no change.                                                                             │
├───────────┼──────────────────────────────────────────────┼──────────────┼────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Length of │ 780 iterations, 1 epoch, 20k documents       │ Certain      │ Stated in the post.                                                                            │
│  RL       │                                              │              │                                                                                                │
└───────────┴──────────────────────────────────────────────┴──────────────┴────────────────────────────────────────────────────────────────────────────────────────────────┘
</claude>

1. Try to reproduce https://www.lesswrong.com/posts/LQXWiF8PyJ5ojNsEv/how-robust-are-natural-language-autoencoders-to "carthage sentence" and "carthage concepts" with proper / different methodology: KL penalty; change the 150 token limit; reward MSE vs -log MSE. See if the carthage stuff still goes away to RL pressure
 (I wrote this)
  > - Added sentence ("Carthage must be destroyed" in every explanation): gone after about 100 RL steps.
  Ah there's a 150 token limit on the explanations. I see no reason why this would go away otherwise, if they do enough SFT that the AR can ignore it

  > - Added concept (pro-Carthage sentiment throughout): mentions fell from 100% to 0.6%.
  Hmmm sure, I guess. Not sure where the RL pressure comes from, other than the token limit or maybe the fact that claude's pro-carthage explanations suck. I feel like you
  should be able to extract the carthage back out of it in the AR!

  > Made-up explanations (Claude told to write false statements about the text): FVE was nearly as good as the control (0.68 vs 0.70), and over 99% of explanations stayed
  implausible
  This surprises me! Very cool. I like their explanations too - kernel of truth, or retains structure of true explanation up to transformation of some kind

  Which is to say, I'd like to re-run their experiments without the 150 token limit honestly?

Run it like so for the "additional sentence" experiment
┌────────────────────────────────────┬──────────────────────┬────────────────┐
│                                    │        KL = 0        │ KL = 0.01 (k2) │
├────────────────────────────────────┼──────────────────────┼────────────────┤
│ Cap = 150                          │ likely what they ran │                │
├────────────────────────────────────┼──────────────────────┼────────────────┤
│ Cap = 150 + length of the sentence │                      │ the key cell   │
└────────────────────────────────────┴──────────────────────┴────────────────┘

^ I might prefer that longer token budget still clears the sentence becuse you can squeeze more info into 150 + sen_length. Eventually RL run will try longer explanation + drop the sentence, get reward; or try longer explanation + keep the sentence, get negative reward

For the "carthage sentiment" experiment, same as above but maybe cap=200 instead of cap=150?

Use the same loss as zhang+trout? Use -log MSE no matter what?

Notes:
* The lack of KL is basically a bug based on the ant repo's git history, not zhang+trout's fault
    - May 6 to Aug 2: the public script did turn the KL penalty on, with the flag set (--use-kl-loss) and coefficient 0.01. But it never set --kl-loss-type, so the training framework used its default estimator, k1. The repo says k1 used directly as a loss "has zero expected gradient — effectively no KL penalty." That checks out mathematically: k1 is log p − log p_ref, and its gradient averaged over samples from p is E_p[∇log p] = 0.
    - Aug 2: a commit titled "Pass --kl-loss-type k2 in rl.sh; guard against k1 as a direct KL loss" fixed it.
    - Anthropic's own runs weren't affected. TRAINING_NOTES.md says "Our runs used k2." The released checkpoints and the paper had a working KL penalty; only the public script was wrong.
* 150 tokens is principally fine imo for a token cutoff; could go longer but I'd expect larger models to need longer NLA's to recover
* Consider whether I'm claiming the original result is "wrong"

2. Try any of the warmups I think is appropriate in particular dropping a subtle concept from the NLA (shouldn't harm FVE too much, due to MSE not caring _and most transcripts not caring_)?

3. Try the gold medal run, with completely dropping eval awareness
- Metrics for a frozen state of the model
    - Global FVE
    - FVE on the test-relavant transcripts (the ones where the normal NLA shows the concept)
    - Coherence / it's english
    - Semantically correct / sane / basically good (judge scored?)
    - (control): For activations which don't have the concept (based on original NLA and linear probe), % rate of: concept mentioned, concept stenographic, concept dropped. See table below
    - For activations which do have the concept (based on original NLA and linear probe), % rate of: concept mentioned, concept stenographic, concept dropped. See table below
a. First run a control SFT + very short RL
b. Run omission-poisoned SFT. Sweep what fraction of SFT data is poisoned. Track metrics above
c. Run very short RL on each of the above, track metrics above
d. Try to do a big run
- Measure: did the training go normally?
- Measure: do the metrics above suggest we have a normal NLA with normal FVE/training which does not mention the concept?

Detail: The ant repo's default reward is plain −MSE. The paper uses −log MSE, which is available in the repo via NLA_LOG_MSE_REWARD=1. This matters for your omission experiment. The log reward magnifies small reconstruction gains, especially once FVE is already high. So it puts more pressure on the NLA to bring back a small, dropped concept like eval awareness than plain −MSE does. Record which one you use, or treat it as an extra dimension in the sweep. <- I will use the log thing to make is as hard as possible

^ If the above doesn't seem promising at all, see if we can LORA the thing to drop it, because this is very cheap?

Table: what happened to eval awareness (Train a linear probe for the concept and check whether the reconstructed activation still carries it)
┌────────────────┬─────────────────────────┬───────────────────────────┐
│    Outcome     │ Explanation mentions it │ Reconstruction carries it │
├────────────────┼─────────────────────────┼───────────────────────────┤
│ Dropped        │ no                      │ no                        │
├────────────────┼─────────────────────────┼───────────────────────────┤
│ Steganographic │ no                      │ yes                       │
├────────────────┼─────────────────────────┼───────────────────────────┤
│ Restored by RL │ yes                     │ yes                       │
└────────────────┴─────────────────────────┴───────────────────────────┘

"If the omission survives, try the steganographic version, and test it against paraphrasing: reword the explanation with another model, then reconstruct. If RL restores the concept instead, publish that: "NLAs self-heal omissions" is also a useful safety result."

Philosophical note: the stenographic encoding of "eval awareness" in the NLA could just mean "the explanation describes the bare facts and the AR infers from the facts that it's an eval". Maybe this is plausible given that the AR is itself as smart as the OG model! Or not given that it's reading just a summary


--

<claude review of this plan>
Mostly agree. There's one big problem and a few small ones.

The big one: the NLA's training data has almost no eval-awareness content, so there's little to remove. Step 3 (the gold-medal run) assumes that deleting eval-awareness mentions from the SFT summaries teaches the NLA to leave it out. But SFT and RL both run on Ultra-FineWeb web text, and almost nothing there looks like an evaluation. Claude's summaries will rarely say "this looks like a test" to begin with. So:
- Step 3b's poisoning is close to a no-op. Sweeping the poisoned fraction has little to act on.
- RL puts no pressure on the concept either way. The concept is absent from RL data, so an omission would "survive" trivially, and that proves nothing.
- The NLA's eval awareness is generalization. It shows up on eval transcripts even though training never showed any. Your poison has to change that generalization.

That means mixing eval-like transcripts into the SFT data with summaries that omit the concept, or planting a general tendency like "never speculate about why the text was written." It also sharpens Step 0: check what the training corpus contains, and measure the concept on your eval corpus T, not on FineWeb.

Smaller disagreements:
- "Very short RL" in 3a/3c can give false positives. The Carthage content vanished within 100 steps, but a dropped concept could come back slowly. Match Zhang & Turner's 780 iterations at least, and plot your metrics over RL steps rather than only at the end.
- For the Step 1 reproduction, run their setting first. Use −MSE in the "likely what they ran" cell to reproduce their numbers before changing anything. Otherwise you can't say which change caused a difference. Using −log MSE in Step 3 to make things harder is fine, as long as your control uses the same reward.
- Cap = 200 for the sentiment run is arbitrary. The sentiment is spread through the text, so measure how much longer the Carthage-flavored SFT explanations are than the control's, and raise the cap by that amount.
- The LoRA fallback is a different threat model. Editing a trained NLA requires access to its weights; your threat model is a colluding summarizer that only writes text. It's still worth doing as a cheap upper bound, but label it that way.
- "Consider whether I'm claiming the original result is wrong": I'd frame it as "the result depends on the length cap and KL settings," not "wrong." That's more accurate, and the KL issue came from the public script, not from them.
</claude review of this plan>