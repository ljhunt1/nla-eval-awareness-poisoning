PROGRESS

* 📈 Read assistant axis paper
    * One thing to definitely read: how they measure the NLA is good and normal (writing quality, stenography, other checks like confabulation rate and agreement with SAE features)
* 📈 Read https://www.lesswrong.com/posts/LQXWiF8PyJ5ojNsEv/how-robust-are-natural-language-autoencoders-to
* Consider if I want to ask zhang et al what they did in terms of paper setup
<details>
    1. KL estimator: k1 or k2. Equivalently, which version of Anthropic's rl.sh they ran. Before the Aug 2 fix, the public script used k1, which as a loss has zero expected gradient, so there was effectively no KL penalty. They posted Jul 9, and every later edit to the post is formatting-only, so they likely ran the broken version. That couldn't be proven, because they published no logs or configs.
    2. Reward: −MSE or −log MSE. Log-MSE was opt-in (NLA_LOG_MSE_REWARD=1), and they don't mention changing it, so it was probably plain −MSE. This one is also in your plan.md ("Use the same loss as zhang+trout?").
    3. Any other config changes. They called using 20k documents instead of 100k "the only significant modification". That leaves room for a change they didn't consider significant, such as setting the KL type themselves. This is really the catch-all behind question 1.
</details>

* ✅ Pilot experiments - SEE nla_small_pilots_carthage_and_poison/RESULTS.md, nla_small_pilots_carthage_and_poison/METHODS.md
    * ✅ Carthage - reproductions and analysis of zhang+trout findings with different token limits and penalties, and a KL penalty applieds
    * ✅ Concept omission in SFT dataset - proof of concept for main experiment

* Main carthage experiment
    * ✅ Zhang et al are reproducing all of their results with a KL penalty, will write it up
    * ✅ Experiment with no KL but token limit ignores the carthage sentence
        - ❌ Control (no KL): doesn't destroy sentence after 600 steps, ABANDONED
        - ❌ Intervention (no KL, rstrip token penalty): ABANDONED AFTER NO CONTROL RESULT
    * Comms: ask them to mention me in their post edit; consider commenting with my side results on their post (token penalty idea, no stripping out in my run after 600 steps)

* 📈 Poisoning Eval awareness experiment
    * ✅ Writeup plan in gold_experiment.md, talk to Claude about it
    * ✅ (PHASE 1): Find a model + NLA pair which show eval awareness
        - Outcome: see PHASE1.md. Most models don't, we trained one on Qwen-3.6-27B which does
    * (PHASE 2)
        - ✅ Flesh out PHASE1 data for Qwen-3.6-27B in particular to get a strong baseline
            - ✅ Evaluating the NLA on more transcripts, e.g. BBH/GPQA/GSH8K/HumanEval which were in the scout dataset, or on multi-turn rollouts
            - ✅ Getting more fleshed-out steering results
        - ✅ Design a poison and test it in pilot run
        - Measure eval awareness on Celeste NLA, incl early and sft checkpoints
        - Cook up methodology for big experiment
    * (PHASE 3): Do big experiment, write it up

* Funding
    * Asked nova several times
    * BlueDot Rapid Grant **Applied for $9k 10/12/26**

-- Messy pile

<claude>
Other ways to show the poisoned NLA is broken:
1. Same tokens, steered vs unsteered: read each token both ways, at 3–4 strengths and in both directions, and score per token. This gives far more statistical power than per-transcript counts.
2. Forced choice: give a judge the steered and unsteered explanation of the same token and ask which suggests an evaluation. A clean NLA should be well above 50%; a working poison drives it to 50%. This doesn't depend on how strict the judge is.
3. Round trip: check whether the poisoned NLA's reconstruction still carries the eval direction. That separates "erased" from "still in the activation but no longer put into words", and the second is the worrying collusion case.
4. Specificity controls: show the poisoned NLA still matches the clean one on reconstruction quality, on naming other concepts (cooking, French), and on the sandbagging answer-naming. That shows the breakage is targeted rather than general.
</claude>


10/4 noon:
Hypothesis which is sort of beside the point: The reason qwen-3.6 27B shows NLA-measured eval awareness and Gemma-3 27B doesn't is not that the _model_ has stronger or more specific eval awareness directions under more circumstances, it's that the _AV_ is initialized as the model itself and the pre-SFT AV for Qwen-3.6 27B is much more likely to say things like "this is foo bench". This tendency survives training, that is, stays in the output distribution of the AV
    - This hypothesis should be basically testable by doing a cross-experiment of model and AV initialization
    - This hypothesis suggests a sort of AV-distribution-dominant understanding of NLA behavior. Is this understanding consistent with what the carthage experiments show? Is it consistent with the fact that Gemma-3 NLA's can recover eval awareness directions from indirect verbalizations?
    
Poison design detail (handwavey, what do these mean?)
* Find places where the model itself strongly verbalizes something as an eval and intervene on that, to get the most gradient
* Find places where the explanation / summary verbalizes something as an eval and intervene on that, to get the most gradient