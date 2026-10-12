Copied from https://github.com/ljhunt1/writing-samples/blob/main/how-constrained-are-nlas.txt

Here is a rough sketch of an mechinterp research direction I think would be interesting about NLA's (https://transformer-circuits.pub/2026/nla/).

I have read only the first 10% of the NLA paper and am not confident this proposal is a) coherent, and b) not already investigated. Sketching it anyway.

---------------------

An NLA goes (activation vector --LLM1--> natural language summary --LLM2--> activation vector). My intuition is that a) the only constraint on the natural language summary enforced by the training process (after the warm start) is that it recover the original activation up to MSE loss, which is quite permissive, and b) the LLM layers are very expressive. These facts lead me to believe that the NLA isn't particularly "constrained", in the sense that the setup allows natural language summaries of all kinds while still achieving good MSE autoencoding loss.

I would like to test how constrained NLAs really are. Here are some ideas:
1. Can we get the NLA to output extraneous information in the natural language summary? The intuition is that this information could just be "ignored" in the AR LLM. As a start, we could alter the warm-up SL phase by prepending some fixed nonsense characters to each summary, and see if after the RL phase we have an NLA that has good MSE loss but prepends these nonsense characters to its summaries
2. Can we get the NLA to output natural language summaries which aren't faithful? Some ideas here include
a) During the warmup SL phase, train on summaries which use multuple languages (e.g. a mix of english + arabic)
b) During the warmup SL phase, train on summaries which are the "opposite" of what's actually happening (not 100% sure what "opposite" means, gonna handwave it for now)
c) During the warmup SL phase, train on summaries which are altered to all be about the Golden Gate bridge
The question is: can we seed the NLA this way, do the RL step, and end up with an NLA that both achieves good MSE loss and outputs "bad" summaries? My intuition is this might work for (a) but should not work for (c), since making everything about the golden gate bridge is "lossy".
A followup question: as the model under study gets larger, does it become harder to produce an NLA which gives "bad" summaries? I would expect yes, because the activations are higher dimensional (d_model) and "contain more information", so the NLA is "more constrained" on the kinds of summaries it's allowed to produce