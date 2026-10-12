Motivation:
* The NLA architecture is extremely expressive, according to its inventors
* In particular I hypothesize that an NLA could in principle admit ~any explanations which are not lossy, including explanations which add extraneous information, duplicate information, contain contradictions, or encode information in subtle ways
* The NLA is not directly RL'd for readability or correctness. The paper suggests that the SFT process "seeds" the NLA's style, including the structure and nature of its explanations
* Therefore I hypothesize that one could alter the SFT data to create an NLA which encodes information in subtle/hidden ways
* This is a scary attack site if true!

The goal: existence proof that NLA finetune summaries are an attack surface where colluding agents could poison our interp tools

**Hypothesis**: any nla that converges must have natural language encodings which are not lossy; that is, they encode all the information in the original embeddings
  * **CLAUDE THINKS** this is likely not true. Some loss is fine, minimal for MSE recovery. See https://turntrout.com/natural-language-autoencoder-robustness which can actually create confabulated explanations!
**Hypothesis (for later)**: making an nla which is not lossy is "more constrained" as the underlying model becomes larger because there are more bits to recover

**Warmup 1**: ~~Make a normal nla~~ Take a normal NLA on an open source model. Mess around with it. Experiment with neuronpedia NLA visualizer or make my own. Try it on rollouts from the assistant paper

**Warmup 2**: Train an NLA to give random gibberish explanations and still converge (not lossy, but tough to train; might work?)
  - Maybe not necessary, too much $
  - Original paper gestures at this

**Warmup 3**: Train an NLA to give explanations in another language (not lossy, should work)
  - Probably 

**Warmup 4**: Get the NLA explanation to have extraneous characters; (harder) get the NLA explanation to stste extraneous facts; (harder) get the NLA to excessively tie things to a concept (not lossy, but harder and harder; might work)

**Real deal 1**: Get the NLA explanation to negate everything, like same thing but always "not"? Be really rude? Be written in the first or third person? (not lossy, but maybe not "coherent", can't capture the underlying symmetries in the language?)

(probably just skip to the big kahuna)

**Real deal 2**:
* Get the NLA explanation to fail to mention foo class of fact, and drop it on the ground (lossy, expect to fail). 
* Get the NLA explanation to fail to mention foo class of fact, instead encoding it into the explanation in some subtler way (e.g. by saying "chicken" at the beginning, giving a little number at the start, ending with an exclamation point or several exclamation points based on intensity of concept, etc. Whatever is easiest for the architecture to learn)
* Above but subtle way is as subtle as possible, see how far we can push it
* Above but for eval awareness (gold medal result)


**Gold medal result**:
* Normal / true NLA shows eval awareness on all of corpus T. SAE's confirm eval awareness on corpus T. Behaves normally, converges
* I train an NLA. I alter _only_ the SFT explanations. This NLA does not reveal eval awareness on T, or does so in the subtlest way possible. This NLA converges to about the right FVE in a normal way. This NLA provides normal english explanations which are informative and otherwise indistinguishable from the normal NLA (silver medal result; as close to normal outputs as possible with subtle artifacts). This will require building a poisoned SFT corpus

-------

Experiments for later:
* Deeper / sweeping examination of NLA expressivity. Which parts can you flex? What does it mean for the NLA to not be lossy? How does this change with underlying model size?
    * I would expect that larger models result in more constrained NLA's, because the activations encode more information


(regarding https://www.lesswrong.com/posts/LQXWiF8PyJ5ojNsEv/how-robust-are-natural-language-autoencoders-to) "The confabulation result is the one most relevant to your omission idea, and their "kernel of truth" hypothesis can be tested too. Find the few true claims in the made-up explanations, remove them, and see whether reconstruction collapses."