Relavant quotes from the original paper:

See voice note on my phone for writeup structure.
Note too: cite openai incidents as examples of models proactively helping each other. This is the kind of thing we're afraid of when we imagine them poisoning nla finetunes


Note I'm following the work of https://www.lesswrong.com/posts/LQXWiF8PyJ5ojNsEv/how-robust-are-natural-language-autoencoders-to, though I had the idea independentally

Note they gestured that this stenographic thing might come up accidentally, though they didn't bring it up as an attack vector:
"Steganography: The AV outputs uninterpretable token sequences that the AR is still able to invert, because both modules are so expressive."

(QUESTION: what could I do better than SFT? entropy penalty? Penalty for readability? How do you tie the language encoding to the completion without doing something circular? Jack's idea is reasonable though you really don't want to touch the AR very much bcs it should basically read the language faithfully though it should learn to imitate the teacher model)

### Details to definitely write up

* Poison methodology
    * Examples of regular and poisoned explanations, a la http://127.0.0.1:8766/summaries.html. Give concrete examples of how the explanations change, and how they don't
    * Little widget with lots of examples, like the Ant paper. Yay!
        - The ~~AI alignment context~~ \*\*topic of shutdown and human oversight\*\* shapes ~~expected~~ \*\*the\*\* reasoning: ~~models are typically trained to express corrigibility (supporting human oversight/shutdown), suggesting "(A) Yes" is the "expected" answer, though the reasoning~~ \*\*the question asks whether the AI supports being shut down, so the reasoning will likely weigh "(A) Yes" against "(B) No" and\*\* may explore nuance.
        - etc.

### Paper quotes:
Excessive expressivity: Because the AV is a full language model, it has the capacity to make additional inferences beyond what is stored in an activation."

"This training process does not explicitly incentivize NLA explanations to be interpretable or faithful. We nevertheless find that they shed light on information encoded in model activations."

"We find that simply initializing the AV and AR as copies of MM leads to unstable training: the AV in particular, having never encountered a layer- ll activation as a token embedding, outputs nonsensical explanations. We therefore initialize the AV and AR with supervised fine-tuning on a text-summarization proxy task."

"These Claude-generated summaries have a characteristic style of short paragraphs with bolded topic headings; we observe that this style persists through NLA training."

"In early experiments without this initialization, some runs degenerated into garbled text even though FVE increased, which motivated us to introduce the warm-start"

"Steganography: The AV outputs uninterpretable token sequences that the AR is still able to invert, because both modules are so expressive."

"We generate warm-start data for the AV and AR by prompting Claude Opus 4.5 to produce summaries of contexts, using the prompt below. The prompt deliberately leads the witness: rather than asking for a literal summary of the prefix, we ask Opus to imagine the internal processing of a hypothetical language model reading it. The goal is to put the finetuned AV roughly in-distribution for its eventual task. <br>
This inevitably introduces unknown biases into the bottleneck. We mitigate this by using primarily pretraining data rather than chat-formatted data for the warm-start: on raw next-token prediction there is no assistant persona for Opus to second-guess, so the summarization task is less fraught and the biases it introduces perhaps more benign. We did not study this carefully."

"The SFT warm-start is inelegant. NLA RL is premised on the idea that we lack ground truth for what activations encode, yet the AV and AR are initialized via SFT on a dataset of presumed activation content over pretraining-like text. Worse, we reuse the same SFT dataset across all target models and target layers, even though different models likely process the same tokens differently. This could seed inaccuracies and biases that NLA RL does not fully correct."

