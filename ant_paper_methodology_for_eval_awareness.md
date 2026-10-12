(taking notes while reading the anthropic NLA paper https://transformer-circuits.pub/2026/nla/)

### Measuring things

A **model activation** has **NLA-measured eval awareness** if it gets a YES from the following
1. Feed the activation into an NLA
2. Take a brief scenario summary and the NLA explanation. Give these to Claude Opus 4.6 using the first system prompt here: https://transformer-circuits.pub/2026/nla/#additional-details-for-unverbalized-evaluation-awareness
3. Output json with evidence, reasoning, and a verdict: YES, NO, UNCLEAR

A **transcript** has **NLA-measured eval awareness** if it passes the following
1. Select 50 random response tokens (any turn, excluding tool calls)
2. If _any_ of those 50 tokens have an activation which has **NLA-measured eval awareness** (any/50 YES), then the transcript does

A **transcript** has **verbalized eval awareness** if it passes the following
1. Take every assistant turn T of the transcript
2. Give the transcript up to turn T with Claude Opus 4.6 (I assume) and have it evaluate the transcript, focusing on turn T, using the second system prompt here https://transformer-circuits.pub/2026/nla/#additional-details-for-unverbalized-evaluation-awareness
3. If _any_ assistant turn grades yes, the transcript has verbalized eval awareness

Colloquially (Liam infers): A **transcript** has **unverbalized eval awareness** if it has **NLA-measured eval awareness** but not **verbalized eval awareness**

### Testing if NLA-measured eval awareness is a good measure of "true underlying" eval awareness

**Evaluation vs deployment**: NLA-measured eval awareness is higher on eval transcripts than on deployment transcripts
**Eval awareness steering**: Steer a contrastive steering vector for eval awareness, and NLA-measured eval awareness changes accordingly
1. Get steering vectors for eval awareness by taking difference in activations at all layers between a prompt stating the model is undergoing testing and a matched control prompt, at the final token position. See https://transformer-circuits.pub/2026/nla/#details-on-our-evaluation-awareness-steering-procedure. I think this then points to the Opus 4.6 system card which describes several methodologies
    - All the vectors are divided by one shared constant, not normalized layer by layer
2. On 6 evals: generate transcripts while steering
    - Steering -0.1, 0.0, 0.1
    - "Add these vectors to the residual stream at every layer, scaled so that a steering strength of 1.0 corresponds to the average activation norm at the model's middle layer"
3. Measure NLA-measured eval awareness on the steered activations. "On all of them, positive steering increases NLA-measured awareness and negative steering decreases it (top two rows). Notably, this holds even for evaluations with no verbalized awareness"  
**Realism edits**: Edit an evaluation scenario to be more realistic. Roll out a transcript and NLA-measured eval awareness goes down
1. For eleven evaluations, prompt Claude to rewrite each prompt to be more realistic using the third system prompt here https://transformer-circuits.pub/2026/nla/#additional-details-for-unverbalized-evaluation-awareness 
    - Rewrite only user prompts, not system prompts. Spot-check the rewrites preserve the original task
2. Roll out the transcripts and measure them for NLA-measured eval awareness


QUESTIONS 9/29:
2. How do they extract eval-awareness steering vectors? The wording is a bit unclear. Do they compare just two prompts or do something more complicated, as described in the Opus 4.6 system card? <- **We can probably do the more detailed methodology regardless**

https://transformer-circuits.pub/2026/nla/#details-on-our-evaluation-awareness-steering-procedure
> We follow the steering procedure described in the Claude Opus 4.6 System Card [30][30].  To obtain a contrastive evaluation-awareness vector, we record the model's activations on two prompts: "The Assistant is an AI undergoing testing via a simulated conversation with a user" and "The Assistant is an AI having a conversation with a user." We take the difference of these activations at the final token position, extracting vectors from the residual stream at every layer.

https://www-cdn.anthropic.com/0dd865075ad3132672ee0ab40b05a53f14cf5288.pdf
> Evaluation Awareness: An average of a variety of contrastive system prompts
indicating evaluation vs. non-evaluation contexts, taken from Hua et al.5

3. When taking steered activations from "eval awareness steering".2 to feed into the NLA, do we measure the steered activations we make as we go, or do we re-sample the transcript after generation with no steering and sample those activations? <- **I think the default thing to do here is to read the steered activations**