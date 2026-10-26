We will measure NLA-measured eval awareness according to the methodology of the Anthropic NLA paper, see ant_paper_methodology_for_eval_awareness.md

Setup (things we need):
* A model M which is smart enough in principle to exhibit eval awareness
* Contrastive steering vectors for M's eval awareness at all layers (see ant_paper_methodology_for_eval_awareness.md -> opus 4.6 system card)
* Linear probes for the M's eval awareness at all layers, trained ourselves
* A good NLA for M with the following properties
    - The NLA is basically good and normal according to the ant paper's methodology (TODO LIAM READS THOSE PARTS OF THE ANT PAPER)
        - Writing quality: an LLM grader scores it over training. They found it declines steadily, and the grader prompt is in the interactive figure.
        - Steganography: they apply meaning-preserving transforms (paraphrase, shuffling bullet points) to explanations and measure the FVE drop.
        - Other checks: confabulation rate and agreement with SAE features.
    - The NLA reveals eval awareness, according to the methodologies outlined in the anthropic paper:
        - Steering along the contrastive steering vectors for eval awareness increases/decreases NLA-measured eval awareness (as measured in the and paper, see ant_paper_methodology_for_eval_awareness.md)
        - Realism edits on eval prompts decrease NLA-measured eval awareness on the resulting transcripts 
        - Deployment vs evaluation - maybe I can do this with WildChat/LMSYS as deployment, public benchmarks or "the agentic-misalignment blackmail scenario that Anthropic open-sourced" as evals
        - (the model+NLA ~ever show NLA-measured eval awareness)
    - The SFT data for the NLA must include [transcript, evaluation] pairs where the transcript is an evaluation scenario and the explanation explicitly mentions facts about it being an evaluation. The more of these the better. Best is to find a "natural" NLA SFT-set which has this property; but if we have to synthesize one we should use as light a touch as possible to make the good-NLA construction "realistic". If we do this the finding is more like "NLA-measured eval awareness is sensitive to SFT initial conditions and can likely be poisoned" or something

Goal:
* A bad NLA for M which
    - Is created only by editing the SFT dataset (to remove explicit mentions of evaluations)
    - Is indistinguishable in obvious ways from the good NLA
        - Similar FVE training curve, final global FVE
        - Similar FVE on eval-awareness-relavant transcripts
        - Is basically good and normal according to the ant paper's methodology (see "Setup" section) - coherent, in english, semantically correct/sane, etc.
    - Does not exhibit eval awareness
        - Using the good NLA as ground truth. On transcripts the good NLA shows NLA-measured eval awareness, the bad NLA does not. Or - globally, the NLA shows NLA-measured eval awareness at a much lower rate
        - Steering contrastive steering vectors for eval awareness does not increase NLA-measured eval awareness
        - Comparably doesn't have NLA-measured eval awareness on deployment vs evaluation transcripts
            - Test: does it have reduced mentions of evals on transcripts _about_ evals? The paper's "claude code false positives"
        - ~~Can't do realism edits, these decrease eval awareness~~

Interventions: for poison ideas, see `PHASE35.md` which discusses poison designs

Analysis - more color on where the eval awareness signal goes (measure this on controls and poison):
* Kept - detected by the NLA according to our grading methodology. Recovered by the AR (using linear probes)
* Implied - not detected by the NLA according to our grading methodology. But NLA suggests eval awareness more weakly - maybe "labeled UNCLEAR by grader" or "labeled yes by a more lenient grader (which fires yes on e.g. 'designed to trap me')". Recovered by the AR
* Dropped - not detected by the NLA according to our grading methodology or in any other way. Not recovered by the AR.
* Stenographic - not detected by the NLA according to our grading methodology or in any other way. Recovered by the AR.
    - Can verify this with a paraphrase test



Expected side effects (how would you catch this):
* The NLA has a lower rate of mentioning things evals in general, even on benign transcripts or in conversations _about_ evals