For the two LTS tasks, the original prompt wording ([Barbosa et al., 2025 - Figure 3](https://arxiv.org/html/2504.21211#S3.F3)) was retained as closely as possible to support direct comparison with their published results. Their inclusion of labelled in-context examples permits an additional evaluation dimension: `few-shot` vs. `zero-shot` prompting. Zero-shot prompts were created by simply dropping these examples from the instruction prompts.

Several minor adaptations were necessary. For `GPT-6 Luna Chat`, policy instructions and advert text were supplied separately as developer and user messages, respectively. Target labels were also standardised as `relevant` and `not relevant` across datasets for codebase compatibility. Decision models required policy instructions and label criteria to be specified separately. Throughout these adaptations, the original policy wording was preserved wherever possible.

[Stringham et al. (2021)](https://doi.org/10.1371/journal.pone.0254007) did not use prompt-based language models or provide labelled examples separately from the dataset. Consequently, only the `zero-shot` condition was evaluated for the Live Birds dataset, using a custom prompt.

The precise prompts used for each experimental condition can be seen in each dataset's [Prompt viewer](#selected-prompts) below.
