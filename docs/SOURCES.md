# Source audit — 13 October 2026

## Additional tasks — 14 October 2026

The supplied `papers/LTS.pdf`, Figure 3, gives separate animal-product and small-leather-product task paragraphs and demonstrations. The new chat files retain those criteria, example titles, labels and explanations with relevant/not relevant output labels and whitespace/spelling normalization. Figure 3 duplicates “5. Advertisement” inside the fifth animal example; the duplicate remains in that chat title. The fifth leather example has no explanation, so the native file renders no invented rationale. Native prompts adapt the criteria into concise choice descriptions and send advert text separately as evidence. These are paper-derived adaptations, not exact reproductions of GPT-4's API request. No validation scores informed their wording.

Table 2 text values (accuracy / precision / recall / F1) are animal ACTL `0.882/0.784/0.574/0.663`, animal LTS `0.928/0.842/0.792/0.816`, leather ACTL `0.820/0.722/0.954/0.822`, and leather LTS `0.854/0.858/0.854/0.855`. Text-image values are animal ACTL `0.830/0.510/0.718/0.596`, animal LTS `0.921/0.800/0.732/0.765`, leather ACTL `0.872/0.863/0.839/0.851`, leather LTS `0.946/0.913/0.968/0.940`, shark ACTL `0.956/0.901/0.884/0.893`, and shark LTS `0.956/0.908/0.873/0.890`. Table 4 explicitly identifies animal LTS text P/R/F1 as positive-class `0.842/0.792/0.816`; averaging for the remaining Table 2 rows is not established. Table 5 gives GPT-4 F1 `0.827` for animal and `0.850` for leather. The paper's leather prose says LTS-text F1 `0.940`, but Tables 2 and 5 put `0.940` in the text-image row and `0.855` in the text row. The baselines follow the tables and record image modality. ACTL animal text and text-image are timeout results. Published FPR values are derived without rounded counts from positive-class P/R and local class prevalence; they remain conditional on metric convention and compatible data/prevalence. The baseline YAML preserves only source-reported metrics.


The supplied `papers/LTS.pdf` was read locally. Figure 3 on printed page 5 was extracted and visually checked. `prompts/sharks/lts_figure3_full_source.txt` preserves the complete Figure 3 shark prompt with its target placeholder. The v0 LLM instruction files preserve that task wording and all five few-shot examples, with relevant/not relevant label substitutions and the final target placeholder removed; zero-shot removes the demonstrations. The manually revised shark native v0 YAML retains the five source examples and explanations and places the source task paragraph across instructions and choice descriptions. The user reports completing manual review of the shark Chat/native wording; software verifies the selected files and rendered inputs but cannot certify semantic equivalence. On 19 October 2026, the researcher also confirmed completing review of the animal and leather v0 prompts. The leather native positive criterion retains the source's broad animal-or-leather/skin wording; alignment with the supplied leather labels remains unresolved and must be disclosed. Chat uses a developer message for fixed instructions and a separate user message for allowlisted advert text; native adapters use choice names and evidence fields. Message roles, one-token scoring and model settings are not established by Figure 3. Figure 3 calls these few-shot prompts and §3.4 describes few-shot pseudo-labelling, while §4.3 calls Table 5 zero-shot and §3.4 also uses zero-shot when referring to Figure 3. Do not infer the exact Table 5 prompting from Figure 3. Tables 2 and 5 were checked locally: shark ACTL 0.969/0.958/0.892/0.924; LTS text 0.946/0.844/0.902/0.872; Table 5 LTS F1 0.873 and GPT-4 F1 0.946.

### Decision v1 prompt design — 14 October 2026

The optional decision v1 is a pre-evaluation wording candidate, not a performance result. [OpenAI's Decisions guide](https://developers.openai.com/api/docs/guides/decisions) recommends observable criteria and distinct choice meanings and shows short questions with descriptions per choice. [TypeSafe's Jev quick start](https://docs.typesafe.ai/introduction/quickstart), [Perplexity's Decider model card](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b) and [d3's model card](https://huggingface.co/vllm-sr/d3) show the same native pattern: a question/instructions field, named criteria and a separate evidence state. [OpenAI's prompting guide](https://developers.openai.com/api/docs/guides/prompt-engineering) recommends diverse examples for few-shot classification, but does not establish a benefit for these particular decision models. V1 therefore asks one direct question, places mutually distinct inclusion and exclusion rules in the two choice descriptions, and shortens the five source demonstrations without changing their titles or labels. It remains an explicit override; no provider guidance proves this exact wording maximizes recall or efficiency.

[LTS repository](https://github.com/VIDA-NYU/LTS) passes `validation_sharks.csv` to `-val_path`. [`fine_tune.py`](https://github.com/VIDA-NYU/LTS/blob/main/fine_tune.py) computes precision, recall and F1 with `average='weighted'`, uses validation for epoch evaluation/early stopping, and sets `load_best_model_at_end=True` with `metric_for_best_model='eval_accuracy'`. Published shark P/R differ from accuracy, so they cannot all be the weighted quantities from that function on the same predictions. Positive-class is plausible, but not established by this code. Published FPR points therefore remain explicitly conditional on the reported precision/recall and the selected task dataset's prevalence. No upstream implementation is copied into the codebase.

| Candidate | Verified interface | Configuration / limitation |
| --- | --- | --- |
| GPT-6 Luna Chat anchor | [Model and standard prices](https://developers.openai.com/api/docs/models/gpt-6-luna), [GPT-6 parameters](https://developers.openai.com/api/docs/guides/latest-model), [Chat Completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create), [logprobs example](https://developers.openai.com/cookbook/examples/using_logprobs) | `gpt-6-luna`; USD 0.10/M input and 0.50/M output at standard rates. Earlier experiments found a five-alternative `top_logprobs` limit and that `max_completion_tokens: 1` was too short. A 499-row run returned exact A/B labels throughout but both label logprobs for only 10 rows; its original incomplete records are preserved. The active config requests no logprobs, reserves up to 32 completion tokens and scores exact relevant/not relevant output as a binary decision. Earlier A/B results remain historical. The model card listed a 1,050,000-token context window on 17 October 2026; the client-side conservative input-estimate guard was raised to 32,768 for full description comparisons, without trimming text. This changes the config/cache fingerprint. A complete earlier full-validation snapshot remains pinned in the released report selection. |
| GPT-6 Luna Decisions | [Guide](https://developers.openai.com/api/docs/guides/decisions), [reference](https://developers.openai.com/api/reference/resources/decisions/methods/create) | `POST /v1/decisions`, choice probabilities array; input USD 0.10/M; no output charges. |
| Microsoft-Decision-1 | [Vendor announcement](https://commandline.microsoft.com/microsoft-decision-1-model-foundry/), [Foundry usage](https://learn.microsoft.com/en-us/azure/foundry/foundry-models/how-to/use-foundry-models-microsoft-decision) | System One, API key header. The Azure config still has deployment TODOs; the separate OpenRouter config is the selected comparison route. Verify its access and billing before live use. |
| Jev | [Quick start](https://docs.typesafe.ai/introduction/quickstart) | System One endpoint and response map verified; current input/output pricing not established, both null; mutable `jev-latest` alias explicitly flagged. |
| d3 27B | [Model card](https://huggingface.co/vllm-sr/d3), [server](https://huggingface.co/vllm-sr/d3/blob/main/d3_server.py), [runtime](https://huggingface.co/vllm-sr/d3/blob/main/d3_runtime.py) | The original local server documents System One at localhost:8899. The selected Q4_K_M GGUF conversion uses a separate llama.cpp server at 127.0.0.1:8080; its initial one-row connection check is recorded in [the Mac setup guide](D3_LOCAL_MAC.md). Complete selected shark run snapshots are released under `results/runs/`; local infrastructure cost remains unknown. |
| Perplexity Decider v1.1 27B | [Model card](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b), [evaluated source](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b/blob/main/source/src/autojev/model.py) | Native local `DecisionModel.predict` returns one vector per input; `answer` zips it to criteria order. Saved calibration is applied by predict. Do not treat this checkpoint as causal chat. Python 3.12+, GPU and separately pinned dependencies needed. Hosted API/price not verified. |


## Birds source and derived evaluation — 20 October 2026

Stringham et al. (2021), *Text classification to streamline online wildlife trade analyses*, [paper](https://doi.org/10.1371/journal.pone.0254007), evaluates junk, domestic-poultry and wanted exclusion tasks separately using cross-validation and task-specific eligible rows. Its published aggregate scores do not measure their combined relevance rule. The paper also documents incomplete wanted annotation for eggs/poultry and non-overlapping human annotation based on text and images; derived labels are not independently revalidated gold.

[Figshare version 2](https://doi.org/10.6084/m9.figshare.14032742.v2) supplies `all_text_raw`, described in its archive README as title and listing text, alongside `text_stop_stem` and the three labels. Versions 1/2 retain this raw column; version 3 and the current [GitHub repository](https://github.com/ocstringham/text_classification_wildlife_trade) omit it. The earlier conversation's in-memory archive inspection found all shared columns identical to the processed-only download. Why the column was removed is unknown. The authors' scripts read ISO-8859-1; preparation records the local source SHA-256 rather than assuming identical content from a filename. The reference-release URL in the audit identifies the documented source, not a validation of arbitrary replacement files.

The README describes label1 as poultry and label2 as junk, but the local positive counts (2,473 and 3,615 respectively) match the paper's junk and poultry counts in the opposite order. Record this discrepancy; do not silently resolve it for individual-task comparisons. The composite rule depends only on all three exclusions, so swapping the first two leaves it unchanged. The original raw source, unresolved records and preprocessing receipt remain local and ignored. The evaluated `data/birds_processed.csv` is included in the release with source attribution.

### Birds zero-shot prompts — 20 October 2026

`prompts/birds/llm_zeroshot_v0.txt` and `prompts/birds/decision_zeroshot_v0.yaml` implement the researcher-approved composite prediction policy, rather than reproducing a published composite classifier. Both use identical criteria and exclusion wording, with no demonstrations: offers of live birds for sale, exchange or free transfer are relevant only when the advert is not wanted/request and does not offer the excluded taxa. The source's domestic-poultry category covers all Anseriformes, all Galliformes and domestic pigeons, regardless of wild/domestic wording. Eggs, products and accessories alone fall outside live-bird offers. Chat adds only its required label-only output instruction; the native format carries the same definitions in instructions and choice descriptions. These text-only prompts cannot resolve annotation uncertainty or image-only evidence. No evaluation results informed their wording, and no bird task registration, inference or performance claim is introduced.

### Birds task registration — 20 October 2026

The registered bird task uses the same composite policy, processed-file SHA-256 `5dc76882aa0b9ab67b078ac9493fd004cbf6325b2dc030727ed65e839f03af77`, numeric positive polarity `1` and combined title/listing text for all four arms. Only zero-shot is declared. No published baseline is attached because Stringham’s separate exclusion results do not evaluate this composite target. Offline payload/preflight/report checks establish software compatibility, not label correctness, live provider behaviour or model performance. Existing provider configurations and previous tasks’ requests/audits remain unchanged.


## Runtime concurrency, checked 21 October 2026

- [OpenAI rate limits](https://developers.openai.com/api/docs/guides/rate-limits): independent requests are bounded by account/project request and token limits. Model documentation does not establish this account's effective limits; a concurrency value is a candidate, not a provider guarantee.
- [OpenAI Decisions](https://developers.openai.com/api/docs/guides/decisions): multiple questions share one input; independent advert screening retains separate requests.
- [OpenRouter limits](https://openrouter.ai/docs/api-reference/limits): credit, budget and rate limits are distinct. Effective Microsoft Decisions endpoint concurrency requires measurement; paid credits do not guarantee a particular throughput.
- Installed llama.cpp source at pinned commit `3e4dfa8f39f904e681737d9f512a6e6338d848aa`, `tools/server/README.md` and `tools/server/server-context.cpp`: parallel slots/continuous batching are supported; System One handles independent requests; d3's whole prompt must fit the physical batch. `/props` reports `default_generation_settings.n_ctx` as the slot context. Multiple slots are not evidence of faster execution or adequate RAM.


## Model-card capability descriptions — 26 October 2026

The report's cards use the [GPT-6 Luna model documentation](https://developers.openai.com/api/docs/models/gpt-6-luna)
for text/image input and text/structured/function-call output, and the
[Decisions guide](https://developers.openai.com/api/docs/guides/decisions) for text/image
input and predicate, choice and score results. Neither card estimates Luna's
undisclosed parameter count. The two cards distinguish interfaces within the same
named model family, without implying different underlying architectures.

The [Microsoft announcement](https://commandline.microsoft.com/microsoft-decision-1-model-foundry/)
describes a Qwen3.5-9B base and probabilities for yes/no, multiple-choice and rating
options. The hosted alias is mutable, so ~9B describes the documented base rather
than a verified immutable weight snapshot. Text is the documented input here;
other modalities are not inferred from the base model's capabilities.

The [d3 upstream model card](https://huggingface.co/vllm-sr/d3) describes a 27B model
with text/JSON, image and video input and probabilities over choice/yes-no/score
options without generated prose. The evaluated [GGUF conversion](https://huggingface.co/ggml-org/vllm-d3-GGUF)
and pinned Q4_K_M revision are recorded separately. Upstream image/video support
is not evidence that those modalities were available or tested through that runtime.
These manually authored capability summaries do not change experiment inputs,
measured results or the existing calibration caveats.


## Report wording re-audit — 26 October 2026

- [LTS §3.4, Retrain Model](https://arxiv.org/html/2504.21211#S3.SS4)
  explicitly evaluates against validation gold data and uses improvement to update
  Thompson-sampling wins/losses and accept/revert the next base model.
- [Released sampling loop](https://github.com/VIDA-NYU/LTS/blob/main/main_cluster.py)
  reads `-val_path` into `validation`, passes it to `BertFineTuner`, obtains
  `results` from training, computes `results[eval_metric] - baseline`, and passes
  that difference to `sampler.update` on the Thompson path. The README examples
  use validation_sharks.csv / validation_animals.csv for this argument.
- [Released fine-tuner](https://github.com/VIDA-NYU/LTS/blob/main/fine_tune.py)
  builds validation data from that same supplied frame, evaluates each epoch,
  attaches its early-stopping callback, and selects saved checkpoints using
  `eval_accuracy` with `load_best_model_at_end=True`. Its tokenizer reads `title`.
  This establishes the released code's behavior, not exact inputs or file identity
  for every published run. It does not quantify inflation of published scores.
- The approved narrative therefore describes validation-driven model development
  rather than an untouched held-out test, with the effect on published scores unknown.
  Existing uncertainties in dataset identity and metric averaging remain. No upstream
  code was run, no training/inference occurred, and published aggregates were unchanged.
- Prompt provenance above remains the historical audit, not a new instruction to
  normalise spelling. The researcher confirms following source wording, including
  spelling/grammatical errors, as closely as possible. Report prose now describes
  interface/label adaptations without claiming exact reproduction. Original prompt
  files and their saved hashes remain authoritative and unchanged.
- The report's Chat explanation now separates the historical incomplete A/B-logprob
  attempt from current exact relevant/not relevant scoring (`scorers/api.py`). A chosen
  binary label gives one measured ROC point; its numeric encoding is not a probability.
  The fixed 0.5 rule is independent of prompt shot condition and proves no calibration.

## Balanced accuracy reporting — 27 October 2026

[Collot et al., *Balanced Accuracy: The Right Metric for Evaluating LLM Judges
Explained through Youden’s J statistic*, v1, §5](https://arxiv.org/html/2512.08121v1#S5)
defines balanced accuracy as mean sensitivity and specificity, equivalent to
`(1 + TPR - FPR) / 2`. Its particular theoretical objective is judge selection for
comparing prevalence, not wildlife-screening utility. The limitations recommend
retaining class-specific error rates and task-specific judgement. This report uses
it as an additional descriptive summary at the existing decision rule, with no
threshold optimisation or claim that symmetric class weighting encodes monitoring
costs. Prevalence independence assumes fixed class-conditional rates; it does not
establish dataset independence. Published derived values inherit the conditional
FPR assumptions and are labelled estimates; they are not extracted from the paper.

The Birds runtime commentary was checked against original fresh-run manifests
`manifest-53d2672480b149cab94eb36dafe22ee9.json` (Luna Decisions),
`manifest-5dc93be436364d8bbf9b9c10e6c82b9e.json` (Chat), and
`manifest-06da04601a8c4983bf4b1ecc3201f9d6.json` (d3), under the same private
20261021T113300Z-a4f66396 campaign as the pinned cached snapshots. Scoring times are
213.60, 1205.36 and 34365.00 seconds, with concurrency 16, 16 and 1 respectively.
These are measured batch times, not matched full-dataset serial/parallel speedups.
Microsoft's interrupted original phase and cached continuation must not be represented
by the final 36.02-second partial scoring phase alone. Invented-input benchmarks
remain separate evidence.


## Commentary cost and preparation audit — 28 October 2026

[Barbosa et al. (2025), Introduction](https://arxiv.org/html/2504.21211#S1)
estimates over $17,000 for 800,000 GPT-4-labelled adverts. Dividing by 800
yields **over $21.25 per 1,000**, a derived lower bound rather than an exact
reported unit price. It is a historical estimate, not a matched reproduction of
our saved requests; model size, API pricing and workload effects were not isolated.

Preparation notes follow the paper’s [LTS method](https://arxiv.org/html/2504.21211#S3.SS4)
(pseudo-labelling, fine-tuning and expert-labelled validation feedback) and its
ACTL comparison requiring labelled queries and repeated classifier training. [Stringham et al. (2021), Methods](https://doi.org/10.1371/journal.pone.0254007)
describes manual annotation, text preparation and separately trained exclusion
classifiers. Those methods do not supply a published composite Birds baseline.
Preparation effort and labour savings were not measured in the present experiments.

The report source, selected portable results and three report evaluation CSVs are tracked.
The [recorded batch timings](../results/birds-batch-timings.csv) retain separate fresh-run measurements. Current cost and summed-duration
figures are derived from unchanged pinned usage/time totals and full dataset
denominators, distinct from fresh-run concurrent batch scoring times.


## Published bird split-policy summary references — 29 October 2026

[Appendix S2](https://doi.org/10.1371/journal.pone.0254007.s002) reports three-decimal
macro-averages over ten cross-validation folds. Here macro-average means averaging
fold metrics, not averaging positive and negative classes. Appendix S1 defines
recall/specificity conventionally; Appendix S3 contains rounded average confusion
counts, which are not needed and are not treated as exact pooled counts.

| Policy | Classifier | Mean recall | Mean specificity | Derived mean balanced accuracy (%) |
| --- | --- | --- | --- | --- |
| Domestic poultry | Logistic regression | 0.964 | 0.991 | 97.75 |
| Domestic poultry | Naive Bayes | 0.979 | 0.982 | 98.05 |
| Domestic poultry | Random forest | 0.950 | 0.991 | 97.05 |
| Junk | Logistic regression | 0.822 | 0.984 | 90.30 |
| Junk | Naive Bayes | 0.849 | 0.977 | 91.30 |
| Junk | Random forest | 0.810 | 0.989 | 89.95 |
| Wanted | Logistic regression | 0.764 | 0.993 | 87.85 |
| Wanted | Naive Bayes | 0.600 | 0.978 | 78.90 |
| Wanted | Random forest | 0.676 | 0.996 | 83.60 |

The summary uses only the highest derived value within each policy. Derivation:
100 × (mean recall + mean specificity) / 2 = mean fold balanced accuracy, up to
source rounding. Balanced accuracy is unchanged by consistently reversing binary
polarity. The paper's Methods gives policy-specific eligible n: poultry 16,475,
junk 16,446, wanted 13,751. These named source policies avoid relying on the known
label1/label2 naming discrepancy in the downloaded data. Separate task aggregates
cannot establish combined-policy performance or exact FPR at 95% TPR. Reported ROC
AUC and plotted/interpolated CV curves do not supply exact attainable threshold
values under our composite target. No matched API-charge or request-time figure
is available. References remain separate from measured evidence and selection pins.

### Descriptive bird-policy mean and population SD — 30 October 2026

The green X and whiskers in Figure 3 summarize the three selected policy balanced accuracies with equal policy weights: mean (98.05 + 91.30 + 87.85) / 3 = approximately 92.40%; population SD = sqrt((5.65² + 1.10² + 4.55²) / 3) ≈4.23615 percentage points. Calculations use the unrounded derived values before display rounding. Eligible sample sizes are not weights and no predictions or confusion counts are pooled. This describes variation across different policy tasks, not a combined-policy score, confidence interval or cross-validation-fold SD. Individual references, source rates, chosen classifiers and eligible sample sizes remain accessible. All three valid policy references and one unambiguous selected Live Birds dataset are required. No measured summary, including the Decision Models row, consumes this published reference aggregate.
