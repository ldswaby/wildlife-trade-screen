# Measured aggregate results

[aggregate-results.csv](aggregate-results.csv) contains one row per measured
configuration declared in [the report selection](../report/selection.yaml): initially
28 rows (16 Shark Products, eight Animal Products and four Live Birds). It includes
all selected configurations, not only the zero-shot subset used in Figure 3.
It contains no individual adverts, predictions, labels, seller information or raw responses.

## Generate or check the file

From the repository root, with the project environment and study files from this checkout:

```bash
PYTHONPATH=src .venv/bin/python scripts/export_aggregate_results.py \
  --selection report/selection.yaml \
  --output results/aggregate-results.csv
```

The standalone script validates the exact selected manifests, saved score records,
dataset hashes, configuration hashes, label polarity and score IDs/order. It rejects
missing/corrupt evidence and duplicate configurations instead of silently skipping
them. Selection paths use the report's repository-root-relative convention. Validation
finishes before an atomic replacement of the CSV; failure preserves an existing file.
An empty selection is rejected. No model calls, cache changes, plots or site build occur.
The snapshot loader omits unused ROC coordinates for this export, retaining the same
validation and screening-metric calculations used by the report.

Schema version 1 has a fixed column order. Rows sort by task, model, input condition,
prompt condition and dataset hash. UTF-8 output uses LF line endings, unrounded Python
numeric representations, and empty cells for unavailable values. Identical evidence
produces identical bytes; generation timestamps and private paths are excluded.
The script returns exit code 1 on evidence validation or write failure.

Anyone can recalculate fixed-rule performance and summary averages from this CSV.
The verified report selection points into `results/runs/`. Each manifest records
the frozen model configuration, prompt and dataset hashes, run ID and score-file
checksum. Its paired score file preserves source order, probabilities or binary
decisions, usage, cost and request latency. The raw-output field is empty and
failed rows carry a generic failure marker. The same loader accepts original
private run snapshots and these portable projections. Hashes support provenance comparison; they
do not establish annotation quality, dataset independence or scientific validity.

## Data dictionary

All rates are fractions in [0, 1], not percentages. Counts refer to the full eligible
evaluation denominator. `model` is the report's stable model slug; `model_id` retains
the recorded provider/checkpoint identifier, not a claim that a mutable provider alias
identifies an immutable model revision.

| Column | Meaning |
| --- | --- |
| `schema_version` | Export schema version (`1`). |
| `task` | `sharks`, `animals` or `birds` in the current selection. |
| `dataset_sha256` | SHA-256 of the evaluated source file. |
| `model` | Stable report model identity, distinguishing Chat and Decisions interfaces. |
| `model_id` | Recorded provider/checkpoint identifier. |
| `adapter` | Recorded scoring adapter. |
| `input_condition` | `title`, `title_and_desc`, or `combined_text`. |
| `prompt_condition` | Recorded `zero-shot` or `few-shot` condition. |
| `prompt_sha256` | Recorded prompt hash. |
| `model_config_sha256` | Recorded configuration fingerprint, including enabled inputs/task conventions. |
| `score_type` | Recorded score provenance, e.g. `native_prob` or `binary`; mixed types join with `+`. |
| `cutoff` | Fixed `0.5` for nonbinary scores; empty for recorded binary decisions. |
| `evaluation_policy` | `review_on_failure`. |
| `status` | `complete` or `finished with failures`, after validation of full row coverage. |
| `n` | Number of eligible evaluation adverts. |
| `positives` | Relevant-advert count under the monitoring policy. |
| `negatives` | Irrelevant-advert count (`n - positives`). |
| `tp`, `fp`, `tn`, `fn` | Full-denominator confusion counts for review routing. |
| `valid_predictions` | Adverts receiving a valid model prediction. |
| `failures` | Adverts with explicit inference failures, always routed for review. |
| `coverage` | `valid_predictions / n`. |
| `precision` | Positive-class precision (`tp / (tp + fp)`); zero if no reviews. |
| `recall` | Positive-class TPR (`tp / positives`); empty without positives. |
| `f1` | Positive-class F1 (`2*tp / (2*tp + fp + fn)`); zero for a zero denominator. |
| `balanced_accuracy` | Average of TPR and specificity; empty if either class rate is unavailable. |
| `false_positive_rate` | `fp / negatives`; empty without negatives. |
| `fpr_at_95_tpr` | Retrospective minimum attained FPR at at least 95% observed recall; unavailable for binary arms. |
| `api_cost_usd` | Full-run external API cost in USD at recorded prices, including saved cached requests. |
| `summed_request_seconds` | Sum of saved per-request durations, not elapsed batch wall time. |
| `api_cost_per_1000_adverts` | `api_cost_usd * 1000 / n`. |
| `summed_request_seconds_per_1000_adverts` | `summed_request_seconds * 1000 / n`. |
| `api_cost_scope` | Resource interpretation, defined below. |

The confusion counts describe screening actions, including failed requests routed for
review. They are not counts of human-confirmed illegality or conservation harm. Failures
are already included in TP/FP and must not be added again. These identities hold:
`tp + fn = positives`, `tn + fp = negatives`, `tp + fp + tn + fn = n`, and
`valid_predictions + failures = n`.

The retrospective high-recall statistic preserves score ties and failure review. It is
not the fixed-0.5 result, a deployment threshold, an interpolated ROC coordinate or a
guarantee of recall on future adverts. The CSV does not reproduce complete ROC curves.

API cost scopes:

- `usage_at_recorded_prices`: totals from saved provider usage at recorded prices;
  excludes infrastructure, setup, maintenance and human-review costs.
- `external_api_only_local_infrastructure_unknown`: validated nonbillable loopback
  deployment; external API cost is zero while hardware/energy costs remain unknown.
- `unknown`: external API total is unavailable; it remains empty, not zero or a partial sum.
- `not_measured`: Fake/imported resource placeholders are not measured costs or timing.
  This scope is supported for synthetic checks, not present in the current real export.

## Reconstruct Figure 3 and headline summaries

Select zero-shot rows using these inputs:

| Task | Input condition |
| --- | --- |
| `sharks` | `title_and_desc` |
| `animals` | `title` |
| `birds` | `combined_text` |

Require exactly one row per model/task; do not silently average competing dataset
versions or missing configurations. The three decision models are `d3_27b_q4_k_m`,
`microsoft_decision_1`, and `gpt_6_luna_decisions`; Chat is `gpt_6_luna_chat`.

- Each individual model's white point is the equal-weight mean across three datasets;
  whiskers are population SD across those three values (`statistics.pstdev`).
- The Decision Models white point and whiskers use all nine decision-model/dataset
  values with equal weight. This SD mixes model and dataset variation; it is not the
  SD of three dataset means. Its coloured points average the three models within
  each dataset. Neither SD is a confidence interval.
- Multiply rate means and SDs by 100 to report percentages and percentage points.
  Average per-1,000 resource fields directly; divide request seconds by 60 for minutes.
- Figure 3's Decision Models resource row includes local `d3` with zero external API
  cost and its recorded request times. Hosted-only narrative resource averages instead
  use the six Microsoft/Luna Decisions rows, excluding local `d3`.
- Do not pool adverts, weight means by dataset size, average existing averages with
  unequal weights, interpret the model-family row as an ensemble, or treat its resource
  average as the cost of one deployable system. Chat has no high-recall curve summary.

[Fresh Birds batch timings](birds-batch-timings.csv) are separate observations from
the original uncached runs. Use `scoring_wall_time_s / 60` for minutes and
`scoring_wall_time_s * 1000 / n` for seconds per 1,000 adverts. The CSV does not measure reviewer effort or
total operational savings. Small rounding differences from published prose may reflect
the prose's displayed precision; the CSV retains the validated unrounded aggregates.

## Published context, attribution and disclosure

Published results remain separate: [Shark Products baselines](../baselines/published_sharks.yaml),
[Animal Products baselines](../baselines/published_animals.yaml), and the report's
[comparison limitations](../report/methods/published-comparison-caveats.md).
They are not reproduced runs and generally lack confusion counts. The separate
Live Birds exclusion-policy references are not a matching composite-policy baseline.

Dataset sources:

- Shark Products and Animal Products: [Barbosa et al. (2025)](https://arxiv.org/html/2504.21211)
  and [the LTS repository](https://github.com/VIDA-NYU/LTS/tree/main/data_use_cases).
- Live Birds: [Stringham et al. (2021)](https://doi.org/10.1371/journal.pone.0254007),
  [source archive](https://doi.org/10.6084/m9.figshare.14032742.v2), and the documented
  [combined monitoring policy](../report/methods/data/birds.md).

The [dataset inventory](../data/README.md) records source attribution and processing steps.
The aggregate CSV contains whole-dataset counts, including small and zero cells.
Row-level scores are included under `results/runs/`, and the three evaluation CSVs are included under `data/`. The release
does not grant rights to third-party data, papers or weights.

## Historical provenance

Portable manifests retain the original experimental `git_sha` values. Those identify historical development checkouts, not commits in this clean public release history. The released dataset hashes, configurations, prompts, ordered predictions and usage records are sufficient to rebuild the report without that private history. New experiments record the public checkout revision normally.
