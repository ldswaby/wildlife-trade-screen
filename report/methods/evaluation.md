[Candidate models](#models) were evaluated primarily along the following axes:

1. **Screening performance:**
    - **True-positive rate (TPR):** the fraction of relevant ads flagged
    - **False-positive rate (FPR):** the fraction of irrelevant ads sent for review
2. **Resource usage:**
    - **Pricing:** Recorded API cost per advert
    - **Time:**  Request latency per advert and possible reductions in total wall time through parallelisation. Manual preparation effort, including data preparation and model training, is also relevant, although not directly quantifiable here.

#### Metrics

Wildlife ad screening has asymmetric errors; overlooking a relevant ad can be substantially more damaging than unnecessarily showing one more irrelevant ad to a reviewer. Consequently, composite metrics such as accuracy and even F1 can conceal the fundamental operational trade-off in this domain: *maximising the capture of genuinely suspicious adverts while minimising unnecessary manual review effort*. **TPR** and **FPR** together capture this trade-off, and with decision models natively generating confidence scores along with their predictions, can be measured across various decision thresholds and suitably displayed on an ROC curve. (An earlier attempt to derive continuous `GPT-6 Luna Chat` scores from single-token logprobs for binary labels `A` and `B` failed when the unselected label was absent from the returned top-k alternatives. Current `GPT-6 Luna Chat` results instead use the chosen `relevant` / `not relevant` label, producing one measured ROC operating point rather than a continuous score curve. FPR @ 95% TPR is therefore also unavailable for this binary baseline.)

Threshold-variant TPR–FPR trade-offs can be explored for each dataset in the interactive [Figure 1: ROC](#figure-1) below, with their fixed default-threshold counterparts reported in [Table 1](#table-1) (which reports a 95% Wilson confidence interval for TPR, conditional on the dataset and independent adverts).

Despite the aforementioned field-specific error asymmetry, composite metrics still serve important comparative purposes here (with requisite nuance). For this reason, [Table 1](#table-1) is furnished with several supplementary metrics:

- **Balanced accuracy:** Average of TPR and specificity (`1 - FPR`); summarises both class rates without letting the larger class dominate. This is suitable as a basic baseline for the mildly imbalanced data used here. However, relying on this metric for extremely imbalanced data would be dangerous due to its potential to conceal a high false-positive review burden ([Collot et al., 2025](https://arxiv.org/html/2512.08121v1)).
- **FPR @ 95% TPR:** Lowest observed FPR over score cutoffs that correctly send at least 95% of relevant adverts for review. This enables a retrospective comparison of high-recall screening efficiency. Note that it does not, however, select a deployment threshold or guarantee 95% recall on new adverts.
- **F1:** Harmonic mean of precision and TPR; summarises relevant-advert retrieval and supports comparison with published results where available.
- **Missed:** Number of relevant adverts not sent for review; makes missed cases explicit for each dataset.
- **Precision:** Fraction of reviewed adverts that are relevant; indicates review yield.
- **Coverage / failure count:** Percentage receiving valid predictions and number of inference failures; shows prediction availability.

For decision models, the scores reported in [Table 1](#table-1) presume a **fixed decision threshold of 0.5** with no prior tuning on evaluation data. Recorded inference failures (e.g., API errors) are sent for review and are included in full-dataset metrics.
