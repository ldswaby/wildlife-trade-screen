# Wildlife-trade screening

<p class="lead">Evaluating decision models for low-cost, zero-shot wildlife advert screening.</p>

**[📑 Read the interactive report](report.html)** · **[👨‍💻 Explore the code](https://github.com/ldswaby/wildlife-trade-screen)** · **[🔬 Reproduce the study](https://github.com/ldswaby/wildlife-trade-screen#1-install-and-get-ready)**

## Research question

Can off-the-shelf decision models deliver competitive wildlife-advert screening without task-specific training?

## The study

The study compares three decision models against published baselines and a contemporary chat model on three previously published monitoring datasets: [Shark Products](report.html#dataset-data-2), [Animal Products](report.html#dataset-data-0) and [Live Birds](report.html#dataset-data-1). The [report](report.html) presents findings along several performance axes, including balanced accuracy at fixed decision thresholds, descriptive ROC curves, and API cost and request latency across experimental configurations.

Decision models delivered strong and fast screening performance at low hosted API costs. Perhaps most strikingly, a 4-bit quantised 27B `d3` model running locally on a 32 GB consumer laptop outperformed baselines across datasets with no training, threshold tuning, or in-context examples.

These results support low-cost policy adaptation through natural language prompting; their operational and conservation benefit remains to be demonstrated prospectively. Comparisons with published baselines remain conditional on differences in data, inputs and evaluation methods.

{{ summary_figure }}

## Explore and reproduce

- **[Interactive report](report.html):** methods, dataset tabs, prompt viewer, results, limitations and downloadable figures.
- **[Repository and usage guide](https://github.com/ldswaby/wildlife-trade-screen):** installation, dataset inspection, individual experiments, campaigns and custom CSVs.
- **[Recorded results](https://github.com/ldswaby/wildlife-trade-screen/tree/main/results):** all 28 selected runs and an independently checkable aggregate CSV.
- **[Dataset sources and preparation](https://github.com/ldswaby/wildlife-trade-screen/blob/main/data/README.md):** the three evaluation CSVs and their provenance.

The released files can rebuild the complete report offline without API credentials or paid inference. New live experiments require separately configured provider access or a local model deployment.
