# Decision models for zero-shot wildlife advert screening

<p class="lead">Evaluating off-the-shelf decision models for low-cost, policy-based screening of wildlife adverts.</p>

**[Read the interactive report](report.html)** · **[Explore the code](https://github.com/ldswaby/wildlife-trade-screen)** · **[Reproduce the study](https://github.com/ldswaby/wildlife-trade-screen#1-install-and-get-ready)**

## Research question

Can off-the-shelf decision models deliver competitive wildlife-advert screening through policy prompt changes alone, without task-specific training?

## The study

The study compares three decision models and a contemporary chat model on historical Shark Products, Animal Products and Live Birds datasets. The report brings together 28 experimental configurations, recorded predictions, descriptive ROC curves, API costs and request timings.

Decision models delivered competitive screening performance at low hosted API costs. The locally deployed, quantised `d3` model was particularly consistent under zero-shot conditions. Comparisons with published baselines remain conditional on differences in data, inputs and evaluation methods.

{{ summary_figure }}

## Explore and reproduce

- **[Interactive report](report.html):** methods, dataset tabs, prompt viewer, results, limitations and downloadable figures.
- **[Repository and usage guide](https://github.com/ldswaby/wildlife-trade-screen):** installation, dataset inspection, individual experiments, campaigns and custom CSVs.
- **[Recorded results](https://github.com/ldswaby/wildlife-trade-screen/tree/main/results):** all 28 selected runs and an independently checkable aggregate CSV.
- **[Dataset sources and preparation](https://github.com/ldswaby/wildlife-trade-screen/blob/main/data/README.md):** the three evaluation CSVs and their provenance.

The released files can rebuild the complete report offline without API credentials or paid inference. New live experiments require separately configured provider access or a local model deployment.

## Interpretation

Relevance means matching the stated monitoring policy. It does not establish illegality, authenticity or conservation harm. These historical results support low-cost policy adaptation through prompts; their operational and conservation benefit remains to be demonstrated prospectively.
