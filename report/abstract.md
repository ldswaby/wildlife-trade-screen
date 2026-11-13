<h1>Wildlife-trade screening</h1>
<p class="lead">Evaluating off-the-shelf decision models for wildlife-trade advert screening.</p>
<div class="notice">The interactive report presents recorded results from historical datasets; prospective operational benefits remain to be evaluated.</div>
<section aria-labelledby="research-question">
  <h2 id="research-question">Research question</h2>
  <p>Can off-the-shelf decision models find wildlife-product adverts with useful TPR, manageable human review and affordable inference?</p>
  <p>The project began with shark-product adverts and extends to animal products, small leather products and live birds. It compares measured model results with published evidence where a suitable baseline exists.</p>
  <p>Here, <strong>relevant means matching the monitoring policy</strong>. It does not establish illegality, authenticity or conservation harm.</p>
</section>
<section aria-labelledby="interactive-report">
  <h2 id="interactive-report">Interactive report</h2>
  <p>The <a href="report.html">saved interactive report</a> brings together dataset tabs, descriptive curves, resource figures and results tables.</p>
  <ul>
    <li><strong>TPR and false-positive rate.</strong> Compare model configurations and fixed-cutoff markers, keeping published comparisons distinct from measured results.</li>
    <li><strong>Cost and request time.</strong> Inspect recorded API charges and average saved request latency. Request latency differs from total run time; unknown costs remain unknown.</li>
    <li><strong>Coverage and results.</strong> Read precision, TPR, F1 and prediction coverage together. Download PNG or SVG figures for closer inspection.</li>
  </ul>
</section>
<section aria-labelledby="methods">
  <h2 id="methods">Methods</h2>
  <p>Compare few-shot and zero-shot screeners using fixed policies and recorded input conditions. Title-only is the default; title + desc includes descriptions.</p>
  <p>Probability arms use a fixed 0.5 cutoff in the comparison table. Curves describe other cutoffs; the 95% TPR reference is not a validated operating threshold. Binary models have a single decision point.</p>
  <p>Explicit inference failures are sent for human review, with metrics using the full evaluation denominator. Prediction coverage is shown; missing or corrupt snapshots cannot be evaluated.</p>
</section>
<section aria-labelledby="interpretation">
  <h2 id="interpretation">Interpretation and scope</h2>
  <p>Published aggregates are separate evidence, not reproduced runs. Derived false-positive rates depend on compatible metric definitions and dataset prevalence. Similar F1 does not prove identical methods.</p>
  <p>This implementation evaluates saved model decisions. Training classifiers, pool pseudo-labelling, cascades, threshold selection and a review interface remain future work.</p>
</section>
<section aria-labelledby="reproducibility">
  <h2 id="reproducibility">Code and reproducibility</h2>
  <p>The repository documents setup, fixed prompts, provider adapters and offline checks. An invented-data example exercises the workflow without private data or API keys.</p>
  <ul class="reference-links">
    <li><a href="https://github.com/ldswaby/wildlife-trade-screen">Browse the repository ↗</a></li>
    <li><a href="https://github.com/ldswaby/wildlife-trade-screen#1-install-and-get-ready">Setup and offline example ↗</a></li>
    <li><a href="https://github.com/ldswaby/wildlife-trade-screen/blob/main/docs/SOURCES.md">Sources and evidence ↗</a></li>
  </ul>
  <p class="subtle">Source data and raw model responses are excluded from the website bundle.</p>
</section>
