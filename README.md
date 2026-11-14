# Wildlife-trade screening

[Project website](https://ldswaby.github.io/wildlife-trade-screen/) · [Interactive report](https://ldswaby.github.io/wildlife-trade-screen/report.html)

Inspect labelled wildlife adverts, evaluate a model on them, and compare the results. The repository includes the three evaluation datasets and saved predictions for the 28 configurations in the study.

Follow the steps below to run new experiments. To view the existing study immediately after installation, run `python site/build.py build` and open `build/site/report.html`.

## 1. Install and get ready

Use Python 3.11 or newer on macOS or Linux. Run these commands once:

```bash
git clone https://github.com/ldswaby/wildlife-trade-screen.git
cd wildlife-trade-screen
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Run subsequent commands from the `wildlife-trade-screen` directory. In a new terminal, return to that directory and run `source .venv/bin/activate`.

### Try it offline first

This example scores 20 invented adverts with two fake models. It exercises the complete workflow without API credentials or spending.

```bash
wildlife-screen run --data tests/fixtures/synthetic20.csv \
  --model configs/models/fake.yaml --model configs/models/fake_binary.yaml \
  --output outputs/example --cache outputs/example-cache
```

Open `outputs/example/few-shot/table1.csv` and `outputs/example/few-shot/figure1.svg`. Repeat the command to reuse the cached predictions. Add `--zero-shot` to run the other prompt condition and write its files under `outputs/example/zero-shot/`.

### Before live scoring

For hosted models, set the credential required by the selected [model configuration](configs/models):

```bash
export OPENAI_API_KEY='YOUR_OPENAI_KEY'
export OPENROUTER_API_KEY='YOUR_OPENROUTER_KEY'
```

Replace the quoted placeholders with your keys. OpenAI models use `OPENAI_API_KEY`; Microsoft via OpenRouter uses `OPENROUTER_API_KEY`. Verify your provider access, billing and current pricing. Local d3 requires the [model and server setup](docs/D3_LOCAL_MAC.md).

Every live command below includes a USD spending ceiling. **The ceiling includes previous spending in the shared cache** and is shared across experiments. The example amounts of 5 and 20 USD must be replaced with the cumulative limits you authorize after reading the dry-run estimate.

## 2. Inspect a dataset

Inspection checks label meaning, missing fields and duplicate titles before scoring. Run the command for your dataset:

```bash
wildlife-screen inspect --task sharks --output outputs/inspect-sharks.json
```

```bash
wildlife-screen inspect --task animals --positive-label 1 --output outputs/inspect-animals.json
```

```bash
wildlife-screen inspect --task birds --output outputs/inspect-birds.json
```

The terminal prints counts and any label-mapping problem. Open the named JSON file for the detailed inspection. The [data README](data/README.md) explains the source and processing of each CSV.

## 3. Run one experiment

An experiment evaluates one model on one dataset with one input and prompt condition. This example uses **Sharks, GPT-6 Luna Decisions, title only, zero-shot**.

First estimate the pending requests and cost without contacting the model:

```bash
wildlife-screen run --task sharks --model configs/models/gpt6_luna_decisions.yaml \
  --zero-shot --dry-run --output outputs/single-sharks-preflight
```

Read the printed estimate and manifest path. Once access and spending are authorized, run the same condition:

```bash
wildlife-screen run --task sharks --model configs/models/gpt6_luna_decisions.yaml \
  --zero-shot --spend-ceiling 5 --output outputs/single-sharks
```

### Choose the dataset, model and condition

| Choice | What to use |
| --- | --- |
| Sharks | `--task sharks`; defaults to `data/validation_sharks.csv` |
| Animals | `--task animals`; defaults to `data/validation_animals.csv` |
| Birds | `--task birds --zero-shot --combined-text`; defaults to `data/birds_processed.csv` |
| GPT-6 Luna Decisions | `--model configs/models/gpt6_luna_decisions.yaml` |
| GPT-6 Luna Chat | `--model configs/models/gpt6_luna_chat_anchor.yaml` |
| Microsoft-Decision-1 | `--model configs/models/microsoft_decision1_openrouter.yaml` |
| Local d3 Q4_K_M | `--model configs/models/d3_gguf_q4.yaml` |
| Few-shot prompt | Omit `--zero-shot`; the fixed prompt includes demonstrations |
| Zero-shot prompt | Add `--zero-shot`; the fixed prompt has no demonstrations |
| Title only | Default input |
| Title + description | Add `--include-description` for separate columns |
| Combined text | Add `--combined-text` for the `title_and_desc` column |

For example, a Birds dry run with the same model is:

```bash
wildlife-screen run --task birds --model configs/models/gpt6_luna_decisions.yaml \
  --zero-shot --combined-text --dry-run --output outputs/single-birds-preflight
```

### Find the outputs

The shark example writes its results under `outputs/single-sharks/zero-shot/`:

| File | What it contains |
| --- | --- |
| `table1.csv` and `table1.md` | Performance and resource summary |
| `figure1.png` and `figure1.svg` | ROC comparison figure |
| `MODEL_FOLDER/manifest-RUN_ID.json` | The dataset, model, prompt, status and usage that identify this particular run |
| `MODEL_FOLDER/scores-RUN_ID.json` | Predictions and recorded failures in the original advert order |
| `MODEL_FOLDER/metrics-RUN_ID.json` | Calculated metrics for that model |

`MODEL_FOLDER` and `RUN_ID` are generated names. The terminal prints the exact file paths. Animals and Birds include the task name inside an explicit output directory, for example `outputs/single-birds-preflight/birds/zero-shot/`. Dry runs write manifests and estimates.

## 4. Run a complete campaign

A campaign runs all four models and every declared condition for one dataset:

| Dataset | Input conditions | Prompt conditions | Configurations |
| --- | --- | --- | ---: |
| Sharks | Title only and title + description | Few-shot and zero-shot | 16 |
| Animals | Title only | Few-shot and zero-shot | 8 |
| Birds | Combined title and description | Zero-shot | 4 |

The four models are GPT-6 Luna Chat, GPT-6 Luna Decisions, Microsoft-Decision-1 and local d3 Q4_K_M.

### Preflight the campaigns

Run each command separately and read its printed audit and estimate before continuing:

```bash
./run_full_factorial.sh --task sharks --prompt-version v0 --preflight-only
```

```bash
./run_full_factorial.sh --task animals --prompt-version v0 --preflight-only
```

```bash
./run_full_factorial.sh --task birds --prompt-version v0 --preflight-only \
  --chat-concurrency 16 --luna-concurrency 16 --microsoft-concurrency 16
```

Preflight writes `audit.json`, `prompt-audit.txt` and dry-run manifests under `outputs/checks/full-factorial/`. The Birds command reproduces the original 16 concurrent requests per hosted model, with one for local d3. Other campaigns use one worker per model.

### Run the campaigns

Start the local d3 server with one slot and at least 8,192 context tokens. All four models must be available. The launcher also requires your code and configuration changes to be committed before live scoring.

After reviewing the estimates, run each campaign with your authorized cumulative ceiling:

```bash
./run_full_factorial.sh --task sharks --prompt-version v0 --spend-ceiling 20
```

```bash
./run_full_factorial.sh --task animals --prompt-version v0 --spend-ceiling 20
```

```bash
./run_full_factorial.sh --task birds --prompt-version v0 --spend-ceiling 20 \
  --chat-concurrency 16 --luna-concurrency 16 --microsoft-concurrency 16
```

Save the printed **Campaign directory** for each dataset. Inside it, `manifest-index.txt` lists each configuration, coverage, failures, cached results and new calls. A finished campaign also produces `selection-candidate.yaml`, which you use in the next step. Campaigns write under `outputs/comparison/reviewed/`.

## 5. Choose results and build the report

A **pin** is the path to one run's manifest. For example, the shark d3 title-only, few-shot pin identifies that single configuration. The report reads its manifest and paired score file to calculate the results.

A campaign's `selection-candidate.yaml` collects these pins for you: 16 for Sharks, eight for Animals or four for Birds. The active report uses the pins in `report/selection.yaml`. The checkout already selects the original 28 study runs.

### Use your new campaigns

Finish all live campaigns before changing the tracked report selection. The importer rejects a second snapshot for an already selected configuration, so start your rerun selection from the new Sharks candidate.

**1. Save the current selection.** Create `outputs/` if needed. Choose another backup filename if this one already exists.

```bash
mkdir -p outputs
cp report/selection.yaml outputs/selection-before-rerun.yaml
```

**2. Select the new Sharks campaign.** Replace `SHARKS_CAMPAIGN_DIRECTORY` below with the directory printed by your completed shark campaign.

```bash
cp SHARKS_CAMPAIGN_DIRECTORY/selection-candidate.yaml report/selection.yaml
```

**3. Add Animals and Birds.** Replace each uppercase directory placeholder with the corresponding completed campaign directory. Run each import and check that it succeeds before continuing.

```bash
python site/build.py import-candidate ANIMALS_CAMPAIGN_DIRECTORY/selection-candidate.yaml
```

```bash
python site/build.py import-candidate BIRDS_CAMPAIGN_DIRECTORY/selection-candidate.yaml
```

The selection now contains 16 + 8 + 4 pins. Keep each campaign's manifest and score files at their recorded paths.

**4. Compile the report.**

```bash
python site/build.py build
```

Open `build/site/report.html` for the tables, figures and narrative. Edit the Markdown in `report/` and repeat the build to update the writing. Tables and charts regenerate from the selected runs; review the prose if fresh results differ.

To export the selected numerical results as one CSV:

```bash
python scripts/export_aggregate_results.py --output outputs/rerun-aggregates.csv
```

See the [results dictionary](results/README.md) for column definitions. To rebuild the original study, keep the shipped selection and run the same build command.

### Website title banners

The Overview and Report use the same decorative, AI-generated illustration behind
their existing titles. The Overview subtitle also appears in its banner. The
tracked asset is `site/wildlife-screening-banner.webp`; `site/styles.css` controls
the centred crop, 70% charcoal overlay, lower-edge fade and responsive heights
(280px desktop, 210px mobile). Original colours are retained in the image itself.
The original PNG is kept locally under `outputs/outreach/`, outside Git.
The independently scrollable page index starts over the banner at the top right
on desktop and stays visible while reading. It has no enclosing box; lighter links
and a subtle text shadow preserve readability over the darkened banner. The Report's
opening Abstract heading omits its usual 48px top margin to bring it closer to the banner.
On mobile, the collapsible index sits above the banner and its expanded list scrolls.

After editing the image or styles, run `python site/build.py build` and open
`build/site/index.html` and `build/site/report.html` to review both pages. The build
copies the image through its explicit asset allowlist into `build/site/`. It uses
saved evidence offline and does not run inference. Rebuilding is local; publication
remains a separate step.

## 6. Use your own data

Choose a task whose monitoring policy matches your labels. Save a CSV such as the following under `outputs/my-adverts.csv`:

```csv
advert_id,title,label
example-1,Invented modern shark jaw for sale,1
example-2,Invented ceramic shark ornament,0
```

Use a unique `advert_id` or `id`, the selected text column(s), and a binary `label`. Set `--positive-label 1` when `1` means relevant, or `--positive-label 0` for the opposite mapping. Without an ID column, IDs depend on the original row order.

Inspect your file:

```bash
wildlife-screen inspect --data outputs/my-adverts.csv --positive-label 1 \
  --output outputs/my-inspection.json
```

Then estimate an individual experiment:

```bash
wildlife-screen run --data outputs/my-adverts.csv --positive-label 1 \
  --model configs/models/gpt6_luna_decisions.yaml --zero-shot --dry-run \
  --output outputs/my-preflight
```

To score it, follow step 3 using the same data, label and model options. For an offline trial, use `configs/models/fake.yaml` and omit `--dry-run`. Title is the default model input; descriptions require the explicit input options in step 3. Gold labels and seller metadata stay out of scorer requests.

## Optional workflows and troubleshooting

- **Interrupted campaign:** rerun its command with `--resume CAMPAIGN_DIRECTORY`, replacing the placeholder with the original directory. Use the original code revision, settings, concurrency and cumulative ceiling.
- **Cached results:** matching completed requests, including recorded failures, are reused. Keep the cache and spending ledger intact. Reconcile uncertain calls before retrying.
- **Failures:** inspect the manifest index and score files. Recorded inference failures are sent for review in the metrics. Individual scoring returns exit status 2 for finished runs with failures, or 1 for a fatal error.
- **Leather:** supply the original `data/leather_validation.csv`, then use the campaign command with `--task leather`. This adds eight configurations and is separate from the three-dataset report.
- **Shark pool:** supply the pool with `wildlife-screen inspect --pool /path/to/shark_trophy.csv` to include its counts. Replace `/path/to/shark_trophy.csv` with your source path.
- **Prepare raw Birds data:** use the [preparation script](src/wildlife_screening/prepare_birds.py). `python -m wildlife_screening.prepare_birds --help` lists source and output options. The processed evaluation CSV is already included.
- **More options:** use `wildlife-screen run --help`, `wildlife-screen inspect --help` or `./run_full_factorial.sh --help`.

Run `pytest -q` for the offline test suite. The [experimental assumptions](ASSUMPTIONS.md) and [source notes](docs/SOURCES.md) explain interpretation. Measured tables use a fixed 0.5 cutoff; curves are descriptive.

## Licence and attribution

Original project code is available under the [MIT licence](LICENSE). Datasets, source prompts, papers, model weights and report prose have separate rights; see [third-party attribution and reuse terms](THIRD_PARTY.md).
