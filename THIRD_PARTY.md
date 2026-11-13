# Attribution and reuse terms

## Original project code

The MIT licence in `LICENSE` applies to original project code, tests, scripts and website implementation. It does not grant additional rights to third-party datasets, quoted source prompts, papers, model weights, or report prose and figures. Report prose and figures remain copyright Luke Swaby unless otherwise attributed; no separate reuse licence is granted here.

## Evaluation datasets

| Release files | Source | Published terms and attribution |
| --- | --- | --- |
| `data/validation_sharks.csv`, `data/validation_animals.csv` | [Barbosa et al., LTS](https://github.com/VIDA-NYU/LTS/tree/main/data_use_cases), associated with [Barbosa et al. (2025)](https://doi.org/10.1145/3725256) | Supplied validation CSVs are preserved unchanged. No licence file or GitHub-detected licence was present in the upstream LTS repository when checked on 13 November 2026. Original rights remain with their respective holders; this repository does not assign an MIT licence to these data or claim additional redistribution permissions. Consult the source authors regarding reuse terms. |
| `data/birds_processed.csv` | [Stringham et al., release version 2](https://doi.org/10.6084/m9.figshare.14032742.v2), associated with [Stringham et al. (2021)](https://doi.org/10.1371/journal.pone.0254007) | The Figshare release declares [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The listing IDs and combined text are retained; exclusion labels are combined into the evaluation target and unresolved rows are omitted, as described in the data README and preparation script. |

See [data/README.md](data/README.md) for the source and processing of each file. The inclusion of advert text does not establish illegality, authenticity or conservation harm.

## Source prompts, models and dependencies

The LTS policy wording is attributed in the report and source notes to Barbosa et al. (2025), Figure 3. Published baseline values remain attributed to the cited studies. Model identifiers and configuration files do not include model weights or transfer model-provider rights. Python and JavaScript dependencies retain their own licences; the website embeds Plotly supplied by the installed Plotly package, including its licence notice. Consult each upstream project or provider for its terms.

## Clean release history

The public repository begins with a reviewed release snapshot. Earlier development history, private notes, caches and raw provider responses are not published. Historical commit identifiers in portable result manifests are retained as provenance rather than rewritten; configurations, prompts, dataset hashes and scores needed for the report are included in this release.
