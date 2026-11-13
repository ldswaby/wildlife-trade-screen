# Evaluation datasets

| File | Source | Processing |
| --- | --- | --- |
| `validation_sharks.csv` | [Barbosa et al., LTS](https://github.com/VIDA-NYU/LTS/tree/main/data_use_cases) | Used the supplied 499-row validation CSV unchanged. |
| `validation_animals.csv` | [Barbosa et al., LTS](https://github.com/VIDA-NYU/LTS/tree/main/data_use_cases) | Used the supplied 499-row validation CSV unchanged. |
| `birds_processed.csv` | [Stringham et al., release v2](https://doi.org/10.6084/m9.figshare.14032742.v2) | Preserved the listing ID and combined advert text. An advert was labeled irrelevant if any of the three exclusion labels was `1`, and relevant if all three were `0`. Of 16,509 source rows, 102 had unresolved labels, leaving 16,407 for evaluation. |

See the [Birds preparation script](../src/wildlife_screening/prepare_birds.py) for the exact transformation.

Dataset attribution and reuse terms are recorded in [THIRD_PARTY.md](../THIRD_PARTY.md); the code licence does not relicense these datasets.
