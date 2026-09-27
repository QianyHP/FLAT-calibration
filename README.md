# FLAT: Fingerprint-guided Learnable Acquisition for Traffic Calibration

![FLAT framework](assets/architecture.png)

FLAT is a simulation-efficient calibration framework for traffic models. It
learns a behavioral fingerprint from trajectory data, uses it to define a
calibration objective, and combines a surrogate model with lower-confidence
bound acquisition under a fixed SUMO evaluation budget.

This repository contains the experiment code, public input data, cached
results, and publication-ready figures for the accompanying study.

## Repository layout

- `code/experiments/` — experiment runners and analysis scripts
- `data/` — scene configurations and public input data
- `outputs/figures/` — PDF figures used in the paper
- `outputs/results/` — cached data supporting the paper tables and figures
- `assets/` — project figures used by this README

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

The experiment runners under `code/experiments/` require a working SUMO/TraCI
installation. Use their command-line help for the available experiment and
scene options.

## Data

The study uses trajectory data from two public sources:

- [SinD](https://github.com/SOTIF-AVLab/SinD) — signalized-intersection scenes
- [Ubiquitous Traffic Eye](https://github.com/Ruyi-Feng/Ubiquitous-Traffic-Eye) — urban-expressway scenes

Please follow the original licenses and attribution requirements for external
datasets.

## License

Code is released under the MIT License; see [LICENSE](LICENSE).
