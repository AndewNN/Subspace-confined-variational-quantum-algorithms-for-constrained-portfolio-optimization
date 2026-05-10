# Subspace-Confined Variational Quantum Algorithms for Constrained Portfolio Optimization

Naravit Namson<sup>1</sup>, Koravich Sangkaew<sup>2</sup>, Kamonluk Suksen<sup>1,4</sup>, Supanut Thanasilp<sup>3,4</sup>, Thiparat Chotibut<sup>3,4</sup>

<sup>1</sup> Department of Computer Engineering, Faculty of Engineering, Chulalongkorn University &nbsp;·&nbsp;
<sup>2</sup> SCBX Company, Limited &nbsp;·&nbsp;
<sup>3</sup> Chula Intelligent and Complex Systems Center of Excellence, Faculty of Science, Chulalongkorn University &nbsp;·&nbsp;
<sup>4</sup> Siam Quantum Square (SQ²), Faculty of Science, Chulalongkorn University

This is the official implementation accompanying our paper *"Subspace-Confined Variational Quantum Algorithms for Constrained Portfolio Optimization."* The codebase reproduces the experiments and figures in the paper, including the soft-penalty (SP) baseline, the subspace-confined (SC) QAOA ansatz, and the genetic-algorithm (GA) preprocessing step.

![Teaser](assets/teaser.png)

## Overview

We replace the standard QAOA penalty term `λ · H_penalty` with a *preserving mixer* `H_M` that confines the variational dynamics to a low-budget-violation subspace `S_ε` obtained by classical preprocessing. The result is **feasibility by construction** — there is no `λ` to tune, and trained states never leave the admissible budget band.

The repository contains:

- **`Utils/`** — core QAOA / Markowitz utilities (`qaoaCUDAQ.py`), graph construction, and helper functions built on top of [CUDA-Q](https://github.com/NVIDIA/cuda-quantum).
- **`ga_solver/`** — a C++ / pybind11 extension implementing the genetic algorithm used to identify the low-violation working subspace `S_ε`. A brute-force baseline (`BF_benchmark.cpp`) is included for comparison.
- **`experiments/`** — runnable Python scripts for the three experiment families plus figure-generation scripts:
  - `PO_Mixer_Benchmark.py` — SP-baseline benchmarks across mixers and depths.
  - `PO_X_Plateau.py`, `PO_new_Plateau.py` — barren-plateau diagnostics (variance scaling vs. qubit count).
  - `PO_new_ApproxRatio.py` — SC-QAOA approximation-ratio sweeps.
  - `PO_random_solution.py` — randomized admissible-portfolio reference cloud used in §08 of the paper.
  - `make_fig2_trainability.py`, `make_fig3_ga_quality.py` — read the cached experiment outputs and produce the paper figures.
  - `models/` — small pickled artefacts (`gaussian_copula*.pkl`) used by the figure-generation scripts.
- **`scripts/`** — three parameterized shell scripts that orchestrate the multi-configuration sweeps used to produce the paper's figures.
- **`prepare_data.py`** — one-shot script that fetches the daily-price universe from Yahoo Finance and produces the two CSVs the experiments expect under `dataset/`.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{namson2026scqaoa,
  title  = {Subspace-Confined Variational Quantum Algorithms for Constrained Portfolio Optimization},
  author = {Namson, Naravit and Sangkaew, Koravich and Suksen, Kamonluk and Thanasilp, Supanut and Chotibut, Thiparat},
  year   = {2026},
}
```

## Cloning the repository

```shell
# SSH
git clone git@github.com:AndewNN/Subspace-confined-variational-quantum-algorithms-for-constrained-portfolio-optimization.git

# HTTPS
git clone https://github.com/AndewNN/Subspace-confined-variational-quantum-algorithms-for-constrained-portfolio-optimization.git
```

## Hardware requirements

- An **NVIDIA GPU** with CUDA support is strongly recommended. All experiments in the paper were run on a single NVIDIA RTX 4080 (16 GB).
- The CUDA-Q statevector simulator scales to roughly `n ≤ 30` qubits on a 16 GB GPU.
- Brute-force enumeration in `ga_solver/BF_benchmark.cpp` is CPU-only and grows exponentially in `n` — typical workstation reaches `n ≈ 24`.

## Software requirements

- Linux (tested on Ubuntu 22.04). macOS / Windows are untested for the CUDA-Q backend.
- C++17-capable compiler (`g++ ≥ 9` or `clang++ ≥ 10`).
- CUDA toolkit compatible with `cuda-quantum-cu12` (CUDA 12.x).
- Conda or Mamba (recommended for reproducibility).

## Setup

### 1. Create the conda environment

```shell
conda env create -f environment.yml
conda activate scqaoa
```

### 2. Build the GA solver

```shell
cd ga_solver
pip install -e .
cd ..
```

This compiles `genetic_solver.cpp` and exposes it as the `ga_solver` Python module used by `experiments/PO_new_ApproxRatio.py`.

### 3. Sanity check

```shell
python -c "import cudaq, ga_solver; from Utils.qaoaCUDAQ import po_normalize; print('OK')"
```

## Workflow

The pipeline runs in three ordered stages: **prepare data → run experiments → open notebooks**.

### Stage 1 — Prepare the dataset

`prepare_data.py` downloads daily prices for a configurable list of US equities from Yahoo Finance (default range 2015-04-01 to 2025-04-01) and produces the two CSV files the experiment scripts expect under `./dataset/`.

```shell
# from the repo root
python prepare_data.py
```

This writes:
- `dataset/top_50_us_stocks_returns_price.csv` — per-ticker mean return + last close + name
- `dataset/top_50_us_stocks_data_20250526_011226_covariance.csv` — daily-return covariance matrix

> **Note.** The `TICKERS` list at the top of `prepare_data.py` is a default placeholder of large-cap US equities. **Edit it to match the exact 50-ticker universe used in the paper** before running this step if you intend to reproduce the published numbers.

### Stage 2 — Run the experiments

All experiment commands run from the **`experiments/` directory** so the relative path `../dataset/...` resolves correctly.

```shell
cd experiments
```

#### Soft-penalty baseline (§03 of the paper)

```shell
python PO_Mixer_Benchmark.py -Q 10 -A 3 -B 3 6 12
```

`-Q` = qubit count, `-A` = asset configuration, `-B` = list of subspace sizes to sweep. The orchestration script `../scripts/run_mixer_benchmark.sh` runs the exact sweeps used to produce Fig. 1:

```shell
bash ../scripts/run_mixer_benchmark.sh full           # full sweep
bash ../scripts/run_mixer_benchmark.sh mini           # quick smoke test
bash ../scripts/run_mixer_benchmark.sh split 0        # one slice on GPU 0
bash ../scripts/run_mixer_benchmark.sh merge 4 3 3    # 4-way Preserving + merger
```

#### Barren-plateau diagnostics (§03b)

```shell
python PO_X_Plateau.py -Q 10 -A 3
python PO_new_Plateau.py -Q 10 -A 3
```

The orchestration script `../scripts/run_plateau.sh` runs the full plateau sweep:

```shell
bash ../scripts/run_plateau.sh x_mixer 0 0.001 1      # GPU 0, λ=0.001, q=1
bash ../scripts/run_plateau.sh preserving             # Preserving-mixer plateau sweep
```

#### Subspace-confined QAOA (§04, §07, §08)

```shell
python PO_new_ApproxRatio.py -Q 10 -A 3 -K 12
```

`-K` = working-subspace dimension (the paper reports `K ∈ {12, 24}`). The orchestration script `../scripts/run_approx.sh` provides the canonical presets:

```shell
bash ../scripts/run_approx.sh preserving_K24          # main SC-QAOA config (Fig 5)
bash ../scripts/run_approx.sh x_baseline              # SP-baseline comparison (Fig 5)
```

#### Random admissible reference cloud (§08)

```shell
python PO_random_solution.py -Q 10 -A 3
```

#### Merging multi-run CSV outputs

```shell
python PO_Mixer_Benchmark_Merger.py
```

Combines per-configuration CSV outputs into a single results table consumed by the figure-generation scripts.

The experiment scripts write their results under `./experiments_*/` (relative to `experiments/`). Those directories are git-ignored — they are produced and consumed locally.

### Stage 3 — Reproduce the figures

Run the figure-generation scripts from the **`experiments/` directory** so the relative paths `./models/`, `./experiments_*/`, and `../dataset/...` resolve correctly:

```shell
cd experiments  # if not already there
python make_fig2_trainability.py
python make_fig3_ga_quality.py
```

| Figure | Script |
|--------|--------|
| Fig. 2 — Trainability | `make_fig2_trainability.py` |
| Fig. 3 — GA timing & quality | `make_fig3_ga_quality.py` |

The scripts retain their original cell structure as `# %% [code cell N]` markers, so they can also be opened in VS Code's Interactive Window or Spyder for cell-by-cell exploration. They expect outputs from Stage 2 to already exist on disk; they do not run any experiments themselves.

## Repository layout

```
.
├── README.md
├── LICENSE
├── environment.yml
├── prepare_data.py             # Stage 1: fetch dataset from Yahoo Finance
├── .gitignore
├── assets/
│   └── teaser.png
├── Utils/                      # Core QAOA / Markowitz utilities (CUDA-Q based)
├── ga_solver/                  # C++ / pybind11 GA preprocessing extension
├── experiments/                # Stages 2 & 3: experiment + figure-generation scripts
│   ├── PO_*.py                 #   experiment scripts
│   ├── make_fig2_trainability.py    # Fig. 2 generator
│   ├── make_fig3_ga_quality.py      # Fig. 3 generator
│   └── models/                 #   small pickled artefacts
└── scripts/                    # Shell orchestration for parameter sweeps
    ├── run_mixer_benchmark.sh  #   §03 sweeps (PO_Mixer_Benchmark)
    ├── run_plateau.sh          #   §03b plateau diagnostics
    └── run_approx.sh           #   §07/§08 SC-QAOA approximation-ratio sweeps
```

## Acknowledgements

- Special thanks to [BallBoii](https://github.com/BallBoii) for the original `Utils/` source on top of which `qaoaCUDAQ.py` was developed.
- This work was supported by the Graduate School Scholarship of Chulalongkorn University (the 72ⁿᵈ HM the King anniversary), the 111ᵗʰ Anniversary Engineering Research Catalyst Fund of the Faculty of Engineering, Chulalongkorn University, and SCB Asset Management for industry insights on the constrained portfolio formulation.

## License

MIT — see [LICENSE](LICENSE).
