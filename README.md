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

- **`Utils/`** — core QAOA / Markowitz utilities (`qaoaCUDAQ.py`), graph construction, and solver helpers built on top of [CUDA-Q](https://github.com/NVIDIA/cuda-quantum).
- **`ga_solver/`** — a C++ / pybind11 extension implementing the genetic algorithm used to identify the low-violation working subspace `S_ε`. Brute-force baseline (`BF_benchmark.cpp`) is included for comparison.
- **`experiments/`** — runnable Python scripts for the three experiment families:
  - `PO_Mixer_Benchmark.py` — SP-baseline benchmarks across mixers and depths.
  - `PO_X_Plateau.py`, `PO_new_Plateau.py` — barren-plateau diagnostics (variance scaling vs. qubit count).
  - `PO_new_ApproxRatio.py` — SC-QAOA approximation-ratio sweeps.
  - `PO_random_solution.py` — randomized admissible-portfolio reference cloud used in §08 of the paper.
  - `adam.py` — Adam optimizer wrapper with cosine-annealing schedule used across experiments.
- **`notebooks/`** — Jupyter notebooks for figure generation and inline exploration. Outputs are stripped; rerun to regenerate.
- **`scripts/`** — shell scripts that orchestrate the multi-configuration sweeps used to produce the paper's figures.

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

## Running the experiments

All commands below are run from the repository root.

### Soft-penalty baseline (§03 of the paper)

```shell
python experiments/PO_Mixer_Benchmark.py -Q 10 -A 3 -B 3 6 12
```

`-Q` is the qubit count, `-A` selects the asset configuration, `-B` is the list of subspace sizes to sweep. See `scripts/run_queue.sh` for the exact sweeps used to produce Fig. 1.

### Barren-plateau diagnostics (§03b)

```shell
python experiments/PO_X_Plateau.py -Q 10 -A 3
python experiments/PO_new_Plateau.py -Q 10 -A 3
```

The corresponding orchestration scripts are `scripts/run_plateau_*.sh`.

### Subspace-confined QAOA (§04, §07, §08)

```shell
python experiments/PO_new_ApproxRatio.py -Q 10 -A 3 -K 12
```

`-K` is the working-subspace dimension. The paper reports `K ∈ {12, 24}`. See `scripts/run_approx_*.sh` for the full configuration matrix.

### Random admissible reference cloud (§08)

```shell
python experiments/PO_random_solution.py -Q 10 -A 3
```

### Merging multi-run CSV outputs

```shell
python experiments/PO_Mixer_Benchmark_Merger.py
```

This combines per-configuration CSV outputs into a single results table consumed by the figure-generation notebooks.

## Reproducing the figures

The notebooks in `notebooks/` regenerate the paper's figures from cached CSV results.

| Figure | Notebook |
|--------|----------|
| Fig. 1 — SP penalty diagnostics | `notebooks/PO_Mixer_Benchmark.ipynb` |
| Fig. 2 — Trainability | `notebooks/PO_Preserving_Mixer.ipynb` |
| Fig. 3 — GA timing & quality  | `notebooks/PO_QAOA.ipynb` |

Launch Jupyter from the repo root so that the `Utils.*` and `ga_solver` imports resolve:

```shell
jupyter notebook
```

## Repository layout

```
.
├── README.md
├── LICENSE
├── environment.yml
├── .gitignore
├── assets/
│   └── teaser.png
├── Utils/                  # Core QAOA / Markowitz utilities (CUDA-Q based)
├── ga_solver/              # C++ / pybind11 GA preprocessing extension
├── experiments/            # Runnable Python scripts
├── notebooks/              # Figure-generation notebooks
└── scripts/                # Shell orchestration for parameter sweeps
```

## Data

Asset prices and returns are loaded from Yahoo Finance via `yfinance`. The paper uses 50 US equities with prices in the range $108–$216 USD over 2015/04 – 2025/04. The exact ticker list is reproduced inside the experiment scripts; rerunning the scripts re-downloads the prices on demand.

## Acknowledgements

- Special thanks to [BallBoii](https://github.com/BallBoii) for the original `Utils/` source on top of which `qaoaCUDAQ.py` was developed.
- This work was supported by the Graduate School Scholarship of Chulalongkorn University (the 72ⁿᵈ HM the King anniversary), the 111ᵗʰ Anniversary Engineering Research Catalyst Fund of the Faculty of Engineering, Chulalongkorn University, and SCB Asset Management for industry insights on the constrained portfolio formulation.

## License

MIT — see [LICENSE](LICENSE).
