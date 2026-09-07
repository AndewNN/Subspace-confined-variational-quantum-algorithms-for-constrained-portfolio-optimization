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
  - `PO_new_ApproxRatio.py` — SC-QAOA approximation-ratio sweeps, covering the soft-penalty (`X`), subspace-confined (`Preserving`), and training-free linear-ramp (`Ramp`, LR-QAOA) ansätze.
  - `PO_random_solution.py` — randomized admissible-portfolio reference cloud used in §08 of the paper.
  - `merge_split_results.py` — stitches the outputs of parallel `-post` slices back into one results folder.
  - `make_fig2_trainability.py`, `make_fig3_ga_quality.py` — read the cached experiment outputs and produce the paper figures.
  - `models/` — small pickled artefacts (`gaussian_copula*.pkl`) used by the figure-generation scripts.
- **`scripts/`** — three parameterized shell scripts that orchestrate the multi-configuration sweeps used to produce the paper's figures.
- **`dataset/`** — the two CSVs the published results were computed from (50-ticker universe, 2025-05-26 snapshot), with `SHA256SUMS` for verification.
- **`prepare_data.py`** — optional: rebuilds those CSVs from Yahoo Finance. Not needed to reproduce the paper.

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

### Stage 1 — Dataset (already included)

**No action needed.** The two CSV files the experiments read are committed under `dataset/`:

- `dataset/top_50_us_stocks_returns_price.csv` — per-ticker mean daily return, last close, and name (50 tickers)
- `dataset/top_50_us_stocks_data_20250526_011226_covariance.csv` — the 50 x 50 daily-return covariance matrix

These are the exact files the published results were computed from, snapshotted on 2025-05-26 over the window 2015-04-01 to 2025-04-01. They are shipped with the repository deliberately, so that reproducing the paper does not depend on a third-party API that may change or revise its history. Verify them with:

```shell
cd dataset && shasum -a 256 -c SHA256SUMS && cd ..
```

#### Optional: rebuilding the dataset from source

`prepare_data.py` re-downloads daily prices from Yahoo Finance for the same 50 tickers and rebuilds both CSVs. It is provided to document how the shipped data was constructed, and is **not** part of the reproduction path.

```shell
python prepare_data.py --out-dir dataset_rebuilt   # writes elsewhere; does not overwrite
```

> **Note.** A rebuild will not reproduce the shipped files bit-for-bit. Yahoo revises adjusted-close series over time for splits and dividends, so a fetch today yields slightly different mean returns and covariances than the 2025-05-26 snapshot. Use the committed CSVs to reproduce the paper; use `prepare_data.py` only to extend the study to a different universe or date range.

### Stage 2 — Run the experiments

All experiment commands run from the **`experiments/` directory** so the relative path `../dataset/...` resolves correctly.

```shell
cd experiments
```

#### Soft-penalty baseline (§03 of the paper)

```shell
python PO_Mixer_Benchmark.py -Q 10 -A 3 -m X  -q 1.5 -L 0.0005 -b_X 5000 -E 10 -p 5
python PO_Mixer_Benchmark.py -Q 10 -A 3 -m Preserving -B 12 -q 1.5 -L 0.0005 -b_P 10500 -E 10 -p 5
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
python PO_new_Plateau.py -Q 2 -A 2 3 4 5 6 7 8 -N 2000 -E 20 -p 5 -Z 2 3 -L 0.0005 -q 1.5 -m Preserving -B 12
```

The orchestration script `../scripts/run_plateau.sh` runs the full plateau sweep:

```shell
bash ../scripts/run_plateau.sh x_mixer 0 0.001 1      # GPU 0, λ=0.001, q=1
bash ../scripts/run_plateau.sh preserving             # Preserving-mixer plateau sweep
```

#### Subspace-confined QAOA (§04, §07, §08)

```shell
python PO_new_ApproxRatio.py -Q 2 -A 6 -m Preserving -B 24
```

`-B` = working-subspace dimension `K` (the paper reports `K ∈ {12, 24}`), `-m` selects the ansatz:

| `-m` | Ansatz |
|------|--------|
| `X` | Soft-penalty baseline — budget enters the cost Hamiltonian as `λ · H_penalty`, standard transverse-field mixer. |
| `Preserving` | Subspace-confined QAOA — the mixer confines the dynamics to `S_ε`; no `λ` to tune. |
| `Ramp` | Linear-Ramp QAOA (LR-QAOA) — no optimization; the schedule `γ_i = δγ·(i+1)/p`, `β_i = δβ·(1 − i/p)` is evaluated once as a training-free reference. |

**Hamiltonian scaling.** Adam runs at a fixed step size, so the factor the cost Hamiltonian is scaled by sets the effective gradient magnitude. Instead of hand-tuning one boost per `(λ, A)` cell, `-norm` derives it from the Ising couplings:

| `-norm` | Boost |
|---------|-------|
| `J` | `1 / max|J_ij|` |
| `Jh` | `1 / max(max|J_ij|, max|h_i|)` — recommended |
| `h` | `1 / max|h_i|` |
| `fixed` | the literal `-b_X` / `-b_P` / `-b_R` value (default) |

`-lr_s` multiplies the derived boost on top, and `-wd` sets Adam's weight decay. All reported expectation values are divided back out by the boost, so runs stay comparable across normalization modes. `--LR_init` starts the optimizer from the linear-ramp schedule (`-d_b`, `-d_g`) rather than from zeros; `--random_init` starts from a random point. `-seed N` repeats each cell with `N` independent initializations.

The orchestration script `../scripts/run_approx.sh` provides the canonical presets:

```shell
bash ../scripts/run_approx.sh preserving_K24          # main SC-QAOA config (Fig 5)
bash ../scripts/run_approx.sh x_baseline              # SP-baseline comparison (Fig 5)
bash ../scripts/run_approx.sh lr_x_sweep              # X mixer, linear-ramp init, (A × λ × p) sweep
bash ../scripts/run_approx.sh lr_preserving_sweep     # Preserving mixer, linear-ramp init, (A × p) sweep
```

Results land under `<root>/exp_L<λ>_q<q>/` as `report_<mode>_boost_<norm>.csv` plus a matching `expectation_*.npz`. The root directory name encodes the run configuration (init scheme, LR scale, weight decay, normalization mode) so concurrent sweeps do not collide; `-root` overrides it.

#### Splitting a sweep across GPUs

Give each slice its own `-post` tag and a disjoint `-e_list`, then merge:

```shell
CUDA_VISIBLE_DEVICES=0 python PO_new_ApproxRatio.py ... -post job1 -e_list 0 1 2 3 4 &
CUDA_VISIBLE_DEVICES=1 python PO_new_ApproxRatio.py ... -post job2 -e_list 5 6 7 8 9 &
wait
python merge_split_results.py -root <root> -post job1 job2 --dry_run   # preview
python merge_split_results.py -root <root> -post job1 job2
```

Rows are de-duplicated on `(Assets, Layer, Exp, Seed, Boost)` and `.npz` keys are merged, with split-run entries winning on conflict.

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

The scripts retain their original cell structure as `# %% [code cell N]` markers, so they can also be opened in VS Code's Interactive Window or Spyder for cell-by-cell exploration.

Both scripts run one optimizer per invocation, cache its trace under `./output_PO/` (Fig. 3) or `./output_PO_mixer/` (Fig. 2), then plot every cached trace they find. Reproducing the full comparison panel therefore means running each script once per optimizer index (`Nelder-Mead, COBYLA, SPSA, Adam, GradientDescent`):

```shell
for i in 0 1 2 3 4; do SCQAOA_OPTIMIZER_IDX=$i python make_fig3_ga_quality.py; done
```

Both seed `numpy` and CUDA-Q with 42, so repeated runs give identical figures.

## Reproducibility

Given a fixed dataset, every stage of the pipeline is deterministic — repeated runs on the same machine produce bit-identical CSV and `.npz` output.

| Source of randomness | How it is controlled |
|---|---|
| Asset subset, budget draw | `np.random.seed(911 + 991·e + 997·A)` — unique but reproducible per (experiment, asset count). |
| Variational parameter init | `np.random.seed(4001 + 4099·e + 4999·A + 5099·s)` — adds the seed index `s` from `-seed N`. |
| GA subspace selection | `ga_solver.GeneticAlgorithm(..., seed=919 + 991·e + 997·A)`. The C++ RNG defaults to `std::random_device` when no seed is passed; the experiment scripts always pass one. |
| Figure scripts | `np.random.seed(42)` and `cudaq.set_random_seed(42)` — the latter governs the 1e6-shot `cudaq.sample()` calls. |
| Optimizer (torch-Adam) | No torch RNG is used; initial points come from the numpy stream above. Gradients are exact statevector expectations, not sampled. |

Caveats worth knowing before comparing numbers:

- **The dataset is the limiting factor**, not the code — see the warning in Stage 1.
- **Rebuild `ga_solver` after pulling**, or the GA seed argument will be missing from your compiled extension: `cd ga_solver && pip install -e . && cd ..`.
- **Versions are unpinned** in `environment.yml`. Results depend most on the `cuda-quantum-cu12`, `torch` and `numpy` versions; see the note at the bottom of that file for capturing an exact lock.
- **Across different GPUs**, floating-point reduction order in the CUDA-Q statevector simulator can differ, so expect agreement to ~1e-10 rather than bit-identity when changing hardware.
- `ga_solver/BF_benchmark.cpp` is a standalone brute-force comparison, not built by `pip install -e .`. It includes the GCC-only `<bits/stdc++.h>`, so it needs `g++` (not `clang++`) to compile.

## Repository layout

```
.
├── README.md
├── LICENSE
├── environment.yml
├── prepare_data.py             # optional: rebuild dataset/ from Yahoo Finance
├── dataset/                    # the CSVs the paper was computed from (+ SHA256SUMS)
├── .gitignore
├── assets/
│   └── teaser.png
├── Utils/                      # Core QAOA / Markowitz utilities (CUDA-Q based)
├── ga_solver/                  # C++ / pybind11 GA preprocessing extension
├── experiments/                # Stages 2 & 3: experiment + figure-generation scripts
│   ├── PO_*.py                 #   experiment scripts
│   ├── merge_split_results.py  #   merge parallel (-post) slices back together
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
