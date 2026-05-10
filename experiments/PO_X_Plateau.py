"""Barren-plateau diagnostics for the X-mixer (soft-penalty) baseline.

For each (qubit count, asset count) configuration, samples N random parameter
points and evaluates ⟨H⟩, accumulating the running sum and sum-of-squares
needed to compute the variance scaling that diagnoses barren plateaus
(see §03b of the paper).
"""
from __future__ import annotations

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

import cudaq

import _paths  # noqa: F401  (puts project root on sys.path for ``from Utils.…``)
from Utils.qaoaCUDAQ import (
    find_budget,
    kernel_qaoa_X,
    po_normalize,
    process_ansatz_values,
    qubo_to_ising,
    ret_cov_to_QUBO,
    write_df,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPORT_COL = ["N", "Sum_1", "Sum_2"]

# How many (qubit, asset) loop iterations to run by default.
DEFAULT_LOOP_COUNT = 100

# Default number of random parameter points sampled per (qubit, asset) cell.
DEFAULT_NUM_SAMPLES = 2000

# Pull this many candidate samples from the GaussianCopula model before
# filtering by the [min_P, max_P] price band, to ensure enough survive.
OVERSAMPLE_FACTOR = 5

# Acceptable price band for the synthetic asset universe (USD).
MIN_PRICE, MAX_PRICE = 125, 250

# Random seeds. 109 produces the reusable RNG snapshot used to redraw QAOA
# parameter points inside the inner loop; 50 produces the RNG snapshot used
# to draw the per-experiment asset cloud from the GaussianCopula models.
# Both must remain fixed across runs to keep results reproducible.
SEED_PARAM_DRAW = 109
SEED_SAMPLE_CLOUD = 50

# Pre-fitted GaussianCopula models for the asset universe (price/return)
# and covariance entries. See Utils.qaoaCUDAQ + the make_fig*.py scripts
# for how they are built.
COPULA_PRICE_RETURN_PKL = "./models/gaussian_copula.pkl"
COPULA_COVARIANCE_PKL = "./models/gaussian_copula_covariance.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Barren-plateau sweep (X-mixer baseline)")
    parser.add_argument("-Q", "--qubit", nargs="+", type=int, default=[5],
                        help="Target qubit counts, e.g. -Q 4 5 6")
    parser.add_argument("-A", "--asset", nargs="+", type=int, default=[3, 4, 5],
                        help="Asset counts, e.g. -A 3 4 5")
    parser.add_argument("-L", "--lamb", type=float, default=4.0,
                        help="Budget penalty λ")
    parser.add_argument("-q", type=float, default=0.0,
                        help="Volatility weight q")
    parser.add_argument("-l", "--layer", type=int, default=5,
                        help="Number of QAOA layers")
    parser.add_argument("-ed", "--end_iter", type=int, default=DEFAULT_LOOP_COUNT,
                        help="Run iterations [0, end_iter)")
    parser.add_argument("-N", type=int, default=DEFAULT_NUM_SAMPLES,
                        help="Number of random parameter points per cell")
    return parser.parse_args()


def _format_param(value: float) -> str:
    """Format a numeric param as int when it has no fractional part."""
    return str(int(value)) if value.is_integer() else str(value)


def _draw_asset_universe(target_qubit_in, n_assets_in, num_samples_per_cell, sample_state):
    """Draw and filter the (price, return) and covariance samples used by the loop."""
    np.random.set_state(sample_state)
    price_return = joblib.load(COPULA_PRICE_RETURN_PKL).sample(
        int(max(n_assets_in) * DEFAULT_LOOP_COUNT * OVERSAMPLE_FACTOR)
    )
    price_return = price_return[
        (price_return["Price"] > MIN_PRICE) & (price_return["Price"] < MAX_PRICE)
    ]
    price_return = price_return.to_numpy()
    assert price_return.shape[0] > max(n_assets_in) * DEFAULT_LOOP_COUNT, (
        "Increase OVERSAMPLE_FACTOR — not enough samples after price filtering"
    )

    np.random.set_state(sample_state)
    covariance = joblib.load(COPULA_COVARIANCE_PKL).sample(
        int(max(n_assets_in) * DEFAULT_LOOP_COUNT)
    )
    covariance = np.abs(covariance.to_numpy())

    return price_return, covariance


def main() -> None:
    cudaq.set_target("nvidia")
    pd.set_option("display.width", 1000)

    args = parse_args()
    target_qubit_in = args.qubit
    n_assets_in = args.asset
    lamb = args.lamb
    q_weight = args.q
    layer_count = args.layer
    iter_end = args.end_iter
    num_samples = args.N

    # Build two RNG snapshots: one for in-loop parameter draws, one for
    # the up-front asset-universe sample. They must not be interleaved —
    # the per-iteration `set_state` calls below restore each snapshot
    # explicitly so the two streams stay synchronised across runs.
    np.random.seed(SEED_PARAM_DRAW)
    param_draw_state = np.random.get_state()
    np.random.seed(SEED_SAMPLE_CLOUD)
    sample_cloud_state = np.random.get_state()

    samples, samples_cov = _draw_asset_universe(
        target_qubit_in, n_assets_in, num_samples, sample_cloud_state
    )
    print(f"Drew {samples.shape[0]} (price, return) samples after price filter")

    # Snapshot the RNG state at the entry of the (qubit, asset) sweep so
    # each (qubit, asset) cell starts from an identical RNG state.
    np.random.set_state(sample_cloud_state)
    state_init_loop = np.random.get_state()

    for target_qubit in target_qubit_in:
        for n_assets in n_assets_in:
            if target_qubit < n_assets:
                continue

            print(
                f"Target Qubit: {target_qubit}, N Assets: {n_assets}, L: {lamb}, "
                f"q: {q_weight}, layers: {layer_count}, N: {num_samples}, ED: {iter_end}"
            )
            dir_name = (
                f"exp_Q{target_qubit}_A{n_assets}"
                f"_L{_format_param(lamb)}_q{_format_param(q_weight)}"
            )
            dir_path = f"./experiments_plateau_X/{dir_name}"
            os.makedirs(dir_path, exist_ok=True)

            np.random.set_state(state_init_loop)

            pbar = tqdm(range(iter_end))
            for i in pbar:
                pbar.set_description("X:init_1")
                price = samples[i * n_assets:(i + 1) * n_assets, 0]
                returns = samples[i * n_assets:(i + 1) * n_assets, 1]
                cov = samples_cov[i * n_assets:(i + 1) * n_assets, :n_assets]
                for j in range(cov.shape[0]):
                    cov[j] = np.roll(cov[j], j)
                cov = (cov + cov.T) / 2

                budget = find_budget(target_qubit, price, MIN_PRICE, MAX_PRICE)
                P_bb, ret_bb, cov_bb, n_qubit, _, _ = po_normalize(budget, price, returns, cov)

                pbar.set_description("X:init_2")
                qubo = -ret_cov_to_QUBO(ret_bb, cov_bb, P_bb, lamb, q_weight)
                hamiltonian = qubo_to_ising(qubo, lamb).canonicalize()
                idx_1, c1, idx_2_a, idx_2_b, c2 = process_ansatz_values(hamiltonian)
                c1, c2 = np.array(c1), np.array(c2)

                # Scale the random θ-range for the QUBO term by π / |smallest non-zero
                # coefficient|, so the parameter draw covers a meaningful slice of the
                # cost-Hamiltonian phase space without underflowing on small couplings.
                min_abs_coeff = min(
                    np.min(np.abs(c1)) if len(c1) else 1e9,
                    np.min(np.abs(c2)) if len(c2) else 1e9,
                )
                theta_scale = np.pi / min_abs_coeff

                parameter_count = layer_count * 2
                ansatz_fixed_param = (
                    int(n_qubit), layer_count, idx_1, c1, idx_2_a, idx_2_b, c2,
                )

                # Resume from a partial run if a report.csv already covers this cell.
                report_path = f"{dir_path}/report.csv"
                resume_iter, sum_1, sum_2 = 0, 0.0, 0.0
                df_now = pd.read_csv(report_path) if os.path.exists(report_path) else None
                if df_now is not None and df_now.shape[0] > i:
                    if df_now.iloc[i]["N"] >= num_samples:
                        continue
                    resume_iter, sum_1, sum_2 = df_now.iloc[i]
                    resume_iter = int(resume_iter)

                np.random.set_state(param_draw_state)
                points = np.random.uniform(-1, 1, (num_samples, parameter_count))
                points[:, ::2] *= theta_scale
                points[:, 1::2] *= np.pi

                pbar.set_description("X:observing")
                expectations = []
                for ii in tqdm(range(resume_iter, num_samples), leave=False):
                    expectations.append(float(
                        cudaq.observe(
                            kernel_qaoa_X, hamiltonian, points[ii], *ansatz_fixed_param,
                        ).expectation()
                    ))
                expectations = np.array(expectations)
                sum_1 += expectations.sum()
                sum_2 += (expectations ** 2).sum()

                write_df(report_path, REPORT_COL, num_samples, sum_1, sum_2, idx=i)


if __name__ == "__main__":
    main()
