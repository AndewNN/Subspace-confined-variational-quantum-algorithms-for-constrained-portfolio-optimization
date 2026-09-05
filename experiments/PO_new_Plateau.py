"""Barren-plateau diagnostics on the SC-QAOA Preserving (or X) ansatz over
real asset data, sweeping (asset count) × (sample count N).

Loads the 50-ticker universe from ../dataset/, picks N_ASSETS tickers per
experiment, builds the QAOA Hamiltonian, samples random parameter points,
and accumulates ⟨H⟩ moments to compute variance scaling — the diagnostic
underlying §03b of the paper.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

import cudaq

import _paths  # noqa: F401  (puts project root on sys.path for ``from Utils.…``)
from _data import load_universe
from Utils.qaoaCUDAQ import (
    all_state_to_return,
    basis_T_to_pauli,
    find_budget,
    get_init_states,
    kernel_qaoa_Preserved,
    kernel_qaoa_X,
    po_normalize,
    process_ansatz_values,
    qubo_to_ising,
    ret_cov_to_QUBO,
    reversed_str_bases_to_init_state,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_MODES = ["X", "Preserving"]
REPORT_COL = ["Exp", "Assets", "Qubits", "N", "Sum_1", "Sum_2", "Coeff", "Budget"]

DEFAULT_NUM_SAMPLES = 2000
DEFAULT_TARGET_QUBIT = 3
DEFAULT_TARGET_ASSET = [3, 4, 5, 6, 7]
MIN_PRICE, MAX_PRICE = 95, 190


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plateau diagnostics over real asset data")
    parser.add_argument("-E", "--exp", type=int, default=100,
                        help="Number of independent experiments")
    parser.add_argument("-Q", "--qubit", type=int, default=DEFAULT_TARGET_QUBIT,
                        help="Number of qubits per asset")
    parser.add_argument("-A", "--asset", nargs="+", type=int, default=DEFAULT_TARGET_ASSET,
                        help="Asset counts to sweep, e.g. -A 3 4 5 6 7")
    parser.add_argument("-L", "--lamb", type=float, default=0.001,
                        help="Budget penalty λ")
    parser.add_argument("-q", type=float, default=1.0,
                        help="Volatility weight q")
    parser.add_argument("-p", "--layer", type=int, default=5,
                        help="Number of QAOA layers")
    parser.add_argument("-N", type=int, default=DEFAULT_NUM_SAMPLES,
                        help="Number of random parameter points per (exp, asset) cell")
    parser.add_argument("-Z", "--basis", type=int, nargs="+", default=None,
                        help="Pauli-Z observable basis (list of qubit indices); "
                             "default uses the full QUBO Hamiltonian")
    parser.add_argument("-m", "--mode", type=str, default="X",
                        help=f"Mode, one of {ALL_MODES}")
    parser.add_argument("-B", "--bases", type=int, default=12,
                        help="Number of preserving bases (only used when -m Preserving)")
    parser.add_argument("--OVERWRITE", action="store_true",
                        help="Overwrite existing rows instead of resuming")
    return parser.parse_args()


def _format_param(value: float) -> str:
    return str(int(value)) if value.is_integer() else str(value)


def _build_observable(Z: list[int] | None, H_ansatz):
    """Pauli-Z product over the listed qubits, or the full QUBO Hamiltonian if Z is None."""
    if Z is None:
        return H_ansatz
    H = 1
    for q in Z:
        H = H * cudaq.spin.z(q)
    return H


def _resume_or_seed_row(df_now, e: int, n_assets: int, target_qubit: int,
                       num_samples: int, overwrite: bool):
    """Return ``(df, it_st, sum_1, sum_2, should_skip)`` for the current cell.

    Encapsulates the mid-CSV resume logic: if the row exists with N >= target,
    skip; if it exists with smaller N, pick up where it left off; if it doesn't
    exist, append a fresh placeholder row.
    """
    if df_now is None or overwrite:
        return (
            pd.DataFrame(
                np.array([e, n_assets, target_qubit * n_assets, 0, 0.0, 0.0, 0.0, 0.0])[None, :],
                columns=REPORT_COL,
            ),
            0, 0.0, 0.0, False,
        )
    mask = (df_now["Assets"] == n_assets) & (df_now["Exp"] == e)
    if mask.any():
        if df_now.loc[mask, "N"].values[0] >= num_samples:
            return df_now, 0, 0.0, 0.0, True
        n_done, sum_1, sum_2 = df_now.loc[mask, ["N", "Sum_1", "Sum_2"]].values[0]
        return df_now, int(n_done), float(sum_1), float(sum_2), False
    df_now.loc[-1] = [e, n_assets, target_qubit * n_assets, 0, 0.0, 0.0, 0.0, 0.0]
    return df_now, 0, 0.0, 0.0, False


def main() -> None:
    cudaq.set_target("nvidia")
    pd.set_option("display.width", 1000)

    args = parse_args()
    target_qubit = args.qubit
    target_asset = args.asset
    lamb = args.lamb
    q_weight = args.q
    layer_count = args.layer
    num_samples = args.N
    Z = args.basis
    num_experiments = args.exp
    mode = args.mode
    num_init_bases = args.bases
    overwrite = args.OVERWRITE

    assert mode in ALL_MODES, f"Mode {mode} not in {ALL_MODES}"

    returns_price, covariance = load_universe(MIN_PRICE, MAX_PRICE)
    cov_no_ticker = covariance.drop("Ticker", axis=1)
    ret_no_ticker = returns_price.drop("Ticker", axis=1)
    # Company_Name is a string column — dropping it keeps the ndarray numeric.
    ret_no_ticker = ret_no_ticker.drop("Company_Name", axis=1)

    dir_name = f"exp_p{layer_count}_L{_format_param(lamb)}_q{_format_param(q_weight)}"
    dir_path = (
        f"./experiments_plateau_{mode}{'' if mode == 'X' else str(num_init_bases)}"
        f"_Q{target_qubit}/{dir_name}"
    )
    file_name = f"report_{'Hall' if Z is None else 'Z' + ''.join(str(z) for z in Z)}.csv"
    os.makedirs(dir_path, exist_ok=True)

    print(
        f"Experiments: {num_experiments}, Qubits/Asset: {target_qubit}, "
        f"Assets: {target_asset}, Lambda: {lamb}, q: {q_weight}, "
        f"Layers: {layer_count}, N: {num_samples}, Z: {Z}, mode: {mode}"
        + (f", num_init_bases: {num_init_bases}" if mode == "Preserving" else "")
    )

    for e in tqdm(range(num_experiments)):
        for n_assets in tqdm(target_asset, leave=False):
            df_path = f"{dir_path}/{file_name}"
            df_now = pd.read_csv(df_path) if os.path.exists(df_path) else None
            df_now, it_st, sum_1, sum_2, skip = _resume_or_seed_row(
                df_now, e, n_assets, target_qubit, num_samples, overwrite,
            )
            if skip:
                continue

            # Per-experiment seed mixing — fixed primes so each (e, n_assets)
            # cell gets a unique but reproducible RNG state.
            np.random.seed(911 + 991 * e + 997 * n_assets)
            state = np.random.get_state()

            asset_idx = np.random.choice(cov_no_ticker.shape[0], n_assets, replace=False)
            data_cov = cov_no_ticker.to_numpy()[asset_idx, :][:, asset_idx]
            data_ret_p = ret_no_ticker.to_numpy()[asset_idx, :]
            data_ret = data_ret_p[:, 0]
            data_p = data_ret_p[:, 1]

            # Pick a budget uniformly between the min and max attainable for this universe.
            np.random.set_state(state)
            weighted = np.random.uniform(0, 1)
            B_min, B_max = find_budget(
                target_qubit * n_assets, data_p, MIN_PRICE, MAX_PRICE, min_mix_mode=True,
            )
            budget = B_min * weighted + B_max * (1 - weighted)
            P_bb, ret_bb, cov_bb, n_qubit, _, _ = po_normalize(
                budget, data_p[:n_assets], data_ret[:n_assets], data_cov[:n_assets, :n_assets],
            )

            QU = ret_cov_to_QUBO(ret_bb, cov_bb, P_bb, lamb, q_weight)
            H_ansatz = -qubo_to_ising(QU, lamb).canonicalize()
            H = _build_observable(Z, H_ansatz)

            idx_1, c1, idx_2_a, idx_2_b, c2 = process_ansatz_values(H_ansatz)
            c1, c2 = np.array(c1), np.array(c2)

            kernel = kernel_qaoa_X if mode == "X" else kernel_qaoa_Preserved
            parameter_count = layer_count * 2

            if mode == "X":
                ansatz_fixed_param = (
                    int(n_qubit), layer_count, idx_1, c1, idx_2_a, idx_2_b, c2,
                )
                mixer_c = np.array([])
            else:
                state_return = all_state_to_return(n_qubit, lamb, QU)
                init_state = get_init_states(state_return, num_init_bases, n_qubit)
                n_bases = len(init_state)
                T = np.zeros((n_bases, n_bases), dtype=np.float32)
                T[:-1, 1:] += np.eye(n_bases - 1, dtype=np.float32)
                T[1:, :-1] += np.eye(n_bases - 1, dtype=np.float32)
                T[0, -1] = T[-1, 0] = 1.0
                mixer_s, mixer_c = basis_T_to_pauli(init_state, T, n_qubit)
                init_bases = reversed_str_bases_to_init_state(init_state, n_qubit)
                ansatz_fixed_param = (
                    int(n_qubit), layer_count, idx_1, c1, idx_2_a, idx_2_b, c2,
                    mixer_s, mixer_c, init_bases,
                )

            # Scale θ-range for the QUBO term by π / |smallest non-zero coefficient|.
            min_abs_coeff = min(
                np.min(np.abs(c1)) if len(c1) else 1e9,
                np.min(np.abs(c2)) if len(c2) else 1e9,
                np.min(np.abs(mixer_c)) if len(mixer_c) else 1e9,
            )
            theta_scale = np.pi / min_abs_coeff

            np.random.seed(4001 + 4099 * e + 4999 * n_assets)
            points = np.random.uniform(-1, 1, (num_samples, parameter_count))
            points[:, ::2] *= theta_scale
            points[:, 1::2] *= np.pi

            expectations = []
            for ii in tqdm(range(it_st, num_samples), leave=False):
                expectations.append(float(
                    cudaq.observe(kernel, H, points[ii], *ansatz_fixed_param).expectation()
                ))
            expectations = np.array(expectations)
            sum_1 += expectations.sum()
            sum_2 += (expectations ** 2).sum()

            df_now.sort_values(by=["Exp", "Assets"], inplace=True)
            mask = (df_now["Assets"] == n_assets) & (df_now["Exp"] == e)
            df_now.loc[mask, ["N", "Sum_1", "Sum_2", "Coeff", "Budget"]] = [
                num_samples, sum_1, sum_2, c2[0], budget,
            ]
            df_now.to_csv(df_path, index=False)


if __name__ == "__main__":
    main()
