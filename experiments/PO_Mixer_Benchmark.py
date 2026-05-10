"""Soft-penalty (X-mixer) baseline benchmark vs. the subspace-confined
(Preserving-mixer) ansatz, across (qubit, asset, num_init_bases) configs.

Produces per-iteration ``X.csv`` / ``Preserving.csv`` reports and per-iteration
``expectations_*/expectations_<i>.npy`` artefacts under
``./experiments/exp_Q*_A*_L*_q*_B*/``. Used to populate Fig. 1 of the paper.
"""
from __future__ import annotations

import argparse
import os
import shutil
import time

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

import cudaq
import torch  # noqa: F401  — imported only to print available CUDA devices below

import _paths  # noqa: F401  (puts project root on sys.path for ``from Utils.…``)
from Utils.qaoaCUDAQ import (
    all_state_to_return,
    basis_T_to_pauli,
    clip_df,
    find_budget,
    get_init_states,
    get_optimizer,
    kernel_flipped,
    kernel_qaoa_Preserved,
    kernel_qaoa_X,
    po_normalize,
    process_ansatz_values,
    qubo_to_ising,
    ret_cov_to_QUBO,
    reversed_str_bases_to_init_state,
    write_df,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_MODES = ["X", "Preserving"]
REPORT_COL = [
    "Approximate_ratio", "MaxProb_ratio",
    "init_1_time", "init_2_time", "optim_time", "observe_time",
]

DEFAULT_LOOP_COUNT = 100
OVERSAMPLE_FACTOR = 5  # samples drawn from copula before price-band filter
OVER_BUDGET_BOUND = 1.0  # admissible budget band: [0, B * OVER_BUDGET_BOUND]
MIN_PRICE, MAX_PRICE = 125, 250

DEFAULT_X_BOOST = 1
DEFAULT_PRESERVING_BOOST = 2000

# Seed for the per-(qubit, asset) RNG snapshot used to draw the asset cloud
# from the GaussianCopula models. Must remain fixed across runs.
SEED_SAMPLE_CLOUD = 50

COPULA_PRICE_RETURN_PKL = "./models/gaussian_copula.pkl"
COPULA_COVARIANCE_PKL = "./models/gaussian_copula_covariance.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mixer-benchmark sweep (X vs Preserving)")
    parser.add_argument("-Q", "--qubit", nargs="+", type=int, default=[5],
                        help="Target qubit counts, e.g. -Q 4 5 6")
    parser.add_argument("-A", "--asset", nargs="+", type=int, default=[3, 4, 5],
                        help="Asset counts, e.g. -A 3 4 5")
    parser.add_argument("-L", "--lamb", type=int, default=4,
                        help="Budget penalty λ (used by the X-mixer arm)")
    parser.add_argument("-B", "--bases", nargs="+", type=int, default=[3, 6, 12],
                        help="Subspace sizes K, e.g. -B 3 6 12 25")
    parser.add_argument("-q", type=int, default=0,
                        help="Volatility weight q")
    parser.add_argument("-l", "--layer", type=int, default=5,
                        help="QAOA layers L")
    parser.add_argument("-st", "--start_iter", type=int, default=0,
                        help="Resume from this iteration index")
    parser.add_argument("-ed", "--end_iter", type=int, default=DEFAULT_LOOP_COUNT,
                        help="Run iterations [start_iter, end_iter)")
    parser.add_argument("-m", "--mode", nargs="+", type=str, default=ALL_MODES,
                        help=f"Modes to run, subset of {ALL_MODES}")
    parser.add_argument("-hx", "--hamiltonian_x_boost", type=float, default=DEFAULT_X_BOOST,
                        help="X-mixer Hamiltonian boost factor α")
    parser.add_argument("-hp", "--hamiltonian_p_boost", type=float, default=DEFAULT_PRESERVING_BOOST,
                        help="Preserving-mixer Hamiltonian boost factor α")
    return parser.parse_args()


def file_copy(src: str, dst: str) -> None:
    """``shutil.copyfile`` that silently no-ops when src and dst are identical."""
    try:
        shutil.copyfile(src, dst)
    except shutil.SameFileError:
        pass


def _print_devices() -> None:
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
    target = cudaq.get_target()
    print("Number of QPUs:", target.num_qpus())


def _draw_asset_universe(n_assets_in, sample_state):
    np.random.set_state(sample_state)
    price_return = joblib.load(COPULA_PRICE_RETURN_PKL).sample(
        int(max(n_assets_in) * DEFAULT_LOOP_COUNT * OVERSAMPLE_FACTOR)
    )
    price_return = price_return[
        (price_return["Price"] > MIN_PRICE) & (price_return["Price"] < MAX_PRICE)
    ]
    print(price_return.shape)
    price_return = price_return.to_numpy()
    assert price_return.shape[0] > max(n_assets_in) * DEFAULT_LOOP_COUNT, (
        "Increase OVERSAMPLE_FACTOR — not enough samples after price filtering"
    )
    return price_return


def _summarise_run(dir_path: str, modes: list[str]) -> None:
    """Compute mean approx-ratio / timings for the X and Preserving runs and persist."""
    if not ("X" in modes and "Preserving" in modes):
        return
    df_X = pd.read_csv(f"{dir_path}/X.csv")
    df_P = pd.read_csv(f"{dir_path}/Preserving.csv")

    metrics_X = {col: df_X[col].mean() for col in REPORT_COL}
    metrics_P = {col: df_P[col].mean() for col in REPORT_COL}

    print(f"Approximate ratio  X: {metrics_X['Approximate_ratio']:.4f}, "
          f"Preserving: {metrics_P['Approximate_ratio']:.4f}")
    print(f"MaxProb ratio      X: {metrics_X['MaxProb_ratio']:.4f}, "
          f"Preserving: {metrics_P['MaxProb_ratio']:.4f}")
    for stage in ("init_1_time", "init_2_time", "optim_time", "observe_time"):
        print(f"{stage:14s}     X: {metrics_X[stage]*1000:.2f} ms, "
              f"Preserving: {metrics_P[stage]*1000:.2f} ms")

    df_result = pd.DataFrame(columns=["Mode"] + REPORT_COL)
    df_result.loc[0] = ["X"] + [metrics_X[c] for c in REPORT_COL]
    df_result.loc[1] = ["Preserving"] + [metrics_P[c] for c in REPORT_COL]
    df_result.to_csv(f"{dir_path}/result.csv", index=False)


def main() -> None:
    cudaq.set_target("nvidia")
    pd.set_option("display.width", 1000)

    args = parse_args()
    target_qubit_in = args.qubit
    n_assets_in = args.asset
    lamb = args.lamb
    q_weight = args.q
    layer_count = args.layer
    num_init_bases_in = args.bases
    iter_start = args.start_iter
    iter_end = args.end_iter
    modes = args.mode
    hamiltonian_X_boost = args.hamiltonian_x_boost
    hamiltonian_P_boost = args.hamiltonian_p_boost

    np.random.seed(SEED_SAMPLE_CLOUD)
    sample_state = np.random.get_state()

    _print_devices()
    samples = _draw_asset_universe(n_assets_in, sample_state)

    np.random.set_state(sample_state)
    state_init_loop = np.random.get_state()

    in_min, in_max = int(1e9), -int(1e9)

    for target_qubit in target_qubit_in:
        for n_assets in n_assets_in:
            for num_init_bases in num_init_bases_in:
                if target_qubit < n_assets:
                    continue

                print(
                    f"Target Qubit: {target_qubit}, N Assets: {n_assets}, "
                    f"Num Init Bases: {num_init_bases}, ST: {iter_start}, "
                    f"ED: {iter_end}, Modes: {modes}"
                )
                dir_name = f"exp_Q{target_qubit}_A{n_assets}_L{lamb}_q{q_weight}_B{num_init_bases}"
                if iter_start != 0 or iter_end != DEFAULT_LOOP_COUNT:
                    dir_name += f"_it{iter_start}-{iter_end - 1}"
                dir_name_Xbase = f"exp_Q{target_qubit}_A{n_assets}_L{lamb}_q{q_weight}_B3"
                dir_path = f"./experiments/{dir_name}"
                dir_path_Xbase = f"./experiments/{dir_name_Xbase}"

                if os.path.exists(f"{dir_path}/result.csv"):
                    print("Completed")
                    continue

                os.makedirs(dir_path, exist_ok=True)
                os.makedirs(f"{dir_path}/expectations_X", exist_ok=True)
                os.makedirs(f"{dir_path}/expectations_Preserving", exist_ok=True)

                np.random.set_state(state_init_loop)

                # Resume logic: figure out where to pick up if any per-mode CSV
                # exists, then truncate any rows past the resume point so each
                # mode's CSV ends at the same iteration index.
                restore_iter = iter_start
                if any(os.path.exists(f"{dir_path}/{m}.csv") for m in modes):
                    tmpp = int(1e9)
                    for mode in modes:
                        if os.path.exists(f"{dir_path}/{mode}.csv"):
                            df = pd.read_csv(f"./{dir_path}/{mode}.csv")
                            tmpp = min(tmpp, df.shape[0] + iter_start)
                        else:
                            tmpp = iter_start
                    restore_iter = max(restore_iter, tmpp)
                    for mode in modes:
                        path = f"{dir_path}/{mode}.csv"
                        if not os.path.exists(path):
                            continue
                        clip_df(pd.read_csv(path), restore_iter - iter_start).to_csv(
                            path, index=False,
                        )
                else:
                    for curr_dir, _, files in os.walk(dir_path):
                        for f in files:
                            os.remove(os.path.join(curr_dir, f))

                for _ in range(restore_iter):
                    np.random.rand(n_assets, n_assets)
                    np.random.uniform(-np.pi / 8, np.pi / 8, layer_count * 4)

                # If a B=3 X-mixer baseline already covers this (Q, A) config,
                # reuse it instead of recomputing.
                X_exist = False
                xbase_csv = f"{dir_path_Xbase}/X.csv"
                if os.path.exists(xbase_csv) and pd.read_csv(xbase_csv).shape[0] >= iter_end:
                    file_copy(xbase_csv, f"{dir_path}/X.csv")
                    pd.read_csv(f"{dir_path}/X.csv").iloc[iter_start:iter_end].to_csv(
                        f"{dir_path}/X.csv", index=False,
                    )
                    for f_i in range(iter_start, iter_end):
                        file_copy(
                            f"{dir_path_Xbase}/expectations_X/expectations_{f_i}.npy",
                            f"{dir_path}/expectations_X/expectations_{f_i}.npy",
                        )
                    X_exist = True

                pbar = tqdm(range(restore_iter, iter_end))
                for i in pbar:
                    pbar.set_description("global:init_1")
                    st = time.time()

                    price = samples[i * n_assets:(i + 1) * n_assets, 0]
                    returns = samples[i * n_assets:(i + 1) * n_assets, 1]
                    cov = np.random.rand(n_assets, n_assets)
                    cov += cov.T

                    budget = find_budget(target_qubit, price, MIN_PRICE, MAX_PRICE)
                    P_bb, ret_bb, cov_bb, n_qubit, _, C = po_normalize(budget, price, returns, cov)
                    state_return, in_budget = all_state_to_return(
                        budget, C, returns, price, OVER_BUDGET_BOUND,
                    )
                    init_state = get_init_states(state_return, in_budget, num_init_bases, n_qubit)

                    in_min = min(in_min, in_budget.sum())
                    in_max = max(in_max, in_budget.sum())

                    feasible_state_return = state_return * in_budget
                    max_return = state_return[int(init_state[0], 2)]
                    init_1_time = time.time() - st

                    for mode in ALL_MODES:
                        # Skip-but-still-advance-RNG: keep the parameter draw
                        # in sync across runs that include / exclude a mode.
                        if mode not in modes or (mode == "X" and X_exist):
                            np.random.uniform(-np.pi / 8, np.pi / 8, 2 * layer_count)
                            continue

                        pbar.set_description(f"{mode}:init_2")
                        st = time.time()
                        # X mixer keeps the budget penalty term; Preserving zeroes
                        # it out (feasibility is enforced architecturally instead).
                        eff_lamb = lamb if mode == "X" else 0
                        boost = hamiltonian_X_boost if mode == "X" else hamiltonian_P_boost

                        QU = -ret_cov_to_QUBO(ret_bb, cov_bb, P_bb, eff_lamb, q_weight)
                        H = qubo_to_ising(QU, eff_lamb).canonicalize() * boost
                        idx_1, c1, idx_2_a, idx_2_b, c2 = process_ansatz_values(H)

                        kernel = kernel_qaoa_X if mode == "X" else kernel_qaoa_Preserved
                        parameter_count = layer_count * 2

                        # Optimizer index 3 = the configured Adam wrapper.
                        # See Utils/qaoaCUDAQ.get_optimizer for the full mapping.
                        OPTIMIZER_IDX = 3
                        optimizer, _, FIND_GRAD = get_optimizer(OPTIMIZER_IDX)
                        optimizer.max_iterations = 1000
                        optimizer.initial_parameters = np.random.uniform(
                            -np.pi / 8, np.pi / 8, 2 * layer_count,
                        )

                        if mode == "X":
                            ansatz_fixed_param = (
                                int(n_qubit), layer_count, idx_1, c1, idx_2_a, idx_2_b, c2,
                            )
                        else:
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
                        init_2_time = time.time() - st

                        pbar.set_description(f"{mode}:optim")
                        st = time.time()
                        expectations: list[float] = []

                        def cost_func(parameters, cal_expectation=False):
                            return float(
                                cudaq.observe(kernel, H, parameters, *ansatz_fixed_param).expectation()
                            ) / boost

                        def objective(parameters):
                            return cost_func(parameters, cal_expectation=True)

                        fd = cudaq.gradients.ForwardDifference()

                        def objective_grad_cuda(parameters):
                            expectation = cost_func(parameters, cal_expectation=True)
                            gradient = fd.compute(parameters, cost_func, expectation)
                            return expectation, gradient

                        objective_func = objective_grad_cuda if FIND_GRAD else objective
                        _, optimal_parameters = optimizer.optimize(
                            dimensions=parameter_count, function=objective_func,
                        )
                        np.save(
                            f"{dir_path}/expectations_{mode}/expectations_{i}.npy",
                            np.array(expectations),
                        )
                        optim_time = time.time() - st

                        pbar.set_description(f"{mode}:observe")
                        st = time.time()
                        result = cudaq.get_state(kernel, optimal_parameters, *ansatz_fixed_param)
                        idx_r_best = np.argmax(np.abs(result))
                        idx_best = bin(idx_r_best)[2:].zfill(n_qubit)[::-1]

                        result_r = cudaq.get_state(kernel_flipped, result, target_qubit)
                        prob = np.abs(result_r) ** 2

                        approx_ratio = (prob * feasible_state_return).sum() / max_return
                        maxprob_ratio = (
                            state_return[int(idx_best, 2)] / max_return
                            if in_budget[int(idx_best, 2)] else 0.0
                        )
                        observe_time = time.time() - st

                        write_df(
                            f"{dir_path}/{mode}.csv", REPORT_COL,
                            approx_ratio, maxprob_ratio,
                            init_1_time, init_2_time, optim_time, observe_time,
                        )

                print(f"Min:Max feasible states: {in_min}:{in_max}")
                if in_min < num_init_bases:
                    with open(f"{dir_path}/flag.txt", "w") as f:
                        f.write(
                            f"Min feasible states {in_min} < num_init_bases {num_init_bases}\n"
                            f"(Max feasible states {in_max})\n"
                        )

                _summarise_run(dir_path, modes)


if __name__ == "__main__":
    main()
