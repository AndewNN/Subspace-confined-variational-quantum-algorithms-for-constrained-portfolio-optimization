"""Merge per-iteration-range outputs from PO_Mixer_Benchmark.py into one
canonical (X.csv, Preserving.csv, result.csv) per (Q, A, B) configuration.

When a sweep is split across multiple GPUs (each producing
``exp_*_it<st>-<ed>/``), this script stitches the slices back together
under the un-suffixed directory and emits an aggregate ``result.csv`` of
the means.
"""
from __future__ import annotations

import argparse
import os
import shutil

import numpy as np
import pandas as pd

ALL_MODES = ["X", "Preserving"]
REPORT_COL = [
    "Approximate_ratio", "MaxProb_ratio",
    "init_1_time", "init_2_time", "optim_time", "observe_time",
]
DEFAULT_LOOP_COUNT = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge split mixer-benchmark outputs")
    parser.add_argument("-Q", "--qubit", nargs="+", type=int, default=[5],
                        help="Target qubit counts, e.g. -Q 4 5 6")
    parser.add_argument("-A", "--asset", nargs="+", type=int, default=[3, 4, 5],
                        help="Asset counts, e.g. -A 3 4 5")
    parser.add_argument("-L", "--lamb", type=int, default=4,
                        help="Budget penalty λ (used in directory name)")
    parser.add_argument("-B", "--bases", nargs="+", type=int, default=[3, 6, 12],
                        help="Subspace sizes K, e.g. -B 3 6 12 25")
    parser.add_argument("-q", type=int, default=0,
                        help="Volatility weight q (used in directory name)")
    parser.add_argument("-ed", "--end_iter", type=int, default=DEFAULT_LOOP_COUNT,
                        help="Total iterations expected per (Q, A, B) cell")
    parser.add_argument("-m", "--mode", nargs="+", type=str, default=ALL_MODES,
                        help=f"Modes to merge, subset of {ALL_MODES}")
    return parser.parse_args()


def find_split_dirs(prefix: str, base_path: str = "./experiments") -> list[tuple[str, int, int]]:
    """Return ``(dir_name, st, ed)`` triples for every split sub-directory under
    ``base_path`` whose name starts with ``prefix`` and ends with ``_it<st>-<ed>``,
    sorted by (st, ed)."""
    out: list[tuple[str, int, int]] = []
    for dir_name in os.listdir(base_path):
        if dir_name.startswith(prefix):
            st, ed = dir_name.split("_it")[-1].split("-")
            out.append((dir_name, int(st), int(ed)))
    return sorted(out, key=lambda x: (x[1], x[2]))


def merge_one(target_qubit: int, n_assets: int, lamb: int, q_weight: int,
              num_init_bases: int, loop_count: int, modes: list[str]) -> None:
    dir_name = f"exp_Q{target_qubit}_A{n_assets}_L{lamb}_q{q_weight}_B{num_init_bases}"
    dir_path = f"./experiments/{dir_name}"

    os.makedirs(dir_path, exist_ok=True)
    for mode in ALL_MODES:
        os.makedirs(f"{dir_path}/expectations_{mode}", exist_ok=True)

    splits = find_split_dirs(dir_name + "_it")
    coverage = np.zeros(loop_count, dtype=bool)
    for _, st, ed in splits:
        coverage[st:ed + 1] = True
    assert np.all(coverage), (
        f"Not all iterations are covered for {dir_name}; "
        f"missing {np.where(~coverage)[0]}"
    )

    cursor = 0
    csvs_by_mode: list[pd.DataFrame | None] = [None for _ in modes]
    for split_dir, st, ed in splits:
        split_path = f"./experiments/{split_dir}"
        print(split_path)
        end = min(ed + 1, loop_count)
        for mode_idx, mode in enumerate(modes):
            for i in range(cursor, end):
                shutil.copyfile(
                    f"{split_path}/expectations_{mode}/expectations_{i}.npy",
                    f"{dir_path}/expectations_{mode}/expectations_{i}.npy",
                )
            split_csv = pd.read_csv(f"{split_path}/{mode}.csv")
            if csvs_by_mode[mode_idx] is None:
                csvs_by_mode[mode_idx] = split_csv
            else:
                csvs_by_mode[mode_idx] = pd.concat(
                    [csvs_by_mode[mode_idx], split_csv.iloc[cursor - st:end - st]],
                    ignore_index=True,
                )
        cursor = ed + 1

    for mode_idx, mode in enumerate(modes):
        csvs_by_mode[mode_idx].to_csv(f"{dir_path}/{mode}.csv", index=False)

    if "X" in modes and "Preserving" in modes:
        _summarise(dir_path)


def _summarise(dir_path: str) -> None:
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
    args = parse_args()
    for target_qubit in args.qubit:
        for n_assets in args.asset:
            for num_init_bases in args.bases:
                merge_one(
                    target_qubit, n_assets, args.lamb, args.q,
                    num_init_bases, args.end_iter, args.mode,
                )


if __name__ == "__main__":
    main()
