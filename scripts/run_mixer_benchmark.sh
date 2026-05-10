#!/bin/bash
# run_mixer_benchmark.sh — orchestrates PO_Mixer_Benchmark.py runs (§03 of the paper).
#
# Modes:
#   full                  Full sweep across {Q=10,11,12,14} × {A=3,4,5} × {B=3,6,12}.
#   mini                  Quick smoke test on Q=4 (X mixer + 4-way-split Preserving).
#   split <GPU>           Q=17 high-qubit run, sliced across 4 GPUs (GPU = 0..3).
#   merge <Q> <A> <B>     4-way-split Preserving + Merger for arbitrary (Q, A, B).
#
# All commands assume the active conda env has CUDA-Q + ga_solver installed.
# Run from the experiments/ directory so '../dataset/' resolves.

set -euo pipefail
mode=${1:-full}

case "$mode" in
  full)
    for q in 10 11 12 14; do
      for a in 3 4 5; do
        python PO_Mixer_Benchmark.py -Q "$q" -A "$a" -B 3 6 12
      done
    done
    ;;

  mini)
    python PO_Mixer_Benchmark.py -Q 4 -A 3 -B 3 -m X
    for range in "0 25" "25 50" "50 75" "75 100"; do
      read -r st ed <<< "$range"
      python PO_Mixer_Benchmark.py -Q 4 -A 3 -B 3 -st "$st" -ed "$ed" -m Preserving
    done
    ;;

  split)
    gpu=${2:-0}
    case "$gpu" in
      0) st=0  ; ed=25  ;;
      1) st=25 ; ed=50  ;;
      2) st=50 ; ed=75  ;;
      3) st=75 ; ed=100 ;;
      *) echo "GPU id must be 0..3, got '$gpu'"; exit 1 ;;
    esac
    export CUDA_VISIBLE_DEVICES="$gpu"
    python PO_Mixer_Benchmark.py -Q 17 -A 3 -B 25 -st "$st" -ed "$ed"
    ;;

  merge)
    Q=${2:?usage: merge <Q> <A> <B>}
    A=${3:?usage: merge <Q> <A> <B>}
    B=${4:?usage: merge <Q> <A> <B>}
    for range in "0 25" "25 50" "50 75" "75 100"; do
      read -r st ed <<< "$range"
      python PO_Mixer_Benchmark.py -Q "$Q" -A "$A" -B "$B" -st "$st" -ed "$ed" -m Preserving
    done
    python PO_Mixer_Benchmark_Merger.py -Q "$Q" -A "$A" -B "$B"
    ;;

  *)
    echo "Usage: $0 {full|mini|split <GPU>|merge <Q> <A> <B>}"
    exit 1 ;;
esac
