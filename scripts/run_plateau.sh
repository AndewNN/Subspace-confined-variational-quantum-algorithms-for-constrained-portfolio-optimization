#!/bin/bash
# run_plateau.sh — barren-plateau diagnostics (§03b of the paper).
#
# Modes:
#   x_mixer <GPU> <L> <Q_PARAM> [QSTART-QEND-QSTEP]
#       Sweeps PO_X_Plateau.py over N=500..4000..500 and the given qubit range.
#       Original 8-script combinations:
#           GPU=0 L=0.001 Q_PARAM=1
#           GPU=1 L=0.002 Q_PARAM=1
#           GPU=2 L=0.001 Q_PARAM=2
#           GPU=3 L=0.002 Q_PARAM=2
#       Each was run once with QSTART-QEND-QSTEP=4-20-2 and once with 5-19-2.
#
#   preserving
#       PO_new_Plateau.py with Preserving mixer (B=12), N=750, p=1..15..2.
#
# Run from the experiments/ directory.

set -euo pipefail
mode=${1:-preserving}

case "$mode" in
  x_mixer)
    GPU=${2:?usage: x_mixer <GPU> <L> <Q_PARAM> [QSTART-QEND-QSTEP]}
    L=${3:?usage: x_mixer <GPU> <L> <Q_PARAM> [QSTART-QEND-QSTEP]}
    Q_PARAM=${4:?usage: x_mixer <GPU> <L> <Q_PARAM> [QSTART-QEND-QSTEP]}
    RANGE=${5:-4-20-2}
    QSTART=$(echo "$RANGE" | cut -d- -f1)
    QEND=$(echo "$RANGE"   | cut -d- -f2)
    QSTEP=$(echo "$RANGE"  | cut -d- -f3)
    export CUDA_VISIBLE_DEVICES="$GPU"
    export RUN_A=3
    for N in 500 1000 1500 2000 2500 3000 3500 4000; do
      for Q in $(seq "$QSTART" "$QSTEP" "$QEND"); do
        python PO_X_Plateau.py -Q "$Q" -A "$RUN_A" -N "$N" -q "$Q_PARAM" -L "$L"
      done
    done
    ;;

  preserving)
    for N in 750; do
      for p in 1 3 5 7 9 11 13 15; do
        python PO_new_Plateau.py -Q 2 -A 2 3 4 5 6 -N "$N" -E 20 -p "$p" -Z 2 3 -m Preserving -B 12
      done
    done
    ;;

  *)
    echo "Usage: $0 {x_mixer <GPU> <L> <Q_PARAM> [QRANGE]|preserving}"
    exit 1 ;;
esac
