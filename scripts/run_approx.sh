#!/bin/bash
# run_approx.sh — SC-QAOA approximation-ratio sweeps (§07 / §08 of the paper).
#
# Presets:
#   preserving_K24        Preserving mixer, K=24, A=6  (the main SC-QAOA config in Fig 5)
#   x_baseline            X mixer baseline, A=3..7    (the SP-baseline comparison in Fig 5)
#   lr_x_sweep            X mixer, linear-ramp init, (A × λ × p) sweep with -norm Jh
#   lr_preserving_sweep   Preserving mixer, linear-ramp init, (A × p) sweep with -norm Jh
#
# The two lr_* presets are the auto-normalized sweeps: instead of a hand-tuned
# -b_X / -b_P per cell, "-norm Jh" derives the Hamiltonian boost from the Ising
# couplings (boost = 1 / max(max|J_ij|, max|h_i|)) and "--LR_init" starts the
# optimizer from the linear-ramp schedule. See the module docstring of
# experiments/PO_new_ApproxRatio.py for the full description.
#
# To split a sweep across GPUs, give each slice its own "-post <tag>" and a
# disjoint "-e_list", then stitch the slices back together:
#   CUDA_VISIBLE_DEVICES=0 python PO_new_ApproxRatio.py ... -post job1 -e_list 0 1 2 3 4 &
#   CUDA_VISIBLE_DEVICES=1 python PO_new_ApproxRatio.py ... -post job2 -e_list 5 6 7 8 9 &
#   wait
#   python merge_split_results.py -root <root> -post job1 job2
#
# Run from the experiments/ directory.

set -euo pipefail
preset=${1:-preserving_K24}

case "$preset" in
  preserving_K24)
    python PO_new_ApproxRatio.py -Q 2 -A 6 -E 35 -p 5 -B 24 -m Preserving -L 0.0005 -q 0.15
    ;;

  x_baseline)
    python PO_new_ApproxRatio.py -Q 2 -A 3 4 5 6 7 -E 35 -p 5 -m X -L 0.05 -q 0.15 -b_X 250
    ;;

  lr_x_sweep)
    for assets in 3 4 5 6 7 8; do
      for lambda in 0.0005 0.005 0.05 0.5 5; do
        for layer in 5 7 9; do
          python PO_new_ApproxRatio.py -Q 2 -A "$assets" -q 1.5 -m X -L "$lambda" \
            -b_X 0 -E 10 -p "$layer" -norm Jh -d_b 0.3 -d_g 0.6 --LR_init
        done
      done
    done
    ;;

  lr_preserving_sweep)
    for assets in 3 4 5 6 7; do
      for layer in 5 7 9; do
        python PO_new_ApproxRatio.py -Q 2 -A "$assets" -q 1.5 -m Preserving -L 1 \
          -b_P 0 -E 10 -p "$layer" -norm Jh -d_b 0.3 -d_g 0.6 --LR_init
      done
    done
    ;;

  *)
    cat <<EOF
Unknown preset: $preset

Available presets:
  preserving_K24        Preserving mixer, K=24, A=6  (main SC-QAOA config, Fig 5)
  x_baseline            X mixer baseline, A=3..7    (SP comparison, Fig 5)
  lr_x_sweep            X mixer, linear-ramp init, (A × λ × p) sweep, -norm Jh
  lr_preserving_sweep   Preserving mixer, linear-ramp init, (A × p) sweep, -norm Jh
EOF
    exit 1 ;;
esac
