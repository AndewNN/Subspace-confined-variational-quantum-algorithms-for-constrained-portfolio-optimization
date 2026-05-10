#!/bin/bash
# run_approx.sh — SC-QAOA approximation-ratio sweeps (§07 / §08 of the paper).
#
# Presets:
#   preserving_K24        Preserving mixer, K=24, A=6  (the main SC-QAOA config in Fig 5)
#   x_baseline            X mixer baseline, A=3..7    (the SP-baseline comparison in Fig 5)
#
# For other configurations, edit this script or run PO_new_ApproxRatio.py
# directly with the desired flags.
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

  *)
    cat <<EOF
Unknown preset: $preset

Available presets:
  preserving_K24    Preserving mixer, K=24, A=6  (main SC-QAOA config, Fig 5)
  x_baseline        X mixer baseline, A=3..7    (SP comparison, Fig 5)
EOF
    exit 1 ;;
esac
