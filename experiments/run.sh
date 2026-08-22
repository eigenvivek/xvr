#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATASETS="deepfluoro femur ljubljana"
METHODS="de_novo finetuned foundation"
CSV="experiments/results/registration.csv"

case "${1:-}" in
  register)
    for ds in $DATASETS; do
      for m in $METHODS; do
        sbatch "experiments/scripts/$ds/register/$m.sh"
      done
    done
    ;;

  evaluate)
    source .venv/bin/activate
    rm -f "$CSV"
    for ds in $DATASETS; do
      for m in $METHODS; do
        python experiments/evaluate.py --dataset "$ds" --result "$m" --main "$CSV"
      done
    done
    ;;

  *)
    echo "usage: $0 {register|evaluate}"
    exit 1
    ;;
esac
