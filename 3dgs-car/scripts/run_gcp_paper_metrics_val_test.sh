#!/usr/bin/env bash

set -uo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
default_python="/export/home2/reny0012/vir_env/3dgr_car_gcp/bin/python"
gcp_eval_python="${GCP_EVAL_PYTHON:-$default_python}"

if [[ $# -gt 1 || ($# -eq 1 && "$1" != "--dry-run") ]]; then
  echo "Usage: $0 [--dry-run]" >&2
  exit 2
fi

if [[ ! -x "$gcp_eval_python" ]]; then
  echo "Evaluation Python is not executable: $gcp_eval_python" >&2
  echo "Set GCP_EVAL_PYTHON to the Stage-2 CUDA environment's Python." >&2
  exit 2
fi

overall_status=0
for artery in lca rca; do
  if [[ "$artery" == "lca" ]]; then
    artery_label="LCA"
  else
    artery_label="RCA"
  fi
  config_path="$script_dir/configs/eval_gcp_paper_metric_${artery}_val_test.json"
  echo "Running $artery_label validation+test paper-metric evaluation"
  run_command=(
    "$gcp_eval_python"
    "$script_dir/evaluate_gcp.py"
    --config "$config_path"
  )
  if [[ $# -eq 1 ]]; then
    run_command+=("--dry-run")
  fi
  "${run_command[@]}"
  run_status=$?
  if [[ $run_status -ne 0 ]]; then
    echo "$artery_label evaluation failed with status $run_status" >&2
    overall_status=$run_status
  fi
done

exit "$overall_status"
