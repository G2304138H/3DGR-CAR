#!/usr/bin/env python3
"""Visualize one LCA or RCA case using GCP then Gaussian optimization.

The artery-specific JSON remains the source of model, checkpoint, geometry,
optimization, metric, and rendering settings.  This small entry point only
selects the configuration and requested case number, then delegates the full
pipeline to :mod:`evaluate_gcp`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import evaluate_gcp


CONFIG_FILENAMES = {
    "lca": "eval_gcp_visualisation_lca_case.json",
    "rca": "eval_gcp_visualisation_rca_case.json",
}


def _positive_case_number(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "case number must be a positive integer"
        ) from error
    if value < 1:
        raise argparse.ArgumentTypeError(
            "case number must be a positive integer"
        )
    return value


def _load_json_object(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, Mapping):
        raise ValueError(f"Visualization configuration must be an object: {path}")
    return value


def artery_config_path(artery: str) -> Path:
    return (
        Path(__file__).resolve().parent
        / "configs"
        / CONFIG_FILENAMES[str(artery).lower()]
    )


def default_case_output_dir(config_path: Path, case_label: str) -> Path:
    config = _load_json_object(config_path)
    raw_output = config.get("eval_output_dir")
    if raw_output is None or not str(raw_output).strip():
        model = config.get("model")
        if not isinstance(model, Mapping) or not model.get("experiment_dir"):
            raise ValueError(
                "Set eval_output_dir or model.experiment_dir in the "
                f"visualization configuration: {config_path}"
            )
        raw_output = Path(str(model["experiment_dir"])) / "evaluation_visualisation"
    output = Path(str(raw_output)).expanduser()
    if not output.is_absolute():
        output = config_path.parent / output
    return (output / case_label).resolve()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize one LCA or RCA case with GCP initialization followed "
            "by per-case Gaussian primitive optimization."
        )
    )
    parser.add_argument("--artery", required=True, choices=("lca", "rca"))
    parser.add_argument(
        "--case-number",
        "--case_number",
        required=True,
        type=_positive_case_number,
        help="Numeric ImageCAS case identifier, for example 508.",
    )
    parser.add_argument(
        "--split",
        choices=("val", "test", "val_test"),
        default=None,
        help="Override the split in the artery configuration.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional pretrained-weight path or best/latest override.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional exact output directory. By default a canonical case "
            "subdirectory is created below eval_output_dir from the JSON."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the resolved configuration and plan without CUDA work.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    artery = str(args.artery).lower()
    case_label = f"{artery}_{int(args.case_number):04d}"
    config_path = artery_config_path(artery)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else default_case_output_dir(config_path, case_label)
    )

    evaluation_arguments = [
        "--config",
        str(config_path),
        "--case-id",
        case_label,
        "--output-dir",
        str(output_dir),
    ]
    if args.split is not None:
        evaluation_arguments.extend(("--split", str(args.split)))
    if args.checkpoint is not None:
        evaluation_arguments.extend(("--checkpoint", str(args.checkpoint)))
    if args.dry_run:
        evaluation_arguments.append("--dry-run")

    print(
        f"Visualizing {case_label}: GCP initialization -> Gaussian optimization"
    )
    print(f"Configuration: {config_path}")
    print(f"Output: {output_dir}")
    return evaluate_gcp.main(evaluation_arguments)


if __name__ == "__main__":
    sys.exit(main())
