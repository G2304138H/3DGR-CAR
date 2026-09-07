#!/usr/bin/env python3
"""Config-driven GCP initialization and per-case Gaussian evaluation.

This is the high-level counterpart to ``evaluate_stage2_npz.py``.  It follows
the parametric evaluator's experiment/checkpoint/config conventions, while the
existing Stage-2 evaluator remains responsible for reconstruction, physical
ground-truth alignment, metrics, and artifact generation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import evaluate_stage2_npz as stage2_evaluation


_CHECKPOINT_FILENAMES = {
    "best": "best_gcp.pt",
    "latest": "last_gcp.pt",
}

_TRAINER_SCALAR_FLAGS = {
    "iterations": "--iterations",
    "position_lr_init": "--position_lr_init",
    "position_lr_final": "--position_lr_final",
    "position_lr_delay_mult": "--position_lr_delay_mult",
    "position_lr_max_steps": "--position_lr_max_steps",
    "density_lr": "--density_lr",
    "sigma_lr": "--sigma_lr",
    "feature_lr": "--feature_lr",
    "opacity_lr": "--opacity_lr",
    "scaling_lr": "--scaling_lr",
    "rotation_lr": "--rotation_lr",
    "percent_dense": "--percent_dense",
    "lambda_dssim": "--lambda_dssim",
    "densification_interval": "--densification_interval",
    "opacity_reset_interval": "--opacity_reset_interval",
    "densify_from_iter": "--densify_from_iter",
    "densify_until_iter": "--densify_until_iter",
    "densify_grad_threshold": "--densify_grad_threshold",
    "volume_size": "--volume-size",
    "volume_extent_m": "--volume-extent-m",
    "source_origin_distance_m": "--source-origin-distance-m",
    "fallback_detector_pixel_spacing_mm": "--fallback-detector-pixel-spacing-mm",
    "fallback_sid_m": "--fallback-sid-m",
    "num_init_gaussians": "--num-init-gaussians",
    "air_threshold": "--air-threshold",
    "initial_density": "--initial-density",
    "initial_sigma": "--initial-sigma",
    "target_type": "--target-type",
    "silhouette_gain": "--silhouette-gain",
    "silhouette_target_level": "--silhouette-target-level",
    "projection_loss_alpha": "--projection-loss-alpha",
    "centerline_threshold": "--centerline-threshold",
    "log_every": "--log-every",
    "monitor_gif_frames": "--monitor-gif-frames",
    "monitor_gif_fps": "--monitor-gif-fps",
    "gpu_index": "--gpu-index",
}

_TRAINER_BOOLEAN_FLAGS = {
    "random_background": "--random_background",
    "no_densify": "--no-densify",
    "no_volume_gif": "--no-volume-gif",
}

_FORBIDDEN_EXTRA_TRAINER_OPTIONS = {
    "--input",
    "--output-dir",
    "--view-indices",
    "--init-method",
    "--gcp-checkpoint",
    "--early-stop-checks",
    "--record-optimization-time",
    "--evaluation-cache-only",
    "--prediction-threshold",
    "--prediction-threshold-percentile",
    "--volume-gif-isovalue",
}


def _load_json_object(path: Path, *, label: str) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return dict(value)


def _resolve_path(raw: object, *, base_dir: Path, label: str) -> Path:
    if raw is None or not str(raw).strip():
        raise ValueError(f"{label} must be configured.")
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _optional_mapping(raw: object, *, label: str) -> Dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label} must be a JSON object.")
    return dict(raw)


def _normalise_checkpoint_choice(raw: object) -> str:
    value = str(raw).strip().lower().replace("-", "_")
    aliases = {
        "best_gcp": "best",
        "best_gcp.pt": "best",
        "last": "latest",
        "last_gcp": "latest",
        "last_gcp.pt": "latest",
    }
    value = aliases.get(value, value)
    if value not in _CHECKPOINT_FILENAMES:
        raise ValueError(
            "checkpoint_choice must be 'best' or 'latest', "
            f"got {raw!r}."
        )
    return value


def _normalise_evaluation_mode(raw: object) -> str:
    value = str(raw if raw is not None else "paper_metric")
    value = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "paper_metrics": "paper_metric",
        "metric": "paper_metric",
        "metrics": "paper_metric",
        "visualization": "visualisation",
        "visualization_demo": "visualisation",
        "visualisation_demo": "visualisation",
        "demo": "visualisation",
        "evaluation": "visualisation",
    }
    value = aliases.get(value, value)
    if value not in {"paper_metric", "visualisation"}:
        raise ValueError(
            "evaluation_mode must be 'paper_metric' or 'visualisation', "
            f"got {raw!r}."
        )
    return value


def _optional_positive_limit(raw: object, *, label: str) -> Optional[int]:
    if raw is None or (isinstance(raw, str) and raw.strip().lower() == "all"):
        return None
    if isinstance(raw, bool):
        raise ValueError(f"{label} must be a positive integer, null, or 'all'.")
    try:
        value = int(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{label} must be a positive integer, null, or 'all'."
        ) from error
    if value < 1:
        raise ValueError(f"{label} must be >= 1, null, or 'all'.")
    return value


def _optional_visualization_limit(raw: object) -> Optional[int]:
    if raw is None or (isinstance(raw, str) and raw.strip().lower() == "all"):
        return None
    if isinstance(raw, bool):
        raise ValueError(
            "max_visualizations must be a non-negative integer, null, or 'all'."
        )
    try:
        value = int(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "max_visualizations must be a non-negative integer, null, or 'all'."
        ) from error
    if value < 0:
        raise ValueError(
            "max_visualizations must be a non-negative integer, null, or 'all'."
        )
    return value


def resolve_view_sweep(config: Mapping[str, Any]) -> List[Tuple[int, List[int]]]:
    """Resolve parametric-style view counts into fixed view-index prefixes."""

    raw_counts = config.get("eval_num_views")
    raw_indices = config.get("eval_view_indices", config.get("view_indices"))
    if raw_counts is None:
        raw_counts = len(raw_indices) if raw_indices is not None else 2
    counts_raw = raw_counts if isinstance(raw_counts, list) else [raw_counts]
    if not counts_raw:
        raise ValueError("eval_num_views must not be an empty list.")
    counts: List[int] = []
    for raw in counts_raw:
        if isinstance(raw, bool):
            raise ValueError("eval_num_views entries must be positive integers.")
        try:
            count = int(raw)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "eval_num_views must be a positive integer or list of integers."
            ) from error
        if count < 1:
            raise ValueError("eval_num_views entries must be >= 1.")
        if count in counts:
            raise ValueError(f"eval_num_views contains duplicate count {count}.")
        counts.append(count)

    if raw_indices is None:
        indices = list(range(max(counts)))
    else:
        if not isinstance(raw_indices, list) or not raw_indices:
            raise ValueError("eval_view_indices must be a non-empty integer list.")
        indices = []
        for raw in raw_indices:
            if isinstance(raw, bool):
                raise ValueError("eval_view_indices entries must be integers >= 0.")
            try:
                index = int(raw)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "eval_view_indices entries must be integers >= 0."
                ) from error
            if index < 0:
                raise ValueError("eval_view_indices entries must be integers >= 0.")
            if index in indices:
                raise ValueError(f"eval_view_indices contains duplicate index {index}.")
            indices.append(index)
    if max(counts) > len(indices):
        raise ValueError(
            "eval_view_indices does not contain enough entries for the largest "
            f"eval_num_views value ({max(counts)} > {len(indices)})."
        )
    return [(count, indices[:count]) for count in counts]


def _resolve_experiment_and_checkpoint(
    config: Mapping[str, Any],
    *,
    config_path: Path,
    checkpoint_override: Optional[str],
) -> Tuple[Path, Path, str]:
    experiment_raw = config.get("experiment_dir")
    experiment_dir = (
        _resolve_path(
            experiment_raw,
            base_dir=config_path.parent,
            label="experiment_dir",
        )
        if experiment_raw is not None
        else None
    )
    checkpoint_raw = checkpoint_override or config.get(
        "checkpoint_path", config.get("checkpoint")
    )
    choice = _normalise_checkpoint_choice(
        config.get("checkpoint_choice", "best")
    )
    if checkpoint_raw is None:
        if experiment_dir is None:
            raise ValueError(
                "Set experiment_dir plus checkpoint_choice, or set checkpoint_path."
            )
        checkpoint_path = experiment_dir / _CHECKPOINT_FILENAMES[choice]
    else:
        checkpoint_text = str(checkpoint_raw).strip()
        try:
            checkpoint_alias = _normalise_checkpoint_choice(checkpoint_text)
        except ValueError:
            checkpoint_alias = None
        if checkpoint_alias is not None:
            if experiment_dir is None:
                raise ValueError(
                    "experiment_dir is required when checkpoint names a choice."
                )
            choice = checkpoint_alias
            checkpoint_path = experiment_dir / _CHECKPOINT_FILENAMES[choice]
        else:
            checkpoint_candidate = Path(checkpoint_text).expanduser()
            if not checkpoint_candidate.is_absolute():
                checkpoint_candidate = (
                    experiment_dir / checkpoint_candidate
                    if experiment_dir is not None
                    else config_path.parent / checkpoint_candidate
                )
            checkpoint_path = checkpoint_candidate.resolve()
            experiment_dir = experiment_dir or checkpoint_path.parent
            choice = next(
                (
                    name
                    for name, filename in _CHECKPOINT_FILENAMES.items()
                    if checkpoint_path.name == filename
                ),
                "explicit",
            )
    checkpoint_path = checkpoint_path.resolve()
    assert experiment_dir is not None
    experiment_dir = experiment_dir.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"GCP checkpoint does not exist: {checkpoint_path}")
    return experiment_dir, checkpoint_path, choice


def _path_from_eval_or_training(
    config: Mapping[str, Any],
    training_config: Mapping[str, Any],
    *,
    eval_keys: Sequence[str],
    training_keys: Sequence[str],
    config_dir: Path,
    experiment_dir: Path,
    label: str,
) -> Path:
    for key in eval_keys:
        value = config.get(key)
        if value is not None and str(value).strip():
            return _resolve_path(value, base_dir=config_dir, label=label)
    for key in training_keys:
        value = training_config.get(key)
        if value is not None and str(value).strip():
            return _resolve_path(value, base_dir=experiment_dir, label=label)
    raise ValueError(
        f"{label} is missing. Set one of {list(eval_keys)} in the evaluation config."
    )


def _three_values(raw: object, *, label: str, cast: Any) -> List[Any]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError(f"{label} must contain exactly three values.")
    try:
        return [cast(value) for value in raw]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} contains an invalid value.") from error


def resolve_evaluation_config(
    config_path: Path,
    *,
    checkpoint_override: Optional[str] = None,
    output_dir_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Load and fully resolve one JSON evaluation configuration."""

    config_path = config_path.expanduser().resolve()
    config = _load_json_object(config_path, label="Evaluation configuration")
    mode = _normalise_evaluation_mode(config.get("evaluation_mode"))
    experiment_dir, checkpoint_path, checkpoint_choice = (
        _resolve_experiment_and_checkpoint(
            config,
            config_path=config_path,
            checkpoint_override=checkpoint_override,
        )
    )

    training_config_path = experiment_dir / "training_config.json"
    training_config = (
        _load_json_object(training_config_path, label="GCP training configuration")
        if training_config_path.is_file()
        else {}
    )
    projection_dir = _path_from_eval_or_training(
        config,
        training_config,
        eval_keys=("evaluation_dataset_dir", "projection_dir", "input_dir"),
        training_keys=("projection_dir",),
        config_dir=config_path.parent,
        experiment_dir=experiment_dir,
        label="evaluation projection directory",
    )
    split_json = _path_from_eval_or_training(
        config,
        training_config,
        eval_keys=("split_json_path", "split_json"),
        training_keys=("split_json",),
        config_dir=config_path.parent,
        experiment_dir=experiment_dir,
        label="split JSON",
    )
    ground_truth_dir = _path_from_eval_or_training(
        config,
        training_config,
        eval_keys=("paper_metric_ground_truth_dir", "ground_truth_dir"),
        training_keys=("ground_truth_dir",),
        config_dir=config_path.parent,
        experiment_dir=experiment_dir,
        label="paper-metric ground-truth directory",
    )
    for path, label in (
        (projection_dir, "Evaluation projection directory"),
        (ground_truth_dir, "Ground-truth directory"),
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if not split_json.is_file():
        raise FileNotFoundError(f"Split JSON does not exist: {split_json}")

    output_raw = output_dir_override or config.get("eval_output_dir")
    if output_raw is None:
        base_name = (
            "evaluation_paper_metric"
            if mode == "paper_metric"
            else "evaluation"
        )
        output_dir = experiment_dir / base_name / checkpoint_path.stem
    else:
        output_dir = _resolve_path(
            output_raw,
            base_dir=config_path.parent,
            label="eval_output_dir",
        )

    optimization = _optional_mapping(
        config.get("gaussian_optimization"),
        label="gaussian_optimization",
    )
    for inherited_key in (
        "source_origin_distance_m",
        "volume_size",
        "fallback_detector_pixel_spacing_mm",
        "fallback_sid_m",
    ):
        if (
            inherited_key not in optimization
            and training_config.get(inherited_key) is not None
        ):
            optimization[inherited_key] = training_config[inherited_key]
    supported_optimization_keys = set(_TRAINER_SCALAR_FLAGS) | set(
        _TRAINER_BOOLEAN_FLAGS
    ) | {
        "early_stop_checks",
        "novel_view_indices",
        "densify",
        "save_volume_gif",
        "extra_trainer_args",
    }
    unknown = sorted(set(optimization) - supported_optimization_keys)
    if unknown:
        raise ValueError(f"Unknown gaussian_optimization options: {unknown}.")

    if "densify" in optimization:
        densify = optimization["densify"]
        if not isinstance(densify, bool):
            raise ValueError("gaussian_optimization.densify must be boolean.")
        if "no_densify" in optimization:
            raise ValueError("Set only one of densify and no_densify.")
        optimization["no_densify"] = not densify
    if "save_volume_gif" in optimization:
        save_gif = optimization["save_volume_gif"]
        if not isinstance(save_gif, bool):
            raise ValueError(
                "gaussian_optimization.save_volume_gif must be boolean."
            )
        if "no_volume_gif" in optimization:
            raise ValueError("Set only one of save_volume_gif and no_volume_gif.")
        optimization["no_volume_gif"] = not save_gif

    early_stop_checks = int(optimization.get("early_stop_checks", 7))
    if early_stop_checks < 1:
        raise ValueError(
            "gaussian_optimization.early_stop_checks must be positive."
        )

    eval_split = str(config.get("eval_split", "val_test")).strip()
    if not eval_split:
        raise ValueError("eval_split must not be empty.")
    eval_case_ids_raw = config.get("eval_case_ids")
    if eval_case_ids_raw is not None:
        if not isinstance(eval_case_ids_raw, list) or not eval_case_ids_raw:
            raise ValueError("eval_case_ids must be null or a non-empty list.")
        eval_case_ids = [str(value) for value in eval_case_ids_raw]
    else:
        eval_case_ids = None

    view_selection = str(config.get("eval_view_selection", "fixed")).strip().lower()
    if view_selection != "fixed":
        raise ValueError(
            "This evaluator currently supports eval_view_selection='fixed'; "
            "set eval_view_indices to choose the fixed order."
        )
    view_sweep = resolve_view_sweep(config)

    prediction_threshold = config.get("paper_metric_volume_threshold")
    if prediction_threshold is None:
        prediction_threshold = config.get("prediction_threshold")
    prediction_percentile = config.get(
        "paper_metric_prediction_threshold_percentile",
        config.get("prediction_threshold_percentile", 97.0),
    )
    if prediction_threshold is not None:
        prediction_threshold = float(prediction_threshold)
        prediction_percentile = None
    elif prediction_percentile is not None:
        prediction_percentile = float(prediction_percentile)
        if not 0.0 <= prediction_percentile < 100.0:
            raise ValueError(
                "paper_metric_prediction_threshold_percentile must be in [0, 100)."
            )

    origin_m_raw = config.get("paper_metric_volume_origin_m")
    origin_mm_raw = config.get("paper_metric_volume_origin_mm")
    if origin_m_raw is not None and origin_mm_raw is not None:
        raise ValueError(
            "Set only one of paper_metric_volume_origin_m and "
            "paper_metric_volume_origin_mm."
        )
    if origin_mm_raw is not None:
        origin_m = [
            float(value) / 1000.0
            for value in _three_values(
                origin_mm_raw,
                label="paper_metric_volume_origin_mm",
                cast=float,
            )
        ]
    else:
        origin_m = _three_values(
            origin_m_raw if origin_m_raw is not None else [0.0, 0.0, 0.0],
            label="paper_metric_volume_origin_m",
            cast=float,
        )
    direction_signs = _three_values(
        config.get("paper_metric_volume_direction_signs", [1, 1, 1]),
        label="paper_metric_volume_direction_signs",
        cast=int,
    )
    if any(value not in {-1, 1} for value in direction_signs):
        raise ValueError(
            "paper_metric_volume_direction_signs entries must be -1 or 1."
        )

    paper_save_masks = config.get("paper_metric_save_masks", True)
    if not isinstance(paper_save_masks, bool):
        raise ValueError("paper_metric_save_masks must be boolean.")
    save_ssim_map = config.get("paper_metric_save_ssim_map", False)
    if not isinstance(save_ssim_map, bool):
        raise ValueError("paper_metric_save_ssim_map must be boolean.")
    if mode == "paper_metric" and save_ssim_map and not paper_save_masks:
        raise ValueError(
            "paper_metric_save_ssim_map=true requires "
            "paper_metric_save_masks=true."
        )
    if config.get("skip_reconstruction", False):
        raise ValueError(
            "evaluate_gcp.py always performs per-case Gaussian optimization; "
            "use evaluate_stage2_npz.py --skip-reconstruction to rescore existing "
            "volumes."
        )

    resolved = {
        **config,
        "config_path": str(config_path),
        "experiment_dir": str(experiment_dir),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_choice": checkpoint_choice,
        "training_config_path": (
            str(training_config_path) if training_config_path.is_file() else None
        ),
        "evaluation_dataset_dir": str(projection_dir),
        "split_json_path": str(split_json),
        "paper_metric_ground_truth_dir": str(ground_truth_dir),
        "eval_output_dir": str(output_dir.resolve()),
        "evaluation_mode": mode,
        "eval_split": eval_split,
        "eval_case_ids": eval_case_ids,
        "num_eval_cases": config.get("num_eval_cases", "all"),
        "eval_view_selection": "fixed",
        "eval_view_selection_seed": int(
            config.get("eval_view_selection_seed", 42)
        ),
        "effective_view_sweep": [
            {"eval_num_views": count, "view_indices": indices}
            for count, indices in view_sweep
        ],
        "max_visualizations": config.get("max_visualizations", "all"),
        "paper_metric_save_masks": paper_save_masks,
        "paper_metric_save_ssim_map": save_ssim_map,
        "paper_metric_prediction_threshold_percentile": prediction_percentile,
        "paper_metric_volume_threshold": prediction_threshold,
        "paper_metric_volume_origin_m": origin_m,
        "paper_metric_volume_direction_signs": direction_signs,
        "gaussian_optimization": optimization,
        "effective_early_stop_checks": early_stop_checks,
        "training_defaults_source": (
            str(training_config_path) if training_config else None
        ),
    }
    _optional_positive_limit(resolved["num_eval_cases"], label="num_eval_cases")
    _optional_visualization_limit(resolved["max_visualizations"])
    return resolved


def _trainer_arguments(resolved: Mapping[str, Any]) -> List[str]:
    optimization = dict(resolved["gaussian_optimization"])
    arguments: List[str] = []
    for key, flag in _TRAINER_SCALAR_FLAGS.items():
        value = optimization.get(key)
        if value is not None:
            if isinstance(value, (dict, list, tuple, bool)):
                raise ValueError(f"gaussian_optimization.{key} must be a scalar.")
            arguments.extend((flag, str(value)))
    for key, flag in _TRAINER_BOOLEAN_FLAGS.items():
        value = optimization.get(key, False)
        if not isinstance(value, bool):
            raise ValueError(f"gaussian_optimization.{key} must be boolean.")
        if value:
            arguments.append(flag)

    if "novel_view_indices" in optimization:
        raw_indices = optimization["novel_view_indices"]
        if not isinstance(raw_indices, list):
            raise ValueError(
                "gaussian_optimization.novel_view_indices must be an integer list."
            )
        arguments.append("--novel-view-indices")
        arguments.extend(str(int(index)) for index in raw_indices)

    extra = optimization.get("extra_trainer_args", [])
    if not isinstance(extra, list) or any(not isinstance(value, str) for value in extra):
        raise ValueError(
            "gaussian_optimization.extra_trainer_args must be a string list."
        )
    conflicting = sorted(
        value
        for value in extra
        if value.split("=", 1)[0] in _FORBIDDEN_EXTRA_TRAINER_OPTIONS
    )
    if conflicting:
        raise ValueError(
            "extra_trainer_args cannot override evaluator-owned options: "
            f"{conflicting}."
        )
    arguments.extend(extra)
    arguments.extend(
        (
            "--init-method",
            "gcp",
            "--gcp-checkpoint",
            str(resolved["checkpoint_path"]),
        )
    )
    return arguments


def build_stage2_arguments(
    resolved: Mapping[str, Any],
    *,
    run_output_dir: Path,
    view_indices: Sequence[int],
) -> List[str]:
    """Translate the resolved JSON config to the existing split evaluator."""

    mode = str(resolved["evaluation_mode"])
    output_mode = "full" if mode == "visualisation" else "json-only"
    arguments = [
        "--input-dir",
        str(resolved["evaluation_dataset_dir"]),
        "--split-json",
        str(resolved["split_json_path"]),
        "--split",
        str(resolved["eval_split"]),
        "--output-dir",
        str(run_output_dir),
        "--ground-truth-dir",
        str(resolved["paper_metric_ground_truth_dir"]),
        "--view-indices",
        *(str(int(index)) for index in view_indices),
        "--early-stop-checks",
        str(int(resolved["effective_early_stop_checks"])),
        "--output-mode",
        output_mode,
        "--ground-truth-axis-order",
        str(resolved.get("paper_metric_ground_truth_axis_order", "auto")),
        "--ground-truth-alignment",
        str(resolved.get("paper_metric_ground_truth_alignment", "auto")),
        "--ground-truth-spacing-key",
        str(resolved.get("paper_metric_spacing_key", "spacing")),
        "--ground-truth-spacing-units",
        str(resolved.get("paper_metric_spacing_units", "mm")),
        "--ground-truth-origin-m",
        *(str(float(value)) for value in resolved["paper_metric_volume_origin_m"]),
        "--ground-truth-direction-signs",
        *(
            str(int(value))
            for value in resolved["paper_metric_volume_direction_signs"]
        ),
        "--ground-truth-interpolation",
        str(resolved.get("paper_metric_ground_truth_interpolation", "nearest")),
        "--projection-offset-mode",
        str(resolved.get("projection_offset_mode", "auto")),
        "--projection-offset-key",
        str(resolved.get("projection_offset_key", "projection_center_offset")),
        "--offset-interpolation",
        str(resolved.get("offset_interpolation", "linear")),
        "--normalisation",
        str(resolved.get("paper_metric_normalisation", "clamp")),
        "--metric-mask",
        str(resolved.get("paper_metric_mask", "ground-truth")),
        "--ssim-window-size",
        str(int(resolved.get("paper_metric_ssim_window_size", 7))),
        "--ground-truth-threshold",
        str(float(resolved.get("paper_metric_ground_truth_threshold", 0.0))),
    ]
    optional_values = (
        ("paper_metric_volume_key", "--ground-truth-key"),
        ("paper_metric_evaluation_mask_key", "--evaluation-mask-key"),
        ("evaluation_volume_extent_m", "--evaluation-volume-extent-m"),
    )
    for key, flag in optional_values:
        value = resolved.get(key)
        if value is not None:
            arguments.extend((flag, str(value)))

    if resolved.get("paper_metric_volume_threshold") is not None:
        arguments.extend(
            (
                "--prediction-threshold",
                str(float(resolved["paper_metric_volume_threshold"])),
            )
        )
    elif resolved.get("paper_metric_prediction_threshold_percentile") is not None:
        arguments.extend(
            (
                "--prediction-threshold-percentile",
                str(
                    float(
                        resolved[
                            "paper_metric_prediction_threshold_percentile"
                        ]
                    )
                ),
            )
        )

    eval_case_ids = resolved.get("eval_case_ids")
    if eval_case_ids is not None:
        arguments.extend(("--eval-case-ids", *(str(value) for value in eval_case_ids)))
    num_cases = _optional_positive_limit(
        resolved.get("num_eval_cases"), label="num_eval_cases"
    )
    if num_cases is not None:
        arguments.extend(("--num-eval-cases", str(num_cases)))

    for key, flag in (
        ("reuse_existing", "--reuse-existing"),
        ("continue_on_error", "--continue-on-error"),
        ("keep_case_cache", "--keep-case-cache"),
    ):
        value = resolved.get(key, False)
        if not isinstance(value, bool):
            raise ValueError(f"{key} must be boolean.")
        if value:
            arguments.append(flag)

    if mode == "paper_metric" and bool(resolved["paper_metric_save_masks"]):
        arguments.append("--save-evaluation-arrays")
    if bool(resolved["paper_metric_save_ssim_map"]):
        arguments.append("--save-ssim-map")
    if mode == "visualisation":
        maximum = _optional_visualization_limit(resolved.get("max_visualizations"))
        if maximum is not None:
            arguments.extend(("--max-visualizations", str(maximum)))

    arguments.extend(_trainer_arguments(resolved))
    return arguments


def _load_run_results(run_dir: Path, output_mode: str) -> Dict[str, Any]:
    if output_mode == "json-only":
        return _load_json_object(
            run_dir / "evaluation_results.json",
            label="Stage-2 evaluation results",
        )
    cases = json.loads(
        (run_dir / "metrics" / "per_case_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    summary = _load_json_object(
        run_dir / "metrics" / "summary_metrics.json",
        label="Stage-2 metric summary",
    )
    completed = [record for record in cases if record.get("status") == "completed"]
    return {
        "format": "3dgr_car_stage2_evaluation_v2",
        "summary": summary,
        "matrix": {
            "case_names": [str(record["case_name"]) for record in completed],
            "metric_names": list(stage2_evaluation.METRIC_NAMES),
            "values": [
                [float(record[name]) for name in stage2_evaluation.METRIC_NAMES]
                for record in completed
            ],
        },
        "cases": cases,
    }


def _scalar_csv_rows(records: Sequence[Mapping[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]]]:
    rows = [
        {
            key: value
            for key, value in record.items()
            if value is None or isinstance(value, (str, int, float, bool))
        }
        for record in records
    ]
    fields = list(dict.fromkeys(key for row in rows for key in row))
    return fields, rows


def _write_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    fields, rows = _scalar_csv_rows(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def _write_combined_outputs(
    output_dir: Path,
    resolved: Mapping[str, Any],
    run_results: Mapping[str, Mapping[str, Any]],
) -> None:
    records: List[Dict[str, Any]] = []
    summaries: Dict[str, Any] = {}
    mask_files: List[Dict[str, Any]] = []
    for view_label, result in run_results.items():
        count = int(view_label[1:])
        summaries[view_label] = {
            "eval_num_views": count,
            **dict(result["summary"]),
        }
        selected_indices = next(
            item["view_indices"]
            for item in resolved["effective_view_sweep"]
            if int(item["eval_num_views"]) == count
        )
        for original in result["cases"]:
            record = dict(original)
            record["eval_num_views"] = count
            record["view_label"] = view_label
            record["role"] = "optimized"
            record["selected_view_indices"] = list(selected_indices)
            records.append(record)
            artifact = record.get("evaluation_arrays")
            if artifact is not None:
                mask_files.append(
                    {
                        "case_id": record.get("case_name"),
                        "split": record.get("split"),
                        "eval_num_views": count,
                        "role": "optimized",
                        "path": str(artifact),
                    }
                )

    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    completed = [record for record in records if record.get("status") == "completed"]
    unique_cases = sorted({str(record.get("case_name")) for record in records})
    evaluation_summary = {
        "num_cases": len(unique_cases),
        "num_case_view_evaluations": len(records),
        "num_completed": len(completed),
        "num_failed": len(records) - len(completed),
        "metrics_by_view_count": summaries,
    }
    performance_summary = {
        "schema_version": 1,
        "comparison_condition": {
            "checkpoint": resolved["checkpoint_path"],
            "checkpoint_choice": resolved["checkpoint_choice"],
            "evaluation_mode": resolved["evaluation_mode"],
            "evaluation_split": resolved["eval_split"],
            "eval_num_views": resolved.get("eval_num_views"),
            "effective_view_sweep": resolved["effective_view_sweep"],
            "eval_view_selection": resolved["eval_view_selection"],
            "eval_view_selection_seed": resolved["eval_view_selection_seed"],
            "initialization": "monocular_gcp_first_selected_view",
            "gaussian_optimization_per_case": True,
        },
        "evaluation": evaluation_summary,
        "roles_by_view_count": {
            label: {"optimized": summary}
            for label, summary in summaries.items()
        },
        "per_case_metrics_file": "performance_per_case.json",
        "output_layout": {
            "visualization": (
                "visualization/<views>/cases/<case>"
                if resolved["evaluation_mode"] == "visualisation"
                else None
            ),
            "metrics_by_view_count": "metrics/by_view_count/<views>",
            "paper_metric_per_case": (
                "metrics/paper_metric_per_case.{json,csv}"
                if resolved["evaluation_mode"] == "paper_metric"
                else None
            ),
            "paper_metric_summary": (
                "metrics/paper_metric_summary.json"
                if resolved["evaluation_mode"] == "paper_metric"
                else None
            ),
            "paper_metric_masks": (
                "metrics/by_view_count/<views>/voxel_masks/<case>.npz"
                if resolved["evaluation_mode"] == "paper_metric"
                and resolved["paper_metric_save_masks"]
                else None
            ),
        },
    }
    stage2_evaluation.write_json(output_dir / "performance_per_case.json", records)
    stage2_evaluation.write_json(output_dir / "performance_summary.json", performance_summary)
    stage2_evaluation.write_json(metrics_dir / "per_case_metrics.json", records)
    stage2_evaluation.write_json(metrics_dir / "summary.json", evaluation_summary)
    _write_csv(metrics_dir / "flat_per_case_metrics.csv", records)

    matrix = np.asarray(
        [
            [float(record[name]) for name in stage2_evaluation.METRIC_NAMES]
            for record in completed
        ],
        dtype=np.float64,
    ).reshape(len(completed), len(stage2_evaluation.METRIC_NAMES))
    np.savez_compressed(
        metrics_dir / "metrics_matrix.npz",
        case_names=np.asarray([str(record["case_name"]) for record in completed]),
        eval_num_views=np.asarray(
            [int(record["eval_num_views"]) for record in completed],
            dtype=np.int32,
        ),
        metric_names=np.asarray(stage2_evaluation.METRIC_NAMES),
        values=matrix,
    )

    if resolved["evaluation_mode"] == "paper_metric":
        paper_summary = {
            "schema_version": 1,
            "protocol": {
                "prediction_threshold": resolved.get(
                    "paper_metric_volume_threshold"
                ),
                "prediction_threshold_percentile": resolved.get(
                    "paper_metric_prediction_threshold_percentile"
                ),
                "ground_truth_threshold": resolved.get(
                    "paper_metric_ground_truth_threshold", 0.0
                ),
                "ssim_window_size": resolved.get(
                    "paper_metric_ssim_window_size", 7
                ),
                "normalisation": resolved.get(
                    "paper_metric_normalisation", "clamp"
                ),
                "metric_mask": resolved.get(
                    "paper_metric_mask", "ground-truth"
                ),
                "physical_alignment": resolved.get(
                    "paper_metric_ground_truth_alignment", "auto"
                ),
                "projection_center_offset_reversal": resolved.get(
                    "projection_offset_mode", "auto"
                ),
            },
            "by_view_count": summaries,
        }
        stage2_evaluation.write_json(
            metrics_dir / "paper_metric_per_case.json", records
        )
        stage2_evaluation.write_json(
            metrics_dir / "paper_metric_summary.json", paper_summary
        )
        _write_csv(metrics_dir / "paper_metric_per_case.csv", records)
        voxel_dir = metrics_dir / "voxel_masks"
        voxel_dir.mkdir(parents=True, exist_ok=True)
        stage2_evaluation.write_json(
            voxel_dir / "manifest.json",
            {
                "format": "compressed_npz",
                "saved": bool(resolved["paper_metric_save_masks"]),
                "num_files": len(mask_files),
                "files": mask_files,
            },
        )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a trained Gaussian Centre Predictor and run per-case Gaussian "
            "optimization for a validation/test split from a JSON config."
        )
    )
    parser.add_argument("--config", required=True, help="Evaluation JSON configuration.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional GCP checkpoint path or best/latest override.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output-directory override.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and write the evaluation plan without running CUDA work.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    resolved = resolve_evaluation_config(
        Path(args.config),
        checkpoint_override=args.checkpoint,
        output_dir_override=args.output_dir,
    )
    output_dir = Path(str(resolved["eval_output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    stage2_evaluation.write_json(output_dir / "resolved_config.json", resolved)

    mode = str(resolved["evaluation_mode"])
    plan_runs: List[Dict[str, Any]] = []
    for view in resolved["effective_view_sweep"]:
        count = int(view["eval_num_views"])
        label = f"k{count}"
        run_dir = (
            output_dir / "visualization" / label
            if mode == "visualisation"
            else output_dir / "metrics" / "by_view_count" / label
        )
        stage2_args = build_stage2_arguments(
            resolved,
            run_output_dir=run_dir,
            view_indices=view["view_indices"],
        )
        plan_runs.append(
            {
                "view_label": label,
                "eval_num_views": count,
                "view_indices": view["view_indices"],
                "output_dir": str(run_dir),
                "stage2_arguments": stage2_args,
            }
        )
    plan = {
        "format": "3dgr_car_gcp_evaluation_plan_v1",
        "evaluation_mode": mode,
        "gcp_checkpoint": resolved["checkpoint_path"],
        "initialization_view_rule": "first selected view only",
        "optimization_view_rule": "all selected views",
        "runs": plan_runs,
    }
    stage2_evaluation.write_json(output_dir / "evaluation_plan.json", plan)
    if args.dry_run:
        print(f"Saved dry-run evaluation plan: {output_dir / 'evaluation_plan.json'}")
        return 0

    run_results: Dict[str, Dict[str, Any]] = {}
    failed = False
    for index, run in enumerate(plan_runs, start=1):
        print(
            f"[{index}/{len(plan_runs)}] GCP evaluation {run['view_label']}: "
            f"views={run['view_indices']}",
            flush=True,
        )
        return_code = stage2_evaluation.main(run["stage2_arguments"])
        run_dir = Path(str(run["output_dir"]))
        output_mode = "full" if mode == "visualisation" else "json-only"
        if return_code != 0:
            failed = True
            if not bool(resolved.get("continue_on_error", False)):
                return return_code
        result_path = (
            run_dir / "evaluation_results.json"
            if output_mode == "json-only"
            else run_dir / "metrics" / "per_case_metrics.json"
        )
        if result_path.is_file():
            run_results[str(run["view_label"])] = _load_run_results(
                run_dir, output_mode
            )

    if run_results:
        _write_combined_outputs(output_dir, resolved, run_results)
    final_path = output_dir / "performance_summary.json"
    print(f"Saved GCP evaluation summary: {final_path}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
