"""Deterministic two-view direction robustness evaluation for GCP + 3DGS."""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


DEFAULT_AXIS_DEGREES = (2.0, 5.0, 10.0, 15.0)
DEFAULT_COMBINED_DEGREES = (5.0, 10.0)
DEFAULT_VISUALIZATION_CONDITIONS = (
    (-10.0, 0.0),
    (10.0, 0.0),
    (0.0, -10.0),
    (0.0, 10.0),
    (-10.0, -10.0),
    (-10.0, 10.0),
    (10.0, -10.0),
    (10.0, 10.0),
)


def _finite(raw: object, *, label: str) -> float:
    if isinstance(raw, bool):
        raise ValueError(f"{label} must be a finite number.")
    try:
        value = float(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite number.") from error
    if not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number.")
    return value


def _positive_list(raw: object, *, label: str) -> list[float]:
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError(f"{label} must be a non-empty list of positive degrees.")
    result: list[float] = []
    for index, item in enumerate(raw):
        value = _finite(item, label=f"{label}[{index}]")
        if value <= 0.0:
            raise ValueError(f"{label}[{index}] must be positive.")
        if value not in result:
            result.append(value)
    return result


def _angle_pairs(raw: object, *, label: str) -> list[Tuple[float, float]]:
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"{label} must be a list of [theta, phi] pairs.")
    result: list[Tuple[float, float]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"{label}[{index}] must be [theta_deg, phi_deg].")
        pair = (
            _finite(item[0], label=f"{label}[{index}][0]"),
            _finite(item[1], label=f"{label}[{index}][1]"),
        )
        if pair not in result:
            result.append(pair)
    return result


def resolve_robustness_plan(config: Mapping[str, Any]) -> Dict[str, Any]:
    raw = config.get("view_direction_robustness", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError("view_direction_robustness must be a JSON object.")

    if raw.get("changes_deg") is not None:
        conditions = _angle_pairs(
            raw["changes_deg"], label="view_direction_robustness.changes_deg"
        )
    else:
        axis = _positive_list(
            raw.get("axis_degrees", list(DEFAULT_AXIS_DEGREES)),
            label="view_direction_robustness.axis_degrees",
        )
        combined = _positive_list(
            raw.get("combined_degrees", list(DEFAULT_COMBINED_DEGREES)),
            label="view_direction_robustness.combined_degrees",
        )
        conditions = []
        for degrees in axis:
            conditions.extend(
                [
                    (-degrees, 0.0),
                    (degrees, 0.0),
                    (0.0, -degrees),
                    (0.0, degrees),
                ]
            )
        for degrees in combined:
            conditions.extend(
                [
                    (-degrees, -degrees),
                    (-degrees, degrees),
                    (degrees, -degrees),
                    (degrees, degrees),
                ]
            )
    if not conditions:
        raise ValueError("The view-direction robustness plan has no conditions.")
    if (0.0, 0.0) in conditions:
        raise ValueError(
            "The robustness grid cannot contain [0, 0]; use the accurate "
            "paper_metric run as the baseline."
        )

    visualization_raw = raw.get("visualization_conditions_deg")
    visualization = (
        [pair for pair in DEFAULT_VISUALIZATION_CONDITIONS if pair in conditions]
        if visualization_raw is None
        else _angle_pairs(
            visualization_raw,
            label="view_direction_robustness.visualization_conditions_deg",
        )
    )
    missing = [pair for pair in visualization if pair not in conditions]
    if missing:
        raise ValueError(
            "Every visualization condition must occur in changes_deg; "
            f"missing={missing}."
        )
    require_baseline = raw.get("require_accurate_baseline", True)
    if not isinstance(require_baseline, bool):
        raise ValueError(
            "view_direction_robustness.require_accurate_baseline must be boolean."
        )
    save_masks = raw.get("save_mask_npz_files", False)
    if not isinstance(save_masks, bool):
        raise ValueError(
            "view_direction_robustness.save_mask_npz_files must be boolean."
        )
    return {
        "conditions": conditions,
        "visualization_conditions": visualization,
        "accurate_baseline_summary": raw.get("accurate_baseline_summary", "auto"),
        "require_accurate_baseline": require_baseline,
        "save_mask_npz_files": save_masks,
    }


def _format_number(value: float) -> str:
    return format(float(value), ".12g").replace("-", "minus").replace(".", "p")


def condition_id(theta_change_deg: float, phi_change_deg: float) -> str:
    return (
        f"theta_change_{_format_number(theta_change_deg)}deg_"
        f"phi_change_{_format_number(phi_change_deg)}deg"
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _load_object(path: Path, *, label: str) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _baseline_path(
    raw: object,
    *,
    experiment_dir: Path,
    checkpoint_path: Path,
    config_path: Path,
) -> Optional[Path]:
    if raw is None or raw is False:
        return None
    if isinstance(raw, str) and raw.strip().lower() == "auto":
        return (
            experiment_dir
            / "evaluation_paper_metric"
            / checkpoint_path.stem
            / "performance_summary.json"
        )
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    path = path.resolve()
    if path.is_dir():
        candidates = (
            path / "performance_summary.json",
            path / "evaluation_results_k2.json",
            path / "evaluation_results.json",
        )
        selected = next((candidate for candidate in candidates if candidate.is_file()), None)
        if selected is None:
            raise FileNotFoundError(
                "Baseline directory contains none of performance_summary.json, "
                "evaluation_results_k2.json, or evaluation_results.json: "
                f"{path}"
            )
        return selected
    return path


def _is_stage2_evaluation_results(performance: Mapping[str, Any]) -> bool:
    return performance.get("format") == "3dgr_car_stage2_evaluation_v2"


def _stage2_configuration(performance: Mapping[str, Any]) -> Mapping[str, Any]:
    configuration = performance.get("configuration")
    if not isinstance(configuration, Mapping):
        raise ValueError(
            "Stage-2 evaluation baseline has no configuration object."
        )
    return configuration


def _trainer_argument_value(
    arguments: object,
    flag: str,
) -> Optional[str]:
    if not isinstance(arguments, list):
        return None
    for index, raw in enumerate(arguments):
        value = str(raw)
        if value == flag:
            if index + 1 >= len(arguments):
                raise ValueError(f"Baseline training_args ends after {flag}.")
            return str(arguments[index + 1])
        prefix = f"{flag}="
        if value.startswith(prefix):
            return value[len(prefix) :]
    return None


def _selected_summary(performance: Mapping[str, Any]) -> Dict[str, Any]:
    if _is_stage2_evaluation_results(performance):
        summary = performance.get("summary")
        if not isinstance(summary, Mapping):
            raise ValueError("Stage-2 evaluation baseline has no summary object.")
        return dict(summary)
    roles = performance.get("roles_by_view_count")
    if not isinstance(roles, Mapping):
        raise ValueError("Performance summary has no roles_by_view_count object.")
    k2 = roles.get("k2")
    if not isinstance(k2, Mapping) or not isinstance(k2.get("optimized"), Mapping):
        raise ValueError("Performance summary has no k2 optimized metrics.")
    return dict(k2["optimized"])


def _metric_means(summary: Mapping[str, Any]) -> Dict[str, float]:
    metrics = summary.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("The k2 optimized summary has no metrics object.")
    means: Dict[str, float] = {}
    for metric, statistics in metrics.items():
        if not isinstance(statistics, Mapping):
            continue
        value = statistics.get("mean")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            means[str(metric)] = float(value)
    return means


def _selected_view_indices(performance: Mapping[str, Any]) -> list[int]:
    if _is_stage2_evaluation_results(performance):
        raw = _stage2_configuration(performance).get("view_indices")
        if not isinstance(raw, list) or len(raw) != 2:
            raise ValueError(
                "Stage-2 evaluation baseline is not a two-view result."
            )
        return [int(index) for index in raw]
    condition = performance.get("comparison_condition")
    if not isinstance(condition, Mapping):
        raise ValueError("Performance summary lacks comparison_condition.")
    sweep = condition.get("effective_view_sweep")
    if not isinstance(sweep, list):
        raise ValueError("Performance summary lacks effective_view_sweep.")
    for item in sweep:
        if isinstance(item, Mapping) and int(item.get("eval_num_views", -1)) == 2:
            raw = item.get("view_indices")
            if isinstance(raw, list):
                return [int(index) for index in raw]
    raise ValueError("Performance summary has no two-view selection.")


def _case_keys(summary_path: Path, performance: Mapping[str, Any]) -> list[Tuple[str, str]]:
    if _is_stage2_evaluation_results(performance):
        value = performance.get("cases")
        if not isinstance(value, list):
            raise ValueError(
                f"Stage-2 baseline cases must be a JSON list: {summary_path}"
            )
        selected = [item for item in value if isinstance(item, Mapping)]
        return [
            (
                str(item.get("case_name", item.get("case_id"))),
                str(item.get("split")),
            )
            for item in selected
        ]
    raw = performance.get("per_case_metrics_file")
    if not raw:
        raise ValueError("Performance summary does not identify per-case metrics.")
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = summary_path.parent / path
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Per-case performance must be a JSON list: {path}")
    selected = [
        item
        for item in value
        if isinstance(item, Mapping) and int(item.get("eval_num_views", -1)) == 2
    ]
    return [
        (str(item.get("case_name", item.get("case_id"))), str(item.get("split")))
        for item in selected
    ]


def _validate_baseline(
    path: Path,
    performance: Mapping[str, Any],
    *,
    checkpoint_path: Path,
    split: str,
    view_indices: Sequence[int],
) -> None:
    if _is_stage2_evaluation_results(performance):
        configuration = _stage2_configuration(performance)
        training_args = configuration.get("training_args")
        recorded_checkpoint = _trainer_argument_value(
            training_args, "--gcp-checkpoint"
        )
        if recorded_checkpoint is None:
            raise ValueError(
                "Stage-2 baseline training_args does not record --gcp-checkpoint."
            )
        if Path(recorded_checkpoint).expanduser().resolve() != checkpoint_path:
            raise ValueError(
                "Baseline checkpoint differs from robustness checkpoint: "
                f"{recorded_checkpoint} != {checkpoint_path}."
            )
        recorded_split = configuration.get(
            "split", performance.get("summary", {}).get("split")
        )
        if str(recorded_split) != str(split):
            raise ValueError("Baseline and robustness evaluation splits differ.")
        if _selected_view_indices(performance) != [
            int(index) for index in view_indices
        ]:
            raise ValueError("Baseline and robustness two-view selections differ.")
        theta_change = _trainer_argument_value(
            training_args, "--view-direction-theta-change-deg"
        )
        phi_change = _trainer_argument_value(
            training_args, "--view-direction-phi-change-deg"
        )
        if float(theta_change or 0.0) != 0.0 or float(phi_change or 0.0) != 0.0:
            raise ValueError(f"Baseline is not an accurate-view run: {path}")
        return

    condition = performance.get("comparison_condition")
    if not isinstance(condition, Mapping):
        raise ValueError(f"Accurate baseline lacks comparison_condition: {path}")
    recorded_checkpoint = condition.get("checkpoint")
    if recorded_checkpoint and Path(str(recorded_checkpoint)).resolve() != checkpoint_path:
        raise ValueError(
            "Baseline checkpoint differs from robustness checkpoint: "
            f"{recorded_checkpoint} != {checkpoint_path}."
        )
    if str(condition.get("evaluation_split")) != str(split):
        raise ValueError("Baseline and robustness evaluation splits differ.")
    if _selected_view_indices(performance) != [int(index) for index in view_indices]:
        raise ValueError("Baseline and robustness two-view selections differ.")
    view_options = condition.get("evaluation_view_directions")
    if isinstance(view_options, Mapping) and not bool(view_options.get("accurate", False)):
        raise ValueError(f"Baseline is not an accurate-view run: {path}")
    if condition.get("view_directions_accurate") is False:
        raise ValueError(f"Baseline is not an accurate-view run: {path}")


def _numeric_comparison(
    current: Mapping[str, float], baseline: Mapping[str, float]
) -> Dict[str, Dict[str, Optional[float]]]:
    result: Dict[str, Dict[str, Optional[float]]] = {}
    for metric in sorted(set(current).intersection(baseline)):
        inaccurate = float(current[metric])
        accurate = float(baseline[metric])
        change = inaccurate - accurate
        result[metric] = {
            "accurate": accurate,
            "inaccurate": inaccurate,
            "signed_change": change,
            "relative_change_percent": (
                None if accurate == 0.0 else 100.0 * change / abs(accurate)
            ),
        }
    return result


def fit_quadratic_response_surface(
    samples: Sequence[Tuple[float, float, float]],
) -> Optional[Dict[str, Any]]:
    finite = [
        (float(theta), float(phi), float(value))
        for theta, phi, value in samples
        if all(math.isfinite(float(item)) for item in (theta, phi, value))
    ]
    if len(finite) < 6:
        return None
    design = np.asarray(
        [
            [1.0, theta, phi, theta * theta, theta * phi, phi * phi]
            for theta, phi, _ in finite
        ],
        dtype=np.float64,
    )
    if int(np.linalg.matrix_rank(design)) < 6:
        return None
    observed = np.asarray([value for _, _, value in finite], dtype=np.float64)
    coefficients, _, _, _ = np.linalg.lstsq(design, observed, rcond=None)
    residual = observed - design @ coefficients
    rss = float(np.sum(residual * residual))
    centered = observed - float(np.mean(observed))
    tss = float(np.sum(centered * centered))
    names = (
        "intercept",
        "theta_linear",
        "phi_linear",
        "theta_squared",
        "theta_phi_interaction",
        "phi_squared",
    )
    return {
        "model": (
            "metric = intercept + theta_linear*theta + phi_linear*phi + "
            "theta_squared*theta^2 + theta_phi_interaction*theta*phi + "
            "phi_squared*phi^2"
        ),
        "coefficients": {
            name: float(value) for name, value in zip(names, coefficients)
        },
        "num_samples": len(finite),
        "r_squared": (
            1.0 - rss / tss if tss > 0.0 else 1.0 if rss == 0.0 else None
        ),
        "rmse": float(math.sqrt(rss / len(finite))),
        "theta_range_deg": [min(row[0] for row in finite), max(row[0] for row in finite)],
        "phi_range_deg": [min(row[1] for row in finite), max(row[1] for row in finite)],
    }


def _response_surfaces(
    results: Sequence[Mapping[str, Any]], baseline: Mapping[str, float]
) -> Dict[str, Any]:
    models: Dict[str, Any] = {}
    names = set(baseline)
    for result in results:
        names.update(result.get("metrics", {}))
    for metric in sorted(names):
        samples: list[Tuple[float, float, float]] = []
        if metric in baseline:
            samples.append((0.0, 0.0, float(baseline[metric])))
        for result in results:
            value = result.get("metrics", {}).get(metric)
            if isinstance(value, (int, float)):
                samples.append(
                    (
                        float(result["theta_change_deg"]),
                        float(result["phi_change_deg"]),
                        float(value),
                    )
                )
        fitted = fit_quadratic_response_surface(samples)
        if fitted is not None:
            models[metric] = fitted
    return {
        "roles": {"optimized": models},
        "interpretation": (
            "Descriptive fixed-grid response surfaces; do not extrapolate "
            "outside the recorded theta/phi ranges."
        ),
    }


def _write_flat_csv(path: Path, results: Sequence[Mapping[str, Any]]) -> None:
    rows = []
    for result in results:
        comparison = (result.get("comparison_to_accurate") or {}).get(
            "optimized", {}
        )
        for metric, value in result.get("metrics", {}).items():
            compared = comparison.get(metric, {})
            rows.append(
                {
                    "condition_id": result["condition_id"],
                    "theta_change_deg": result["theta_change_deg"],
                    "phi_change_deg": result["phi_change_deg"],
                    "angular_offset_magnitude_deg": result[
                        "angular_offset_magnitude_deg"
                    ],
                    "evaluation_mode": result["evaluation_mode"],
                    "role": "optimized",
                    "metric": metric,
                    "value": value,
                    "accurate_value": compared.get("accurate"),
                    "signed_change_from_accurate": compared.get("signed_change"),
                    "relative_change_from_accurate_percent": compared.get(
                        "relative_change_percent"
                    ),
                }
            )
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_view_direction_robustness(
    *,
    resolved: Mapping[str, Any],
    config_path: Path,
    evaluator_path: Path,
    dry_run: bool = False,
) -> Path:
    """Run or plan the paired signed-angle sweep."""

    plan = resolve_robustness_plan(resolved)
    output_dir = Path(str(resolved["eval_output_dir"])).resolve()
    config_dir = output_dir / "run_configs"
    conditions_dir = output_dir / "conditions"
    config_dir.mkdir(parents=True, exist_ok=True)
    conditions_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(str(resolved["checkpoint_path"])).resolve()
    experiment = Path(str(resolved["experiment_dir"])).resolve()
    selected_views = [
        int(index) for index in resolved["effective_view_sweep"][0]["view_indices"]
    ]
    baseline_path = _baseline_path(
        plan["accurate_baseline_summary"],
        experiment_dir=experiment,
        checkpoint_path=checkpoint,
        config_path=config_path,
    )
    baseline_performance: Optional[Dict[str, Any]] = None
    baseline_summary: Optional[Dict[str, Any]] = None
    baseline_metrics: Dict[str, float] = {}
    baseline_cases: Optional[list[Tuple[str, str]]] = None
    if baseline_path is not None and baseline_path.is_file():
        baseline_performance = _load_object(
            baseline_path, label="Accurate performance summary"
        )
        _validate_baseline(
            baseline_path,
            baseline_performance,
            checkpoint_path=checkpoint,
            split=str(resolved["eval_split"]),
            view_indices=selected_views,
        )
        baseline_summary = _selected_summary(baseline_performance)
        baseline_metrics = _metric_means(baseline_summary)
        baseline_cases = _case_keys(baseline_path, baseline_performance)
    elif plan["require_accurate_baseline"] and not dry_run:
        raise FileNotFoundError(
            "Run the accurate two-view paper_metric evaluation first, or set "
            "view_direction_robustness.accurate_baseline_summary. Expected: "
            f"{baseline_path}"
        )

    child_base = dict(resolved)
    for key in (
        "config_path",
        "training_config_path",
        "training_defaults_source",
        "effective_view_sweep",
        "effective_early_stop_checks",
        "view_direction_robustness",
    ):
        child_base.pop(key, None)
    child_base["eval_num_views"] = 2
    child_base["eval_view_indices"] = selected_views
    child_base["checkpoint_path"] = str(checkpoint)
    child_base["experiment_dir"] = str(experiment)

    aggregate_path = output_dir / "view_direction_robustness_summary.json"
    aggregate: Dict[str, Any] = {
        "schema_version": 1,
        "status": "planned" if dry_run else "running",
        "method": "3DGR-CAR GCP initialization plus Gaussian optimization",
        "checkpoint": str(checkpoint),
        "checkpoint_choice": resolved["checkpoint_choice"],
        "evaluation_split": resolved["eval_split"],
        "eval_num_views": 2,
        "eval_view_indices": selected_views,
        "error_application": {
            "distribution": "fixed_per_selected_view",
            "same_change_applied_to_both_input_views": True,
            "gcp_lifting_uses_inaccurate_first_view_geometry": True,
            "gaussian_optimization_uses_inaccurate_two_view_geometry": True,
            "projection_images_changed": False,
            "ground_truth_changed": False,
            "novel_view_geometry_changed": False,
        },
        "accurate_baseline_summary": (
            None if baseline_path is None else str(baseline_path)
        ),
        "accurate_baseline": (
            None
            if baseline_summary is None
            else {
                "case_count": len(baseline_cases or []),
                "metrics": baseline_metrics,
                "roles": {"optimized": baseline_metrics},
                "metric_statistics": baseline_summary.get("metrics"),
                "timing": baseline_summary.get("timing"),
            }
        ),
        "plan": {
            "error_model": "fixed_signed_theta_phi_grid",
            "conditions": [list(pair) for pair in plan["conditions"]],
            "visualization_conditions": [
                list(pair) for pair in plan["visualization_conditions"]
            ],
            "require_accurate_baseline": plan["require_accurate_baseline"],
            "save_mask_npz_files": plan["save_mask_npz_files"],
        },
        "results": [],
    }
    condition_plan = []
    for theta_change, phi_change in plan["conditions"]:
        identifier = condition_id(theta_change, phi_change)
        condition_output = conditions_dir / identifier
        condition_mode = (
            "visualisation"
            if (theta_change, phi_change) in plan["visualization_conditions"]
            else "paper_metric"
        )
        child = dict(child_base)
        child["evaluation_mode"] = condition_mode
        child["eval_output_dir"] = str(condition_output)
        child["paper_metric_save_masks"] = plan["save_mask_npz_files"]
        child["evaluation_view_directions"] = {
            "accurate": False,
            "theta_change_deg": theta_change,
            "phi_change_deg": phi_change,
        }
        child_path = config_dir / f"{identifier}.json"
        _write_json(child_path, child)
        condition_plan.append(
            {
                "condition_id": identifier,
                "theta_change_deg": theta_change,
                "phi_change_deg": phi_change,
                "evaluation_mode": condition_mode,
                "config": str(child_path),
                "output_dir": str(condition_output),
            }
        )
    aggregate["condition_runs"] = condition_plan
    _write_json(aggregate_path, aggregate)
    if dry_run:
        return aggregate_path

    expected_cases = baseline_cases
    for index, run in enumerate(condition_plan, start=1):
        identifier = str(run["condition_id"])
        print(
            f"[{index}/{len(condition_plan)}] inaccurate view directions: "
            f"theta={float(run['theta_change_deg']):+.3f} deg, "
            f"phi={float(run['phi_change_deg']):+.3f} deg",
            flush=True,
        )
        completed = subprocess.run(
            [
                sys.executable,
                str(evaluator_path),
                "--config",
                str(run["config"]),
            ],
            cwd=evaluator_path.parent,
            check=False,
        )
        if completed.returncode != 0:
            aggregate["status"] = "failed"
            aggregate["failed_condition"] = identifier
            aggregate["failed_return_code"] = int(completed.returncode)
            _write_json(aggregate_path, aggregate)
            raise RuntimeError(
                f"View-direction condition {identifier} failed with exit code "
                f"{completed.returncode}. Partial summary: {aggregate_path}"
            )
        performance_path = Path(str(run["output_dir"])) / "performance_summary.json"
        performance = _load_object(
            performance_path, label=f"Performance summary for {identifier}"
        )
        comparison_condition = performance.get("comparison_condition")
        if not isinstance(comparison_condition, Mapping):
            raise ValueError(
                f"Condition {identifier} lacks comparison_condition."
            )
        recorded_checkpoint = comparison_condition.get("checkpoint")
        if (
            recorded_checkpoint
            and Path(str(recorded_checkpoint)).resolve() != checkpoint
        ):
            raise ValueError(
                f"Condition {identifier} used a different checkpoint."
            )
        if str(comparison_condition.get("evaluation_split")) != str(
            resolved["eval_split"]
        ):
            raise ValueError(
                f"Condition {identifier} used a different evaluation split."
            )
        if _selected_view_indices(performance) != selected_views:
            raise ValueError(
                f"Condition {identifier} used a different two-view selection."
            )
        current_cases = _case_keys(performance_path, performance)
        if expected_cases is None:
            expected_cases = current_cases
        elif current_cases != expected_cases:
            aggregate["status"] = "failed"
            aggregate["failed_condition"] = identifier
            aggregate["failure_reason"] = "Condition evaluated different cases/order."
            _write_json(aggregate_path, aggregate)
            raise ValueError(
                f"Condition {identifier} evaluated different cases/order from "
                "the accurate baseline."
            )
        selected = _selected_summary(performance)
        metrics = _metric_means(selected)
        result = {
            "condition_id": identifier,
            "theta_change_deg": float(run["theta_change_deg"]),
            "phi_change_deg": float(run["phi_change_deg"]),
            "angular_offset_magnitude_deg": math.hypot(
                float(run["theta_change_deg"]), float(run["phi_change_deg"])
            ),
            "evaluation_mode": run["evaluation_mode"],
            "output_dir": run["output_dir"],
            "performance_summary": str(performance_path),
            "case_count": len(current_cases),
            "metrics": metrics,
            "roles": {"optimized": metrics},
            "metric_statistics": selected.get("metrics"),
            "timing": selected.get("timing"),
            "comparison_to_accurate": (
                {
                    "optimized": _numeric_comparison(
                        metrics, baseline_metrics
                    )
                }
                if baseline_summary is not None
                else None
            ),
        }
        aggregate["results"].append(result)
        _write_json(aggregate_path, aggregate)

    aggregate["status"] = "complete"
    aggregate["num_conditions"] = len(aggregate["results"])
    aggregate["case_count"] = len(expected_cases or [])
    aggregate["quadratic_response_surfaces"] = _response_surfaces(
        aggregate["results"], baseline_metrics
    )
    _write_json(aggregate_path, aggregate)
    _write_flat_csv(
        output_dir / "view_direction_robustness_metrics.csv",
        aggregate["results"],
    )
    return aggregate_path
