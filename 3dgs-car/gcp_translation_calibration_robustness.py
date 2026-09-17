"""Fixed two-view translational calibration stress test for GCP + 3DGS."""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from evaluate_stage2_npz import METRIC_NAMES
from gcp_view_direction_robustness import (
    _baseline_path,
    _case_keys,
    _load_object,
    _metric_means,
    _numeric_comparison,
    _selected_summary,
    _selected_view_indices,
    _validate_baseline,
)
from stage2_translation_projection import translation_vector_mm


TRANSLATION_FAMILIES = ("Y", "XZ", "XYZ")
TRANSLATION_MAGNITUDES_MM = (5.0, 10.0, 20.0)
TRANSLATION_DIAGNOSTIC_FIELDS = (
    "translation_clean_rerender_dice",
    "translation_original_view2_foreground_pixel_ratio",
    "translation_zero_rerender_view2_foreground_pixel_ratio",
    "translation_translated_view2_foreground_pixel_ratio",
    "translation_zero_visible_centerline_fraction",
    "translation_visible_centerline_fraction",
    "translation_zero_visible_vessel_surface_fraction",
    "translation_visible_vessel_surface_fraction",
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _bool(raw: object, *, label: str) -> bool:
    if not isinstance(raw, bool):
        raise ValueError(f"{label} must be boolean.")
    return raw


def resolve_translation_plan(config: Mapping[str, Any]) -> Dict[str, Any]:
    raw = config.get("translational_calibration_robustness", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError(
            "translational_calibration_robustness must be a JSON object."
        )
    supported = {
        "accurate_baseline_summary",
        "require_accurate_baseline",
        "save_mask_npz_files",
        "renderer_num_circle_points",
        "clean_rerender_min_dice",
        "visibility_warning_threshold",
        "fail_below_visibility_threshold",
    }
    unknown = sorted(set(raw) - supported)
    if unknown:
        raise ValueError(
            "Unknown translational_calibration_robustness options: "
            f"{unknown}. The nine positive-direction conditions are fixed by "
            "the protocol."
        )
    circle_points = int(raw.get("renderer_num_circle_points", 120))
    if circle_points < 8:
        raise ValueError("renderer_num_circle_points must be at least 8.")
    clean_minimum = float(raw.get("clean_rerender_min_dice", 0.98))
    if not math.isfinite(clean_minimum) or not 0.0 <= clean_minimum <= 1.0:
        raise ValueError("clean_rerender_min_dice must be in [0,1].")
    visibility_threshold = float(raw.get("visibility_warning_threshold", 0.95))
    if not math.isfinite(visibility_threshold) or not 0.0 <= visibility_threshold <= 1.0:
        raise ValueError("visibility_warning_threshold must be in [0,1].")
    conditions = []
    for family in TRANSLATION_FAMILIES:
        for magnitude in TRANSLATION_MAGNITUDES_MM:
            vector = translation_vector_mm(family, magnitude)
            conditions.append(
                {
                    "family": family,
                    "magnitude_mm": magnitude,
                    "vector_xyz_mm": list(vector),
                }
            )
    return {
        "conditions": conditions,
        "accurate_baseline_summary": raw.get(
            "accurate_baseline_summary", "auto"
        ),
        "require_accurate_baseline": _bool(
            raw.get("require_accurate_baseline", True),
            label=(
                "translational_calibration_robustness."
                "require_accurate_baseline"
            ),
        ),
        "save_mask_npz_files": _bool(
            raw.get("save_mask_npz_files", False),
            label=(
                "translational_calibration_robustness.save_mask_npz_files"
            ),
        ),
        "renderer_num_circle_points": circle_points,
        "clean_rerender_min_dice": clean_minimum,
        "visibility_warning_threshold": visibility_threshold,
        "fail_below_visibility_threshold": _bool(
            raw.get("fail_below_visibility_threshold", False),
            label=(
                "translational_calibration_robustness."
                "fail_below_visibility_threshold"
            ),
        ),
    }


def translation_condition_id(family: str, magnitude_mm: float) -> str:
    value = float(magnitude_mm)
    magnitude_text = (
        str(int(value)) if value.is_integer() else format(value, ".12g").replace(".", "p")
    )
    return f"{str(family).strip().lower()}_{magnitude_text}mm"


def _performance_case_records(
    summary_path: Path,
    performance: Mapping[str, Any],
) -> list[Dict[str, Any]]:
    if performance.get("format") == "3dgr_car_stage2_evaluation_v2":
        raw = performance.get("cases")
        if not isinstance(raw, list):
            raise ValueError(f"Stage-2 result has no cases list: {summary_path}")
        return [dict(item) for item in raw if isinstance(item, Mapping)]
    relative = performance.get("per_case_metrics_file")
    if not relative:
        raise ValueError(
            f"Performance summary does not identify per-case metrics: {summary_path}"
        )
    case_path = Path(str(relative)).expanduser()
    if not case_path.is_absolute():
        case_path = summary_path.parent / case_path
    raw = json.loads(case_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Per-case metrics must be a JSON list: {case_path}")
    return [
        dict(item)
        for item in raw
        if isinstance(item, Mapping)
        and int(item.get("eval_num_views", 2)) == 2
    ]


def _case_key(record: Mapping[str, Any]) -> Tuple[str, str]:
    return (
        str(record.get("case_name", record.get("case_id"))),
        str(record.get("split")),
    )


def _validate_no_baseline_translation(
    path: Path,
    performance: Mapping[str, Any],
) -> None:
    if performance.get("format") == "3dgr_car_stage2_evaluation_v2":
        configuration = performance.get("configuration")
        if isinstance(configuration, Mapping) and configuration.get(
            "view2_translation_mm"
        ) is not None:
            raise ValueError(
                f"Accurate baseline is itself a translated-view run: {path}"
            )
        return
    condition = performance.get("comparison_condition")
    if isinstance(condition, Mapping) and condition.get(
        "view2_translation_mm"
    ) is not None:
        raise ValueError(
            f"Accurate baseline is itself a translated-view run: {path}"
        )


def _case_metric_values(record: Mapping[str, Any]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for key in METRIC_NAMES:
        raw = record.get(key)
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            value = float(raw)
            if math.isfinite(value):
                metrics[key] = value
    return metrics


def _case_role_metrics(record: Mapping[str, Any]) -> Dict[str, Dict[str, float]]:
    roles = {"optimized": _case_metric_values(record)}
    for role, prefix in (
        ("gcp_initialization", "gcp_initial_"),
        ("optimized_minus_gcp_initialization", "optimized_minus_gcp_initial_"),
    ):
        values: Dict[str, float] = {}
        for metric in METRIC_NAMES:
            raw = record.get(f"{prefix}{metric}")
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                value = float(raw)
                if math.isfinite(value):
                    values[metric] = value
        if values:
            roles[role] = values
    return roles


def _performance_role_means(
    performance: Mapping[str, Any],
) -> Dict[str, Dict[str, float]]:
    if performance.get("format") == "3dgr_car_stage2_evaluation_v2":
        selected = _selected_summary(performance)
        roles = {"optimized": _metric_means(selected)}
        raw_selected_roles = selected.get("roles")
        if isinstance(raw_selected_roles, Mapping):
            for role, summary in raw_selected_roles.items():
                if isinstance(summary, Mapping):
                    means = _metric_means(summary)
                    if means:
                        roles[str(role)] = means
        return roles
    raw_roles = performance.get("roles_by_view_count")
    if not isinstance(raw_roles, Mapping):
        return {"optimized": _metric_means(_selected_summary(performance))}
    k2 = raw_roles.get("k2")
    if not isinstance(k2, Mapping):
        return {"optimized": _metric_means(_selected_summary(performance))}
    roles: Dict[str, Dict[str, float]] = {}
    for role, summary in k2.items():
        if isinstance(summary, Mapping):
            means = _metric_means(summary)
            if means:
                roles[str(role)] = means
    roles.setdefault("optimized", _metric_means(_selected_summary(performance)))
    return roles


def _role_comparison(
    current: Mapping[str, Mapping[str, float]],
    baseline: Mapping[str, Mapping[str, float]],
) -> Dict[str, Any]:
    return {
        role: _numeric_comparison(metrics, baseline[role])
        for role, metrics in current.items()
        if role in baseline
    }


def _diagnostic_statistics(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for field in TRANSLATION_DIAGNOSTIC_FIELDS:
        values = np.asarray(
            [
                float(record[field])
                for record in records
                if isinstance(record.get(field), (int, float))
                and math.isfinite(float(record[field]))
            ],
            dtype=np.float64,
        )
        result[field] = {
            "mean": float(np.mean(values)) if values.size else None,
            "minimum": float(np.min(values)) if values.size else None,
            "maximum": float(np.max(values)) if values.size else None,
            "num_finite": int(values.size),
        }
    return result


def _write_per_case_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    rows = []
    for record in records:
        row: Dict[str, Any] = {}
        for key, value in record.items():
            if value is None or isinstance(value, (str, int, float, bool)):
                row[key] = value
            elif key in {
                "translation_vector_xyz_mm",
                "translation_selected_view_indices",
                "translation_nominal_theta_deg",
                "translation_nominal_phi_deg",
                "comparison_to_accurate",
            }:
                row[key] = json.dumps(value, separators=(",", ":"))
        rows.append(row)
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_condition_metrics_csv(
    path: Path,
    results: Sequence[Mapping[str, Any]],
) -> None:
    rows = []
    for result in results:
        vector = result["translation_vector_xyz_mm"]
        comparisons = result.get("comparison_to_accurate") or {}
        for role, metrics in result.get("roles", {}).items():
            if not isinstance(metrics, Mapping):
                continue
            for metric, value in metrics.items():
                if not isinstance(value, (int, float)):
                    continue
                comparison = comparisons.get(role, {}).get(metric, {})
                rows.append(
                    {
                        "condition_id": result["condition_id"],
                        "translation_family": result["translation_family"],
                        "translation_magnitude_mm": result[
                            "translation_magnitude_mm"
                        ],
                        "translation_x_mm": vector[0],
                        "translation_y_mm": vector[1],
                        "translation_z_mm": vector[2],
                        "delta_theta_deg": 0.0,
                        "delta_phi_deg": 0.0,
                        "role": role,
                        "metric": metric,
                        "value": value,
                        "accurate_value": comparison.get("accurate"),
                        "signed_change_from_accurate": comparison.get(
                            "signed_change"
                        ),
                        "relative_change_from_accurate_percent": comparison.get(
                            "relative_change_percent"
                        ),
                    }
                )
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _validate_condition_records(
    records: Sequence[Mapping[str, Any]],
    *,
    selected_views: Sequence[int],
    vector_xyz_mm: Sequence[float],
    clean_minimum: float,
) -> None:
    expected_vector = np.asarray(vector_xyz_mm, dtype=np.float64)
    for record in records:
        if record.get("status") != "completed":
            continue
        if [int(value) for value in record.get("translation_selected_view_indices", [])] != [
            int(value) for value in selected_views
        ]:
            raise ValueError(
                f"Case {_case_key(record)} did not use the configured ordered views."
            )
        recorded_vector = np.asarray(
            record.get("translation_vector_xyz_mm", []), dtype=np.float64
        )
        if recorded_vector.shape != (3,) or not np.allclose(
            recorded_vector, expected_vector, atol=1.0e-6, rtol=0.0
        ):
            raise ValueError(
                f"Case {_case_key(record)} used translation {recorded_vector.tolist()} "
                f"instead of {expected_vector.tolist()}."
            )
        if float(record.get("translation_delta_theta_deg", math.nan)) != 0.0 or float(
            record.get("translation_delta_phi_deg", math.nan)
        ) != 0.0:
            raise ValueError(f"Case {_case_key(record)} changed a camera angle.")
        clean_dice = float(record.get("translation_clean_rerender_dice", math.nan))
        if not math.isfinite(clean_dice) or clean_dice < clean_minimum:
            raise ValueError(
                f"Case {_case_key(record)} failed the clean re-render control."
            )


def run_translation_calibration_robustness(
    *,
    resolved: Mapping[str, Any],
    config_path: Path,
    evaluator_path: Path,
    dry_run: bool = False,
) -> Path:
    """Run or plan the fixed nine-condition positive-translation test."""

    plan = resolve_translation_plan(resolved)
    if resolved.get("reuse_existing", False) is not False:
        raise ValueError(
            "Translational calibration robustness requires reuse_existing=false "
            "so every replaced image is passed through the GCP again."
        )
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
    baseline_roles: Dict[str, Dict[str, float]] = {}
    baseline_cases: Optional[list[Tuple[str, str]]] = None
    baseline_case_records: list[Dict[str, Any]] = []
    baseline_case_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
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
        _validate_no_baseline_translation(
            baseline_path,
            baseline_performance,
        )
        baseline_summary = _selected_summary(baseline_performance)
        baseline_metrics = _metric_means(baseline_summary)
        baseline_roles = _performance_role_means(baseline_performance)
        baseline_cases = _case_keys(baseline_path, baseline_performance)
        baseline_case_records = _performance_case_records(
            baseline_path, baseline_performance
        )
        baseline_case_map = {
            _case_key(record): record
            for record in baseline_case_records
            if record.get("status") == "completed"
        }
    elif plan["require_accurate_baseline"] and not dry_run:
        raise FileNotFoundError(
            "Run the accurate two-view paper_metric evaluation first, or set "
            "translational_calibration_robustness.accurate_baseline_summary. "
            f"Expected: {baseline_path}"
        )

    child_base = dict(resolved)
    for key in (
        "config_path",
        "training_config_path",
        "training_defaults_source",
        "effective_view_sweep",
        "effective_early_stop_checks",
        "translational_calibration_robustness",
    ):
        child_base.pop(key, None)
    child_base["eval_num_views"] = 2
    child_base["eval_view_indices"] = selected_views
    child_base["checkpoint_path"] = str(checkpoint)
    child_base["experiment_dir"] = str(experiment)
    child_base["evaluation_view_directions"] = {
        "accurate": True,
        "theta_change_deg": 0.0,
        "phi_change_deg": 0.0,
    }

    aggregate_path = output_dir / "translation_calibration_robustness_summary.json"
    aggregate: Dict[str, Any] = {
        "schema_version": 1,
        "status": "planned" if dry_run else "running",
        "evaluation_name": (
            "fixed two-view translational calibration robustness evaluation"
        ),
        "qualification": (
            "fixed-positive-direction translational stress test; not "
            "direction-independent translation robustness"
        ),
        "method": "3DGR-CAR GCP initialization plus Gaussian optimization",
        "reported_metric_roles": [
            "gcp_initialization",
            "optimized",
            "optimized_minus_gcp_initialization",
        ],
        "stage_mapping": {
            "coarse_equivalent": "Gaussian centre predictor initialization",
            "refined_equivalent": "per-case Gaussian primitive optimization",
            "note": (
                "Paper metrics are emitted for the GCP-initialized volume and "
                "the optimized reconstruction, together with optimized-minus-"
                "initialization effects when the accurate baseline was produced "
                "by this evaluator version."
            ),
        },
        "checkpoint": str(checkpoint),
        "checkpoint_choice": resolved["checkpoint_choice"],
        "evaluation_split": resolved["eval_split"],
        "eval_num_views": 2,
        "eval_view_indices": selected_views,
        "camera_angle_perturbation": {
            "applied": False,
            "delta_theta_deg": 0.0,
            "delta_phi_deg": 0.0,
            "nominal_angle_encodings_unchanged": True,
        },
        "translation_application": {
            "view_1": "stored accurate image used directly",
            "view_2": "projection-centred artery re-rendered after +delta_t",
            "convention": (
                "artery + delta_t; equivalent to source-detector system or "
                "isocentre - delta_t"
            ),
            "coordinate_system": {
                "+x": "patient left",
                "+y": "patient anterior, away from table",
                "+z": "patient superior, toward head",
            },
            "ground_truth_changed": False,
            "model_weights_changed": False,
            "feature_cache_used": False,
            "image_features_recomputed_after_view_2_replacement": True,
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
                "roles": baseline_roles,
                "metric_statistics": baseline_summary.get("metrics"),
                "timing": baseline_summary.get("timing"),
            }
        ),
        "plan": {
            "families": list(TRANSLATION_FAMILIES),
            "magnitudes_mm": list(TRANSLATION_MAGNITUDES_MM),
            "conditions": plan["conditions"],
            "positive_directions_only": True,
            "configured_magnitude_is_euclidean_norm": True,
            "clean_rerender_min_dice": plan["clean_rerender_min_dice"],
            "visibility_warning_threshold": plan[
                "visibility_warning_threshold"
            ],
            "fail_below_visibility_threshold": plan[
                "fail_below_visibility_threshold"
            ],
            "renderer_num_circle_points": plan[
                "renderer_num_circle_points"
            ],
            "require_accurate_baseline": plan["require_accurate_baseline"],
            "save_mask_npz_files": plan["save_mask_npz_files"],
        },
        "results": [],
    }

    condition_runs = []
    for condition in plan["conditions"]:
        family = str(condition["family"])
        magnitude = float(condition["magnitude_mm"])
        identifier = translation_condition_id(family, magnitude)
        condition_output = conditions_dir / identifier
        child = dict(child_base)
        child["evaluation_mode"] = "paper_metric"
        child["eval_output_dir"] = str(condition_output)
        child["paper_metric_save_masks"] = plan["save_mask_npz_files"]
        child["view2_translation_mm"] = list(condition["vector_xyz_mm"])
        child["translation_renderer_num_circle_points"] = plan[
            "renderer_num_circle_points"
        ]
        child["translation_clean_rerender_min_dice"] = plan[
            "clean_rerender_min_dice"
        ]
        child["translation_visibility_warning_threshold"] = plan[
            "visibility_warning_threshold"
        ]
        child["translation_fail_below_visibility_threshold"] = plan[
            "fail_below_visibility_threshold"
        ]
        child_path = config_dir / f"{identifier}.json"
        _write_json(child_path, child)
        condition_runs.append(
            {
                "condition_id": identifier,
                "translation_family": family,
                "translation_magnitude_mm": magnitude,
                "translation_vector_xyz_mm": list(condition["vector_xyz_mm"]),
                "delta_theta_deg": 0.0,
                "delta_phi_deg": 0.0,
                "config": str(child_path),
                "output_dir": str(condition_output),
            }
        )
    aggregate["condition_runs"] = condition_runs
    _write_json(aggregate_path, aggregate)
    if dry_run:
        return aggregate_path

    expected_cases = baseline_cases
    all_per_case: list[Dict[str, Any]] = []
    for run_index, run in enumerate(condition_runs, start=1):
        identifier = str(run["condition_id"])
        print(
            f"[{run_index}/{len(condition_runs)}] translated view 2: "
            f"{run['translation_family']} {float(run['translation_magnitude_mm']):g} mm, "
            f"vector={run['translation_vector_xyz_mm']}",
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
                f"Translation condition {identifier} failed with exit code "
                f"{completed.returncode}. Partial summary: {aggregate_path}"
            )

        performance_path = Path(str(run["output_dir"])) / "performance_summary.json"
        performance = _load_object(
            performance_path, label=f"Performance summary for {identifier}"
        )
        condition_record = performance.get("comparison_condition")
        if not isinstance(condition_record, Mapping):
            raise ValueError(f"Condition {identifier} lacks comparison_condition.")
        recorded_checkpoint = condition_record.get("checkpoint")
        if recorded_checkpoint and Path(str(recorded_checkpoint)).resolve() != checkpoint:
            raise ValueError(f"Condition {identifier} used a different checkpoint.")
        if str(condition_record.get("evaluation_split")) != str(
            resolved["eval_split"]
        ):
            raise ValueError(
                f"Condition {identifier} used a different evaluation split."
            )
        direction_record = condition_record.get("evaluation_view_directions")
        if isinstance(direction_record, Mapping) and (
            not bool(direction_record.get("accurate", False))
            or float(direction_record.get("theta_change_deg", 0.0)) != 0.0
            or float(direction_record.get("phi_change_deg", 0.0)) != 0.0
        ):
            raise ValueError(f"Condition {identifier} changed a camera angle.")
        configured_vector = np.asarray(
            condition_record.get("view2_translation_mm", []), dtype=np.float64
        )
        if configured_vector.shape != (3,) or not np.allclose(
            configured_vector,
            np.asarray(run["translation_vector_xyz_mm"], dtype=np.float64),
            atol=1.0e-6,
            rtol=0.0,
        ):
            raise ValueError(
                f"Condition {identifier} performance summary records the wrong "
                "view-2 translation."
            )
        if _selected_view_indices(performance) != selected_views:
            raise ValueError(
                f"Condition {identifier} used a different two-view selection."
            )
        current_cases = _case_keys(performance_path, performance)
        if expected_cases is None:
            expected_cases = current_cases
        elif current_cases != expected_cases:
            raise ValueError(
                f"Condition {identifier} evaluated different cases/order from "
                "the accurate baseline."
            )

        case_records = _performance_case_records(performance_path, performance)
        _validate_condition_records(
            case_records,
            selected_views=selected_views,
            vector_xyz_mm=run["translation_vector_xyz_mm"],
            clean_minimum=plan["clean_rerender_min_dice"],
        )
        selected = _selected_summary(performance)
        metrics = _metric_means(selected)
        condition_roles = _performance_role_means(performance)
        completed_cases = [
            record for record in case_records if record.get("status") == "completed"
        ]
        for source in completed_cases:
            per_case = dict(source)
            per_case["condition_id"] = identifier
            per_case["translation_family"] = run["translation_family"]
            per_case["role"] = "optimized"
            baseline_case = baseline_case_map.get(_case_key(source))
            source_roles = _case_role_metrics(source)
            baseline_case_roles = (
                {} if baseline_case is None else _case_role_metrics(baseline_case)
            )
            per_case["comparison_to_accurate"] = (
                None
                if baseline_case is None
                else _role_comparison(source_roles, baseline_case_roles)
            )
            per_case["roles"] = source_roles
            all_per_case.append(per_case)

        result = {
            "condition_id": identifier,
            "translation_family": run["translation_family"],
            "translation_magnitude_mm": run["translation_magnitude_mm"],
            "translation_vector_xyz_mm": run["translation_vector_xyz_mm"],
            "translation_vector_norm_mm": float(
                np.linalg.norm(run["translation_vector_xyz_mm"])
            ),
            "delta_theta_deg": 0.0,
            "delta_phi_deg": 0.0,
            "output_dir": run["output_dir"],
            "performance_summary": str(performance_path),
            "case_count": len(current_cases),
            "completed_case_count": len(completed_cases),
            "metrics": metrics,
            "roles": condition_roles,
            "optimization_effect": condition_roles.get(
                "optimized_minus_gcp_initialization"
            ),
            "metric_statistics": selected.get("metrics"),
            "timing": selected.get("timing"),
            "reprojection_diagnostics": _diagnostic_statistics(completed_cases),
            "cases_below_visibility_warning_threshold": int(
                sum(
                    bool(
                        record.get(
                            "translation_below_visibility_warning_threshold"
                        )
                    )
                    for record in completed_cases
                )
            ),
            "comparison_to_accurate": (
                _role_comparison(condition_roles, baseline_roles)
                if baseline_summary is not None
                else None
            ),
        }
        aggregate["results"].append(result)
        aggregate["per_case_results_file"] = (
            "translation_calibration_robustness_per_case.json"
        )
        _write_json(
            output_dir / "translation_calibration_robustness_per_case.json",
            all_per_case,
        )
        _write_json(aggregate_path, aggregate)

    aggregate["status"] = "complete"
    aggregate["num_conditions"] = len(aggregate["results"])
    aggregate["case_count"] = len(expected_cases or [])
    _write_json(aggregate_path, aggregate)
    _write_json(
        output_dir / "translation_calibration_robustness_per_case.json",
        all_per_case,
    )
    _write_per_case_csv(
        output_dir / "translation_calibration_robustness_per_case.csv",
        all_per_case,
    )
    _write_condition_metrics_csv(
        output_dir / "translation_calibration_robustness_metrics.csv",
        aggregate["results"],
    )
    return aggregate_path
