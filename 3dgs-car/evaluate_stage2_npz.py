#!/usr/bin/env python3
"""Run and evaluate Stage-2 NPZ reconstructions for one dataset split.

The script resolves cases from a split JSON, runs ``train_stage2_npz.py`` once
per case, and compares each ``reconstructed_volume_zyx.npy`` with a matching
ground-truth volume NPZ.  Evaluation itself is NumPy-only so that existing
reconstructions can be scored on a machine without CUDA by passing
``--skip-reconstruction``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from volume_gif import (
    DEFAULT_VOLUME_GIF_POSITIVE_PERCENTILE,
    resolve_volume_isovalue,
)


VOLUME_KEY_CANDIDATES = (
    "vol",
    "volume",
    "voxel",
    "voxels",
    "gt_volume",
    "ground_truth",
    "segmentation",
    "label",
    "mask",
    "gt",
    "arr_0",
)
CASE_ID_KEYS = (
    "path",
    "file",
    "source_path",
    "case_name",
    "sample_name",
    "case_id",
    "case_number",
    "case",
    "id",
    "number",
    "name",
    "filename",
)
METRIC_NAMES = (
    "masked_dice_3d",
    "mse_3d",
    "ssim_3d",
    "masked_mse",
    "masked_mae",
    "masked_psnr",
    "masked_ssim_3d",
)
TIMING_NAMES = (
    "case_wall_time_seconds",
    "reconstruction_wall_time_seconds",
    "optimization_elapsed_seconds",
    "metrics_wall_time_seconds",
)


def _normalise_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _canonical_case_reference(value: object) -> str:
    """Convert parametric feature paths to Stage-2 projection case names."""

    text = str(value).strip()
    if not text:
        return text
    if isinstance(value, (int, np.integer)) or text.isdigit():
        return text

    parts = [part for part in text.replace("\\", "/").split("/") if part]
    stem = parts[-1].rsplit(".", 1)[0] if parts else text
    direct = re.fullmatch(r"(?i)(lca|rca)[_-]?0*([0-9]+)", stem)
    if direct is not None:
        vessel, case_number = direct.groups()
        return f"{vessel.lower()}_{int(case_number):04d}"

    for index, part in enumerate(parts[:-1]):
        vessel = part.lower()
        if vessel not in {"lca", "rca"}:
            continue
        for child in parts[index + 1 : -1]:
            if child.isdigit():
                return f"{vessel}_{int(child):04d}"
    return text


def _optional_nonnegative_integer(value: str) -> Optional[int]:
    """Parse an optional case/artifact limit accepted by the CLI."""

    if str(value).strip().lower() == "all":
        return None
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected a non-negative integer or 'all'"
        ) from error
    if parsed < 0:
        raise argparse.ArgumentTypeError(
            "expected a non-negative integer or 'all'"
        )
    return parsed


def select_case_references(
    references: Sequence[str],
    requested_case_ids: Optional[Sequence[str]],
    num_cases: Optional[int],
) -> List[str]:
    """Apply explicit case selection and a deterministic prefix limit."""

    selected = list(references)
    if requested_case_ids is not None:
        requested = [str(value).strip() for value in requested_case_ids]
        if not requested or any(not value for value in requested):
            raise ValueError("--eval-case-ids must contain non-empty identifiers.")
        duplicate_requests = sorted(
            value for value in set(requested) if requested.count(value) > 1
        )
        if duplicate_requests:
            raise ValueError(
                f"--eval-case-ids contains duplicates: {duplicate_requests}."
            )

        exact = {reference: reference for reference in selected}
        normalised: Dict[str, List[str]] = {}
        for reference in selected:
            normalised.setdefault(_normalise_key(reference), []).append(reference)
        resolved: List[str] = []
        missing: List[str] = []
        for request in requested:
            if request in exact:
                resolved.append(exact[request])
                continue
            matches = normalised.get(_normalise_key(request), [])
            if len(matches) == 1:
                resolved.append(matches[0])
            elif len(matches) > 1:
                raise ValueError(
                    f"Evaluation case identifier {request!r} is ambiguous: {matches}."
                )
            else:
                request_number = _numeric_case_id(request)
                numeric_matches = (
                    []
                    if request_number is None
                    else [
                        reference
                        for reference in selected
                        if _numeric_case_id(reference) == request_number
                    ]
                )
                if len(numeric_matches) == 1:
                    resolved.append(numeric_matches[0])
                elif len(numeric_matches) > 1:
                    raise ValueError(
                        f"Evaluation case identifier {request!r} is ambiguous: "
                        f"{numeric_matches}. Use a full case name."
                    )
                else:
                    missing.append(request)
        if missing:
            raise ValueError(
                f"Evaluation cases are not present in the selected split: {missing}."
            )
        selected = resolved

    if num_cases is not None:
        selected = selected[: int(num_cases)]
    if not selected:
        raise ValueError("Evaluation case selection is empty.")
    return selected


def _split_aliases(split: str) -> set[str]:
    normalised = _normalise_key(split)
    if normalised in {"val", "valid", "validation", "dev"}:
        return {"val", "valid", "validation", "dev"}
    if normalised in {"train", "training"}:
        return {"train", "training"}
    if normalised in {"test", "testing"}:
        return {"test", "testing"}
    return {normalised}


def _find_split_value(document: object, split: str) -> object:
    aliases = _split_aliases(split)
    split_suffixes = {"", "cases", "caselist", "caseids", "casenumbers", "ids", "samples"}
    queue: deque[object] = deque([document])
    while queue:
        value = queue.popleft()
        if isinstance(value, Mapping):
            for key, child in value.items():
                normalised_key = _normalise_key(key)
                if any(
                    normalised_key == alias + suffix
                    for alias in aliases
                    for suffix in split_suffixes
                ):
                    return child
            queue.extend(value.values())
        elif isinstance(value, list):
            queue.extend(value)
    raise KeyError(
        f"Could not find split {split!r} in the JSON. Accepted aliases: "
        f"{sorted(aliases)}."
    )


def _case_reference_from_record(record: Mapping[object, object]) -> Optional[str]:
    normalised_items = {_normalise_key(key): value for key, value in record.items()}
    for key in CASE_ID_KEYS:
        value = normalised_items.get(_normalise_key(key))
        if value is not None and not isinstance(value, (dict, list)):
            return _canonical_case_reference(value)
    return None


def _case_references(value: object) -> List[str]:
    if isinstance(value, (str, int, np.integer)):
        return [str(value)]
    if isinstance(value, list):
        references: List[str] = []
        for item in value:
            if isinstance(item, Mapping):
                reference = _case_reference_from_record(item)
                if reference is None:
                    raise ValueError(
                        f"Split record has no recognised case identifier: {item!r}."
                    )
                references.append(reference)
            elif isinstance(item, (str, int, np.integer)):
                references.append(_canonical_case_reference(item))
            else:
                raise TypeError(f"Unsupported split entry: {item!r}.")
        return references
    if isinstance(value, Mapping):
        for container_key in (
            "cases",
            "case_names",
            "case_ids",
            "case_numbers",
            "items",
            "samples",
        ):
            for key, child in value.items():
                if _normalise_key(key) == _normalise_key(container_key):
                    return _case_references(child)
        one_record = _case_reference_from_record(value)
        if one_record is not None:
            return [one_record]
        # Also accept {"case_1": true, "case_2": true} split maps.
        return [str(key) for key, enabled in value.items() if bool(enabled)]
    raise TypeError(f"Split must contain a list or mapping of cases, got {type(value).__name__}.")


def load_split_case_references(path: Path, split: str) -> List[str]:
    with path.open("r", encoding="utf-8") as stream:
        document = json.load(stream)

    def validated_references(split_name: str) -> List[str]:
        references = [
            reference.strip()
            for reference in _case_references(
                _find_split_value(document, split_name)
            )
        ]
        if not references:
            raise ValueError(f"Split {split_name!r} contains no cases.")
        if any(not reference for reference in references):
            raise ValueError(
                f"Split {split_name!r} contains an empty case identifier."
            )
        duplicates = sorted(
            {
                reference
                for reference in references
                if references.count(reference) > 1
            }
        )
        if duplicates:
            raise ValueError(
                f"Split {split_name!r} contains duplicate cases: {duplicates}."
            )
        return references

    if _normalise_key(split) in {"valtest", "validationtest"}:
        combined: List[str] = []
        seen: set[str] = set()
        for split_name in ("val", "test"):
            for reference in validated_references(split_name):
                if reference not in seen:
                    combined.append(reference)
                    seen.add(reference)
        if not combined:
            raise ValueError("Combined split 'val_test' contains no cases.")
        return combined

    return validated_references(split)


def _numeric_case_id(value: str) -> Optional[int]:
    match = re.search(r"(\d+)$", Path(value).stem)
    return None if match is None else int(match.group(1))


class NpzIndex:
    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        if not self.root.is_dir():
            raise NotADirectoryError(f"NPZ directory does not exist: {self.root}")
        self.paths = sorted(path.resolve() for path in self.root.rglob("*.npz"))
        if not self.paths:
            raise FileNotFoundError(f"No .npz files found under {self.root}.")
        self.by_stem: Dict[str, List[Path]] = {}
        self.by_number: Dict[int, List[Path]] = {}
        for path in self.paths:
            self.by_stem.setdefault(path.stem.lower(), []).append(path)
            number = _numeric_case_id(path.stem)
            if number is not None:
                self.by_number.setdefault(number, []).append(path)

    def resolve(self, references: Iterable[str], kind: str) -> Path:
        references = tuple(str(reference) for reference in references)
        for reference in references:
            relative = Path(reference).expanduser()
            candidates = [relative]
            if relative.suffix.lower() != ".npz":
                candidates.append(relative.with_suffix(".npz"))
            for candidate in candidates:
                path = candidate.resolve() if candidate.is_absolute() else (self.root / candidate).resolve()
                if path.is_file() and (path == self.root or self.root in path.parents):
                    return path

        for reference in references:
            matches = self.by_stem.get(Path(reference).stem.lower(), [])
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise ValueError(f"Ambiguous {kind} case {reference!r}: {matches}.")

        for reference in references:
            number = _numeric_case_id(reference)
            if number is None:
                continue
            matches = self.by_number.get(number, [])
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise ValueError(
                    f"Case number {number} matches multiple {kind} NPZs: {matches}. "
                    "Use full case names in the split JSON."
                )
        raise FileNotFoundError(
            f"Could not match {kind} NPZ for any of {references!r} under {self.root}."
        )


def projection_case_identity(path: Path) -> Tuple[str, Optional[str], Optional[str]]:
    sample_name = path.stem
    vessel_type: Optional[str] = None
    case_id: Optional[str] = None
    try:
        with np.load(path, allow_pickle=False) as archive:
            if "sample_name" in archive.files:
                sample_name = str(
                    np.asarray(archive["sample_name"]).reshape(()).item()
                )
            if "vessel_type" in archive.files:
                vessel_type = str(
                    np.asarray(archive["vessel_type"]).reshape(()).item()
                ).strip().lower()
            if "case_id" in archive.files:
                case_id = str(
                    np.asarray(archive["case_id"]).reshape(()).item()
                ).strip()
    except (OSError, ValueError):
        pass
    match = re.match(r"^([a-zA-Z]+)[_-]?(\d+)$", sample_name)
    if match is not None:
        if not vessel_type:
            vessel_type = match.group(1).lower()
        if not case_id:
            case_id = str(int(match.group(2)))
    return sample_name, vessel_type or None, case_id or None


def projection_sample_name(path: Path) -> str:
    return projection_case_identity(path)[0]


def ground_truth_case_references(
    projection_path: Path,
    sample_name: str,
    split_reference: str,
) -> Tuple[List[str], Optional[str], Optional[str]]:
    _, vessel_type, case_id = projection_case_identity(projection_path)
    references: List[str] = []
    if vessel_type and case_id:
        references.append(str(Path(vessel_type) / case_id))
        numeric_case_id = _numeric_case_id(case_id)
        if numeric_case_id is not None:
            references.append(str(Path(vessel_type) / str(numeric_case_id)))
    if case_id:
        references.append(case_id)
    references.extend((sample_name, projection_path.stem, split_reference))
    unique_references = list(dict.fromkeys(references))
    return unique_references, vessel_type, case_id


def _squeeze_volume(array: np.ndarray, source: str) -> np.ndarray:
    volume = np.asarray(array)
    if volume.dtype.kind not in "biuf":
        raise TypeError(f"{source} must be numeric, got dtype {volume.dtype}.")
    volume = np.squeeze(volume)
    if volume.ndim != 3:
        raise ValueError(f"{source} must reduce to a 3D array, got shape {volume.shape}.")
    volume = np.asarray(volume, dtype=np.float32)
    if not np.isfinite(volume).all():
        raise ValueError(f"{source} contains NaN or infinity.")
    return np.ascontiguousarray(volume)


def _select_npz_volume_key(
    archive: np.lib.npyio.NpzFile,
    path: Path,
    key: Optional[str],
) -> str:
    if key is not None:
        if key not in archive.files:
            raise KeyError(f"{path} has no key {key!r}; available keys: {archive.files}.")
        return key
    key_lookup = {name.lower(): name for name in archive.files}
    selected_key = next(
        (
            key_lookup[candidate.lower()]
            for candidate in VOLUME_KEY_CANDIDATES
            if candidate.lower() in key_lookup
        ),
        None,
    )
    if selected_key is not None:
        return selected_key
    compatible = []
    for name in archive.files:
        candidate = np.asarray(archive[name])
        if candidate.dtype.kind in "biuf" and np.squeeze(candidate).ndim == 3:
            compatible.append(name)
    if len(compatible) != 1:
        raise KeyError(
            f"Could not identify one volume in {path}; 3D candidates are "
            f"{compatible}. Pass --ground-truth-key."
        )
    return compatible[0]


def _effective_ground_truth_axis_order(requested: str, selected_key: str) -> str:
    if requested != "auto":
        return requested
    # The supplied ImageCAS artery-mask converter stores nibabel-style `vol`
    # arrays as [x,y,z]. Other historical GT arrays in this repository are ZYX.
    return "xyz" if selected_key.lower() == "vol" else "zyx"


def load_ground_truth_volume(
    path: Path,
    key: Optional[str],
    axis_order: str,
    spacing_key: str,
    spacing_units: str,
) -> Tuple[np.ndarray, str, str, Optional[np.ndarray]]:
    """Load GT as ZYX and return physical XYZ spacing in metres when present."""
    with np.load(path, allow_pickle=False) as archive:
        selected_key = _select_npz_volume_key(archive, path, key)
        effective_axis_order = _effective_ground_truth_axis_order(
            axis_order,
            selected_key,
        )
        volume = _squeeze_volume(archive[selected_key], f"{path}:{selected_key}")
        spacing_stored = (
            np.asarray(archive[spacing_key], dtype=np.float64).reshape(-1)
            if spacing_key in archive.files
            else None
        )
    if effective_axis_order == "xyz":
        volume = np.transpose(volume, (2, 1, 0))
    if spacing_stored is None:
        spacing_xyz_m = None
    else:
        if spacing_stored.shape != (3,) or not np.isfinite(spacing_stored).all():
            raise ValueError(
                f"{path}:{spacing_key} must contain three finite values, got "
                f"{spacing_stored}."
            )
        if np.any(spacing_stored <= 0.0):
            raise ValueError(f"{path}:{spacing_key} must be strictly positive.")
        spacing_xyz = (
            spacing_stored
            if effective_axis_order == "xyz"
            else spacing_stored[::-1]
        )
        factor = 1.0e-3 if spacing_units == "mm" else 1.0
        spacing_xyz_m = spacing_xyz * factor
    return (
        np.ascontiguousarray(volume),
        selected_key,
        effective_axis_order,
        spacing_xyz_m,
    )


def load_npz_volume(path: Path, key: Optional[str], axis_order: str) -> np.ndarray:
    volume, _, _, _ = load_ground_truth_volume(
        path,
        key,
        axis_order,
        spacing_key="spacing",
        spacing_units="mm",
    )
    return volume


def load_optional_mask(path: Path, key: Optional[str], axis_order: str) -> Optional[np.ndarray]:
    if key is None:
        return None
    with np.load(path, allow_pickle=False) as archive:
        if key not in archive.files:
            raise KeyError(f"{path} has no ROI mask key {key!r}; available keys: {archive.files}.")
        mask = _squeeze_volume(archive[key], f"{path}:{key}") > 0
    if axis_order == "xyz":
        mask = np.transpose(mask, (2, 1, 0))
    return np.ascontiguousarray(mask, dtype=bool)


def load_projection_center_offset_m(
    path: Path,
    key: str = "projection_center_offset",
) -> Tuple[np.ndarray, bool]:
    """Return the renderer's XYZ centering offset in metres and whether it exists."""
    with np.load(path, allow_pickle=False) as archive:
        if key not in archive.files:
            return np.zeros(3, dtype=np.float64), False
        offset_xyz_m = np.asarray(archive[key], dtype=np.float64).reshape(-1)
    if offset_xyz_m.shape != (3,):
        raise ValueError(
            f"{path}:{key} must contain exactly three XYZ values, got "
            f"shape {offset_xyz_m.shape}."
        )
    if not np.isfinite(offset_xyz_m).all():
        raise ValueError(f"{path}:{key} contains NaN or infinity.")
    return offset_xyz_m, True


def load_reconstruction_volume_extent_m(
    case_output_dir: Path,
    explicit_extent_m: Optional[float],
) -> Tuple[float, str]:
    if explicit_extent_m is not None:
        extent_m = float(explicit_extent_m)
        source = "--evaluation-volume-extent-m"
    else:
        metadata_path = case_output_dir / "run_metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                "Physical voxel alignment requires the reconstruction's physical "
                f"extent, but {metadata_path} does not exist. Run the current trainer "
                "first or pass --evaluation-volume-extent-m."
            )
        with metadata_path.open("r", encoding="utf-8") as stream:
            metadata = json.load(stream)
        if "volume_extent_m" not in metadata:
            raise KeyError(f"{metadata_path} has no volume_extent_m value.")
        extent_m = float(metadata["volume_extent_m"])
        source = str(metadata_path)
    if not math.isfinite(extent_m) or extent_m <= 0.0:
        raise ValueError(f"Reconstruction volume extent must be positive, got {extent_m}.")
    return extent_m, source


def projection_offset_to_voxel_shift_zyx(
    offset_xyz_m: np.ndarray,
    volume_shape_zyx: Sequence[int],
    volume_extent_m: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert the positive inverse-centering XYZ translation to ZYX voxels."""
    shape_zyx = np.asarray(tuple(int(size) for size in volume_shape_zyx), dtype=np.int64)
    if shape_zyx.shape != (3,) or np.any(shape_zyx < 2):
        raise ValueError(f"Volume shape must have three dimensions >=2, got {shape_zyx.tolist()}.")
    # create_grid_3d samples both physical endpoints with torch.linspace, so its
    # sample spacing is extent/(N-1), rather than the cell width extent/N.
    spacing_zyx_m = float(volume_extent_m) / (shape_zyx.astype(np.float64) - 1.0)
    offset_zyx_m = np.asarray(offset_xyz_m, dtype=np.float64)[::-1]
    shift_zyx_voxels = offset_zyx_m / spacing_zyx_m
    return shift_zyx_voxels, spacing_zyx_m


def _resample_axis_zero_padded(
    volume: np.ndarray,
    source_coordinates: np.ndarray,
    axis: int,
    interpolation: str,
) -> np.ndarray:
    size = int(volume.shape[axis])
    source_coordinates = np.asarray(source_coordinates, dtype=np.float64).reshape(-1)
    output_size = int(source_coordinates.size)
    broadcast_shape = [1, 1, 1]
    broadcast_shape[axis] = output_size

    if interpolation == "nearest":
        indices = np.floor(source_coordinates + 0.5).astype(np.int64)
        valid = (indices >= 0) & (indices < size)
        sampled = np.take(volume, np.clip(indices, 0, size - 1), axis=axis)
        return np.ascontiguousarray(
            sampled * valid.reshape(broadcast_shape), dtype=np.float32
        )
    if interpolation != "linear":
        raise ValueError(f"Unsupported offset interpolation: {interpolation!r}.")

    lower = np.floor(source_coordinates).astype(np.int64)
    upper = lower + 1
    upper_weight = source_coordinates - lower
    lower_weight = 1.0 - upper_weight
    lower_valid = (lower >= 0) & (lower < size)
    upper_valid = (upper >= 0) & (upper < size)
    lower_values = np.take(volume, np.clip(lower, 0, size - 1), axis=axis)
    upper_values = np.take(volume, np.clip(upper, 0, size - 1), axis=axis)
    lower_factors = (lower_weight * lower_valid).reshape(broadcast_shape)
    upper_factors = (upper_weight * upper_valid).reshape(broadcast_shape)
    return np.ascontiguousarray(
        lower_values * lower_factors + upper_values * upper_factors,
        dtype=np.float32,
    )


def _translate_axis_zero_padded(
    volume: np.ndarray,
    shift_voxels: float,
    axis: int,
    interpolation: str,
) -> np.ndarray:
    if abs(float(shift_voxels)) <= 1.0e-12:
        return np.ascontiguousarray(volume)
    source_coordinates = (
        np.arange(int(volume.shape[axis]), dtype=np.float64)
        - float(shift_voxels)
    )
    return _resample_axis_zero_padded(
        volume,
        source_coordinates=source_coordinates,
        axis=axis,
        interpolation=interpolation,
    )


def translate_volume_zyx(
    volume_zyx: np.ndarray,
    shift_zyx_voxels: Sequence[float],
    interpolation: str = "linear",
) -> np.ndarray:
    """Translate a ZYX volume with zero padding and no wraparound.

    A positive shift moves the object towards increasing array indices. Output
    samples follow ``output[index] = input[index - shift]``.
    """
    translated = _squeeze_volume(volume_zyx, "volume to translate")
    shifts = np.asarray(tuple(float(value) for value in shift_zyx_voxels), dtype=np.float64)
    if shifts.shape != (3,) or not np.isfinite(shifts).all():
        raise ValueError(f"Voxel shift must contain three finite ZYX values, got {shifts}.")
    for axis, shift in enumerate(shifts):
        translated = _translate_axis_zero_padded(
            translated,
            shift_voxels=float(shift),
            axis=axis,
            interpolation=interpolation,
        )
    return np.ascontiguousarray(translated, dtype=np.float32)


def resample_ground_truth_to_prediction_grid(
    ground_truth_zyx: np.ndarray,
    ground_truth_spacing_xyz_m: Sequence[float],
    prediction_shape_zyx: Sequence[int],
    prediction_extent_m: float,
    projection_center_offset_xyz_m: Sequence[float],
    ground_truth_origin_xyz_m: Sequence[float] = (0.0, 0.0, 0.0),
    ground_truth_direction_sign_xyz: Sequence[int] = (1, 1, 1),
    interpolation: str = "nearest",
) -> np.ndarray:
    """Sample a physical GT volume on the centered prediction's ZYX grid.

    The renderer uses ``centered_xyz = original_xyz - center_offset_xyz``.
    A prediction sample at centered coordinate ``q`` therefore corresponds to
    GT world coordinate ``q + center_offset_xyz``.
    """
    ground_truth = _squeeze_volume(ground_truth_zyx, "ground-truth volume")
    prediction_shape = np.asarray(
        tuple(int(size) for size in prediction_shape_zyx), dtype=np.int64
    )
    spacing_xyz_m = np.asarray(ground_truth_spacing_xyz_m, dtype=np.float64)
    center_xyz_m = np.asarray(projection_center_offset_xyz_m, dtype=np.float64)
    origin_xyz_m = np.asarray(ground_truth_origin_xyz_m, dtype=np.float64)
    direction_xyz = np.asarray(ground_truth_direction_sign_xyz, dtype=np.float64)
    if prediction_shape.shape != (3,) or np.any(prediction_shape < 2):
        raise ValueError(f"Invalid prediction shape: {prediction_shape.tolist()}.")
    for name, value in (
        ("ground-truth spacing", spacing_xyz_m),
        ("projection center offset", center_xyz_m),
        ("ground-truth origin", origin_xyz_m),
        ("ground-truth direction signs", direction_xyz),
    ):
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError(f"{name} must contain three finite XYZ values, got {value}.")
    if np.any(spacing_xyz_m <= 0.0):
        raise ValueError("Ground-truth spacing must be strictly positive.")
    if not np.all(np.isin(direction_xyz, (-1.0, 1.0))):
        raise ValueError("Ground-truth direction signs must each be -1 or +1.")

    prediction_shape_xyz = prediction_shape[::-1]
    source_coordinates_xyz = []
    for axis in range(3):
        centered_coordinates_m = np.linspace(
            -float(prediction_extent_m) / 2.0,
            float(prediction_extent_m) / 2.0,
            int(prediction_shape_xyz[axis]),
            dtype=np.float64,
        )
        world_coordinates_m = centered_coordinates_m + center_xyz_m[axis]
        source_coordinates_xyz.append(
            (world_coordinates_m - origin_xyz_m[axis])
            / (direction_xyz[axis] * spacing_xyz_m[axis])
        )
    source_coordinates_zyx = source_coordinates_xyz[::-1]

    # Reduce the largest dimensions first to keep peak memory low for full CT
    # masks such as 512x512x275.
    resampled = ground_truth
    for axis in (2, 1, 0):
        resampled = _resample_axis_zero_padded(
            resampled,
            source_coordinates=source_coordinates_zyx[axis],
            axis=axis,
            interpolation=interpolation,
        )
    return np.ascontiguousarray(resampled, dtype=np.float32)


def normalise_volumes(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    method: str,
) -> Tuple[np.ndarray, np.ndarray, float]:
    prediction = np.asarray(prediction, dtype=np.float32)
    ground_truth = np.asarray(ground_truth, dtype=np.float32)
    if method == "clamp":
        prediction = np.clip(prediction, 0.0, 1.0)
        ground_truth = np.clip(ground_truth, 0.0, 1.0)
        data_range = 1.0
    elif method == "minmax":
        normalised = []
        for volume in (prediction, ground_truth):
            minimum = float(volume.min())
            value_range = float(volume.max()) - minimum
            normalised.append(
                np.zeros_like(volume) if value_range <= 0.0 else (volume - minimum) / value_range
            )
        prediction, ground_truth = normalised
        data_range = 1.0
    elif method == "none":
        minimum = min(float(prediction.min()), float(ground_truth.min()))
        maximum = max(float(prediction.max()), float(ground_truth.max()))
        data_range = maximum - minimum
        if data_range <= 0.0:
            data_range = 1.0
    else:
        raise ValueError(f"Unsupported normalisation method: {method!r}.")
    return (
        np.ascontiguousarray(prediction, dtype=np.float32),
        np.ascontiguousarray(ground_truth, dtype=np.float32),
        float(data_range),
    )


def positive_percentile_threshold(volume: np.ndarray, percentile: float) -> float:
    """Return the GIF-style isovalue from strictly positive raw voxels."""
    return resolve_volume_isovalue(
        volume,
        positive_percentile=float(percentile),
    )


def _box_mean_valid(volume: np.ndarray, window_size: int) -> np.ndarray:
    values = np.asarray(volume, dtype=np.float64)
    integral = np.pad(values, ((1, 0), (1, 0), (1, 0)), mode="constant")
    integral = integral.cumsum(axis=0).cumsum(axis=1).cumsum(axis=2)
    width = int(window_size)
    sums = (
        integral[width:, width:, width:]
        - integral[:-width, width:, width:]
        - integral[width:, :-width, width:]
        - integral[width:, width:, :-width]
        + integral[:-width, :-width, width:]
        + integral[:-width, width:, :-width]
        + integral[width:, :-width, :-width]
        - integral[:-width, :-width, :-width]
    )
    return sums / float(width ** 3)


def structural_similarity_3d(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    data_range: float,
    window_size: int = 7,
) -> Tuple[float, np.ndarray]:
    """Return uniform-window 3D SSIM and its valid-window map.

    Constants and sample covariance match the standard Wang et al. SSIM
    defaults (K1=0.01, K2=0.03). The returned map excludes the border where a
    complete 3D window cannot be formed.
    """
    ground_truth = np.asarray(ground_truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if ground_truth.shape != prediction.shape or ground_truth.ndim != 3:
        raise ValueError(
            f"3D SSIM inputs must have equal shapes, got {ground_truth.shape} and "
            f"{prediction.shape}."
        )
    window_size = int(window_size)
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("SSIM window size must be an odd integer >= 3.")
    if min(ground_truth.shape) < window_size:
        raise ValueError(
            f"SSIM window {window_size} exceeds volume shape {ground_truth.shape}."
        )
    if not math.isfinite(data_range) or data_range <= 0.0:
        raise ValueError(f"SSIM data_range must be positive, got {data_range}.")

    mean_gt = _box_mean_valid(ground_truth, window_size)
    mean_prediction = _box_mean_valid(prediction, window_size)
    covariance_scale = float(window_size ** 3) / float(window_size ** 3 - 1)
    variance_gt = covariance_scale * (
        _box_mean_valid(ground_truth * ground_truth, window_size) - mean_gt * mean_gt
    )
    variance_prediction = covariance_scale * (
        _box_mean_valid(prediction * prediction, window_size)
        - mean_prediction * mean_prediction
    )
    covariance = covariance_scale * (
        _box_mean_valid(ground_truth * prediction, window_size)
        - mean_gt * mean_prediction
    )
    variance_gt = np.maximum(variance_gt, 0.0)
    variance_prediction = np.maximum(variance_prediction, 0.0)
    c1 = (0.01 * float(data_range)) ** 2
    c2 = (0.03 * float(data_range)) ** 2
    numerator = (2.0 * mean_gt * mean_prediction + c1) * (2.0 * covariance + c2)
    denominator = (
        (mean_gt * mean_gt + mean_prediction * mean_prediction + c1)
        * (variance_gt + variance_prediction + c2)
    )
    ssim_map = np.divide(
        numerator,
        denominator,
        out=np.ones_like(numerator),
        where=denominator != 0.0,
    )
    return float(np.mean(ssim_map, dtype=np.float64)), np.asarray(ssim_map, dtype=np.float32)


def compute_volume_metrics(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    *,
    prediction_threshold: float,
    ground_truth_threshold: float,
    normalisation: str,
    metric_mask: str,
    roi_mask: Optional[np.ndarray],
    ssim_window_size: int,
    prediction_threshold_percentile: Optional[float] = None,
) -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
    prediction = _squeeze_volume(prediction, "prediction volume")
    ground_truth = _squeeze_volume(ground_truth, "ground-truth volume")
    if prediction.shape != ground_truth.shape:
        raise ValueError(
            f"Prediction and ground truth shapes differ: {prediction.shape} versus "
            f"{ground_truth.shape}. Set the training --volume-size correctly; automatic "
            "resampling is intentionally not performed."
        )
    if roi_mask is None:
        roi = np.ones(prediction.shape, dtype=bool)
    else:
        roi = np.asarray(roi_mask, dtype=bool)
        if roi.shape != prediction.shape:
            raise ValueError(f"ROI mask shape {roi.shape} differs from volume shape {prediction.shape}.")
    if not np.any(roi):
        raise ValueError("The evaluation ROI contains no voxels.")

    prediction_eval, ground_truth_eval, data_range = normalise_volumes(
        prediction, ground_truth, normalisation
    )
    if prediction_threshold_percentile is None:
        applied_prediction_threshold = float(prediction_threshold)
        prediction_threshold_mode = "absolute"
        prediction_threshold_domain = "raw_prediction"
        prediction_mask = (prediction > applied_prediction_threshold) & roi
    else:
        applied_prediction_threshold = positive_percentile_threshold(
            prediction,
            prediction_threshold_percentile,
        )
        prediction_threshold_mode = "positive-percentile"
        prediction_threshold_domain = "raw_prediction"
        prediction_mask = (prediction > applied_prediction_threshold) & roi
    ground_truth_mask = (ground_truth_eval > float(ground_truth_threshold)) & roi
    intersection_count = int(np.count_nonzero(prediction_mask & ground_truth_mask))
    prediction_count = int(np.count_nonzero(prediction_mask))
    ground_truth_count = int(np.count_nonzero(ground_truth_mask))
    denominator = prediction_count + ground_truth_count
    dice = 1.0 if denominator == 0 else 2.0 * intersection_count / denominator

    if metric_mask == "ground-truth":
        evaluation_mask = ground_truth_mask.copy()
    elif metric_mask == "union":
        evaluation_mask = prediction_mask | ground_truth_mask
    elif metric_mask == "all":
        evaluation_mask = roi.copy()
    else:
        raise ValueError(f"Unsupported metric mask: {metric_mask!r}.")
    evaluation_count = int(np.count_nonzero(evaluation_mask))
    difference = prediction_eval.astype(np.float64) - ground_truth_eval.astype(np.float64)
    mse_3d = float(np.mean(difference[roi] ** 2, dtype=np.float64))
    if evaluation_count:
        masked_difference = difference[evaluation_mask]
        masked_mse = float(np.mean(masked_difference ** 2, dtype=np.float64))
        masked_mae = float(np.mean(np.abs(masked_difference), dtype=np.float64))
        masked_psnr = (
            float("inf")
            if masked_mse == 0.0
            else 20.0 * math.log10(float(data_range) / math.sqrt(masked_mse))
        )
    else:
        masked_mse = float("nan")
        masked_mae = float("nan")
        masked_psnr = float("nan")

    ssim, ssim_map = structural_similarity_3d(
        ground_truth_eval,
        prediction_eval,
        data_range=data_range,
        window_size=ssim_window_size,
    )
    border = ssim_window_size // 2
    valid_evaluation_mask = evaluation_mask[
        border : evaluation_mask.shape[0] - border,
        border : evaluation_mask.shape[1] - border,
        border : evaluation_mask.shape[2] - border,
    ]
    masked_ssim = (
        float(np.mean(ssim_map[valid_evaluation_mask], dtype=np.float64))
        if np.any(valid_evaluation_mask)
        else float("nan")
    )

    metrics: Dict[str, object] = {
        "masked_dice_3d": float(dice),
        "mse_3d": mse_3d,
        "masked_mse": masked_mse,
        "masked_mae": masked_mae,
        "masked_psnr": masked_psnr,
        "ssim_3d": ssim,
        "masked_ssim_3d": masked_ssim,
        "prediction_foreground_voxels": prediction_count,
        "ground_truth_foreground_voxels": ground_truth_count,
        "intersection_voxels": intersection_count,
        "evaluation_mask_voxels": evaluation_count,
        "roi_voxels": int(np.count_nonzero(roi)),
        "total_voxels": int(prediction.size),
        "volume_shape_zyx": list(prediction.shape),
        "prediction_threshold": float(applied_prediction_threshold),
        "prediction_threshold_mode": prediction_threshold_mode,
        "prediction_threshold_domain": prediction_threshold_domain,
        "prediction_threshold_percentile": (
            None
            if prediction_threshold_percentile is None
            else float(prediction_threshold_percentile)
        ),
        "ground_truth_threshold": float(ground_truth_threshold),
        "ssim_data_range": float(data_range),
        "ssim_window_size": int(ssim_window_size),
    }
    arrays = {
        "prediction_volume_zyx": prediction_eval,
        "ground_truth_volume_zyx": ground_truth_eval,
        "prediction_mask_zyx": prediction_mask,
        "ground_truth_mask_zyx": ground_truth_mask,
        "evaluation_mask_zyx": evaluation_mask,
        "roi_mask_zyx": roi,
        "ssim_map_valid_zyx": ssim_map,
    }
    return metrics, arrays


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number):
            return None
        if math.isinf(number):
            return "Infinity" if number > 0 else "-Infinity"
        return number
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(_json_safe(value), indent=2), encoding="utf-8")


def finite_descriptive_statistics(values: np.ndarray) -> Dict[str, object]:
    """Summarize finite values and return the sample-based standard error."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    count = int(finite.size)
    sample_std = (
        float(np.std(finite, ddof=1))
        if count >= 2
        else float("nan")
    )
    return {
        "mean": float(np.mean(finite)) if count else float("nan"),
        # Preserve the existing population-standard-deviation convention.
        "std": float(np.std(finite, ddof=0)) if count else float("nan"),
        "standard_error": (
            sample_std / math.sqrt(float(count))
            if count >= 2
            else float("nan")
        ),
        "min": float(np.min(finite)) if count else float("nan"),
        "max": float(np.max(finite)) if count else float("nan"),
        "num_finite": count,
        "num_nonfinite": int(values.size - count),
    }


def summarise_metrics(
    records: Sequence[Mapping[str, object]],
    split: str,
    num_cases_requested: Optional[int] = None,
) -> Dict[str, object]:
    completed = [record for record in records if record.get("status") == "completed"]
    requested = len(records) if num_cases_requested is None else int(num_cases_requested)
    summary: Dict[str, object] = {
        "split": split,
        "num_cases_requested": requested,
        "num_cases_recorded": len(records),
        "num_cases_completed": len(completed),
        "num_cases_failed": len(records) - len(completed),
        "num_cases_pending": max(requested - len(records), 0),
        "metrics": {},
    }
    metrics_summary: Dict[str, object] = {}
    for name in METRIC_NAMES:
        values = np.asarray([float(record[name]) for record in completed], dtype=np.float64)
        metrics_summary[name] = finite_descriptive_statistics(values)
    summary["metrics"] = metrics_summary
    return summary


def summarise_timings(
    records: Sequence[Mapping[str, object]],
    num_cases_requested: int,
) -> Dict[str, object]:
    completed = [record for record in records if record.get("status") == "completed"]
    timing_summary: Dict[str, object] = {}
    for name in TIMING_NAMES:
        values = np.asarray(
            [
                float(record[name])
                for record in completed
                if record.get(name) is not None
            ],
            dtype=np.float64,
        )
        statistics = finite_descriptive_statistics(values)
        statistics["num_cases"] = statistics.pop("num_finite")
        timing_summary[name] = statistics
    case_values = np.asarray(
        [
            float(record["case_wall_time_seconds"])
            for record in completed
            if record.get("case_wall_time_seconds") is not None
        ],
        dtype=np.float64,
    )
    finite_case_values = case_values[np.isfinite(case_values)]
    average_case_seconds = (
        float(np.mean(finite_case_values))
        if finite_case_values.size
        else float("nan")
    )
    remaining_cases = max(int(num_cases_requested) - len(records), 0)
    timing_summary["average_case_seconds"] = average_case_seconds
    timing_summary["average_case_seconds_standard_error"] = timing_summary[
        "case_wall_time_seconds"
    ]["standard_error"]
    timing_summary["estimated_full_split_seconds"] = (
        average_case_seconds * int(num_cases_requested)
        if math.isfinite(average_case_seconds)
        else float("nan")
    )
    timing_summary["estimated_remaining_seconds"] = (
        average_case_seconds * remaining_cases
        if math.isfinite(average_case_seconds)
        else float("nan")
    )
    timing_summary["remaining_cases"] = remaining_cases
    return timing_summary


def write_aggregate_reports(
    output_dir: Path,
    records: Sequence[Mapping[str, object]],
    split: str,
    num_cases_requested: Optional[int] = None,
    output_mode: str = "full",
    evaluation_config: Optional[Mapping[str, object]] = None,
) -> None:
    requested = len(records) if num_cases_requested is None else int(num_cases_requested)
    summary = summarise_metrics(records, split, requested)
    summary["timing"] = summarise_timings(records, requested)
    if output_mode == "json-only":
        completed = [
            record for record in records if record.get("status") == "completed"
        ]
        write_json(
            output_dir / "evaluation_results.json",
            {
                "format": "3dgr_car_stage2_evaluation_v2",
                "configuration": dict(evaluation_config or {}),
                "summary": summary,
                "matrix": {
                    "case_names": [
                        str(record["case_name"]) for record in completed
                    ],
                    "metric_names": list(METRIC_NAMES),
                    "values": [
                        [float(record[name]) for name in METRIC_NAMES]
                        for record in completed
                    ],
                    "column_mean": [
                        float(summary["metrics"][name]["mean"])
                        for name in METRIC_NAMES
                    ],
                    "column_standard_error": [
                        float(summary["metrics"][name]["standard_error"])
                        for name in METRIC_NAMES
                    ],
                    "column_num_finite": [
                        int(summary["metrics"][name]["num_finite"])
                        for name in METRIC_NAMES
                    ],
                },
                "cases": list(records),
            },
        )
        return
    if output_mode != "full":
        raise ValueError(f"Unsupported output mode: {output_mode!r}.")
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    write_json(metrics_dir / "per_case_metrics.json", list(records))
    write_json(
        metrics_dir / "summary_metrics.json",
        summary,
    )

    fieldnames = [
        "split",
        "evaluation_split",
        "split_case_id",
        "case_name",
        "vessel_type",
        "source_case_id",
        "status",
        *METRIC_NAMES,
        "prediction_foreground_voxels",
        "ground_truth_foreground_voxels",
        "intersection_voxels",
        "evaluation_mask_voxels",
        "total_voxels",
        "projection_npz",
        "ground_truth_npz",
        "reconstruction_volume",
        "evaluation_arrays",
        "prediction_threshold",
        "prediction_threshold_mode",
        "prediction_threshold_domain",
        "prediction_threshold_percentile",
        "ground_truth_threshold",
        "normalisation",
        "metric_mask",
        "ground_truth_alignment",
        "ground_truth_selected_key",
        "ground_truth_effective_axis_order",
        "ground_truth_spacing_xyz_m",
        "projection_center_offset_found",
        "projection_offset_applied",
        "projection_center_offset_xyz_m",
        "applied_prediction_shift_zyx_voxels",
        "volume_extent_m_for_offset",
        "offset_interpolation",
        *TIMING_NAMES,
        "optimization_seconds_per_iteration",
        "optimization_iterations_completed",
        "optimization_early_stopped",
        "visualization_artifacts_enabled",
        "case_cache_removed",
        "error",
    ]
    with (metrics_dir / "per_case_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    completed = [record for record in records if record.get("status") == "completed"]
    matrix = np.asarray(
        [[float(record[name]) for name in METRIC_NAMES] for record in completed],
        dtype=np.float64,
    ).reshape(len(completed), len(METRIC_NAMES))
    np.savez_compressed(
        metrics_dir / "metrics_matrix.npz",
        case_names=np.asarray([str(record["case_name"]) for record in completed]),
        metric_names=np.asarray(METRIC_NAMES),
        values=matrix,
        column_mean=np.asarray(
            [summary["metrics"][name]["mean"] for name in METRIC_NAMES],
            dtype=np.float64,
        ),
        column_standard_error=np.asarray(
            [summary["metrics"][name]["standard_error"] for name in METRIC_NAMES],
            dtype=np.float64,
        ),
        column_num_finite=np.asarray(
            [summary["metrics"][name]["num_finite"] for name in METRIC_NAMES],
            dtype=np.int64,
        ),
    )


def load_case_optimization_timing(case_output_dir: Path) -> Dict[str, object]:
    timing_path = case_output_dir / "optimization_timing.json"
    if not timing_path.is_file():
        return {}
    with timing_path.open("r", encoding="utf-8") as stream:
        timing = json.load(stream)
    return {
        "optimization_elapsed_seconds": timing.get("elapsed_seconds"),
        "optimization_seconds_per_iteration": timing.get("seconds_per_iteration"),
        "optimization_iterations_requested": timing.get("iterations_requested"),
        "optimization_iterations_completed": timing.get("iterations_completed"),
        "optimization_early_stopped": timing.get("early_stopped"),
        "optimization_gpu_name": timing.get("gpu_name"),
    }


def remove_temporary_case_cache(case_output_dir: Path, cache_root: Path) -> None:
    resolved_case = case_output_dir.resolve()
    resolved_root = cache_root.resolve()
    if resolved_case.parent != resolved_root or resolved_case == resolved_root:
        raise ValueError(f"Refusing to remove unsafe case cache path: {resolved_case}")
    if case_output_dir.is_symlink():
        raise ValueError(f"Refusing to remove symlinked case cache: {case_output_dir}")
    if case_output_dir.exists():
        shutil.rmtree(case_output_dir)


def run_reconstruction(
    training_script: Path,
    projection_path: Path,
    case_output_dir: Path,
    training_args: Sequence[str],
) -> None:
    command = [
        sys.executable,
        str(training_script),
        "--input",
        str(projection_path),
        "--output-dir",
        str(case_output_dir),
        *training_args,
    ]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def parse_args(argv: Optional[Sequence[str]] = None) -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Run Stage-2 reconstruction for cases in one split and write per-case "
            "3D mask/SSIM metrics. Unrecognised options are forwarded to "
            "train_stage2_npz.py."
        )
    )
    parser.add_argument("--input-dir", required=True, help="Directory of Stage-2 projection NPZs.")
    parser.add_argument("--split-json", required=True, help="JSON containing train/validation/test cases.")
    parser.add_argument(
        "--split",
        required=True,
        help=(
            "Split to evaluate: train, validation/val, test, or val_test "
            "(validation followed by test)."
        ),
    )
    parser.add_argument(
        "--eval-case-ids",
        nargs="+",
        default=None,
        help="Optional ordered subset of case identifiers from the selected split.",
    )
    parser.add_argument(
        "--num-eval-cases",
        type=_optional_nonnegative_integer,
        default=None,
        help="Evaluate only the first N selected cases; 'all' keeps every case.",
    )
    parser.add_argument("--output-dir", required=True, help="Root for reconstructions and metric reports.")
    parser.add_argument(
        "--view-indices",
        nargs="+",
        type=int,
        default=[0, 1],
        help=(
            "One or more stored view indices used as reconstruction inputs for "
            "every case in this evaluation run."
        ),
    )
    parser.add_argument(
        "--ground-truth-dir",
        "--gt-dir",
        dest="ground_truth_dir",
        required=True,
        help="Directory of case-matched NPZs containing ground-truth 3D volumes.",
    )
    parser.add_argument("--ground-truth-key", default=None, help="NPZ array key; auto-detected by default.")
    parser.add_argument(
        "--ground-truth-axis-order",
        choices=("auto", "zyx", "xyz"),
        default="auto",
        help="Axis order in the GT NPZ. Auto treats `vol` as XYZ and other keys as ZYX.",
    )
    parser.add_argument(
        "--ground-truth-alignment",
        choices=("auto", "physical", "same-grid"),
        default="auto",
        help=(
            "Auto uses physical alignment when GT spacing exists or shapes differ; "
            "same-grid requires matching arrays."
        ),
    )
    parser.add_argument(
        "--ground-truth-spacing-key",
        default="spacing",
        help="GT NPZ key containing three voxel spacings.",
    )
    parser.add_argument(
        "--ground-truth-spacing-units",
        choices=("mm", "m"),
        default="mm",
        help="Units of the GT spacing array; ImageCAS masks use millimetres.",
    )
    parser.add_argument(
        "--ground-truth-origin-m",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="Physical XYZ coordinate in metres of GT array index [0,0,0].",
    )
    parser.add_argument(
        "--ground-truth-direction-signs",
        nargs=3,
        type=int,
        choices=(-1, 1),
        default=(1, 1, 1),
        metavar=("SX", "SY", "SZ"),
        help="Axis-aligned GT physical direction signs for X, Y, and Z.",
    )
    parser.add_argument(
        "--ground-truth-interpolation",
        choices=("nearest", "linear"),
        default="nearest",
        help="Interpolation used to sample GT on the prediction grid.",
    )
    parser.add_argument(
        "--evaluation-mask-key",
        default=None,
        help="Optional ROI mask key in each GT NPZ; Dice and other masked metrics are restricted to it.",
    )
    parser.add_argument(
        "--projection-offset-mode",
        choices=("auto", "required", "ignore"),
        default="auto",
        help=(
            "Undo projection centering when the input NPZ contains an offset "
            "(default); require the key, or explicitly ignore it."
        ),
    )
    parser.add_argument(
        "--projection-offset-key",
        default="projection_center_offset",
        help="Projection NPZ key containing the XYZ centering offset in metres.",
    )
    parser.add_argument(
        "--offset-interpolation",
        choices=("linear", "nearest"),
        default="linear",
        help="Interpolation used when translating the prediction back to its original position.",
    )
    parser.add_argument(
        "--evaluation-volume-extent-m",
        type=float,
        default=None,
        help=(
            "Physical side length used to convert offsets to voxels. By default "
            "it is read from each reconstruction's run_metadata.json."
        ),
    )
    prediction_threshold_group = parser.add_mutually_exclusive_group()
    prediction_threshold_group.add_argument(
        "--prediction-threshold",
        "--volume-gif-isovalue",
        dest="prediction_threshold",
        type=float,
        default=None,
        help=(
            "Use a fixed raw-prediction threshold instead of the default "
            "GIF-matched positive-voxel percentile. --volume-gif-isovalue is "
            "retained as a compatible alias."
        ),
    )
    prediction_threshold_group.add_argument(
        "--prediction-threshold-percentile",
        type=float,
        default=None,
        help=(
            "Per-case percentile of strictly positive raw prediction voxels; "
            "defaults to P97, matching reconstructed_volume.gif."
        ),
    )
    parser.add_argument("--ground-truth-threshold", type=float, default=0.0)
    parser.add_argument(
        "--normalisation",
        choices=("clamp", "minmax", "none"),
        default="clamp",
        help="Intensity handling before thresholds and SSIM; clamp limits values to [0,1].",
    )
    parser.add_argument(
        "--metric-mask",
        choices=("ground-truth", "union", "all"),
        default="ground-truth",
        help="Mask used for masked MSE/MAE/PSNR/SSIM. Dice always compares the two binary masks.",
    )
    parser.add_argument("--ssim-window-size", type=int, default=7)
    parser.add_argument(
        "--save-ssim-map",
        action="store_true",
        help="Include the valid-window 3D SSIM map in each evaluation_arrays.npz.",
    )
    parser.add_argument(
        "--save-evaluation-arrays",
        action="store_true",
        help=(
            "Persist each aligned prediction, ground truth, and binary mask even "
            "in json-only mode. Arrays are written under voxel_masks/."
        ),
    )
    parser.add_argument(
        "--output-mode",
        choices=("json-only", "full"),
        default="json-only",
        help=(
            "Write one evaluation_results.json and remove temporary case artifacts "
            "by default; full preserves the legacy CSV/NPZ/per-case outputs."
        ),
    )
    parser.add_argument(
        "--keep-case-cache",
        action="store_true",
        help="Keep temporary reconstruction/timing files after JSON-only evaluation.",
    )
    parser.add_argument(
        "--max-visualizations",
        type=_optional_nonnegative_integer,
        default=None,
        help=(
            "In full mode, create complete reconstruction/visualization artifacts "
            "for only the first N cases; 'all' (the default) creates them for all."
        ),
    )
    parser.add_argument(
        "--skip-reconstruction",
        action="store_true",
        help="Only score existing cases/<case>/reconstructed_volume_zyx.npy files.",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse an existing reconstructed volume per case; otherwise run training.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record failed cases and continue; the process still exits nonzero if any case fails.",
    )
    parser.add_argument(
        "--early-stop-checks",
        type=int,
        default=7,
        help=(
            "Required positive number of consecutive non-improving logging checks "
            "before each case optimization stops (default: 7)."
        ),
    )
    args, training_args = parser.parse_known_args(argv)
    forbidden = {"--input", "--output-dir"}
    conflicting = sorted(forbidden.intersection(training_args))
    if conflicting:
        parser.error(f"Evaluation owns these trainer options: {conflicting}.")
    if args.skip_reconstruction and training_args:
        parser.error(f"Trainer options have no effect with --skip-reconstruction: {training_args}.")
    if args.ssim_window_size < 3 or args.ssim_window_size % 2 == 0:
        parser.error("--ssim-window-size must be an odd integer >= 3.")
    if (
        args.output_mode == "json-only"
        and args.save_ssim_map
        and not args.save_evaluation_arrays
    ):
        parser.error(
            "--save-ssim-map in json-only mode requires --save-evaluation-arrays."
        )
    if args.early_stop_checks <= 0:
        parser.error(
            "--early-stop-checks must be positive; split evaluation requires "
            "early stopping for every optimized case."
        )
    if args.prediction_threshold is None and args.prediction_threshold_percentile is None:
        args.prediction_threshold_percentile = float(
            DEFAULT_VOLUME_GIF_POSITIVE_PERCENTILE
        )
    if (
        args.prediction_threshold_percentile is not None
        and not 0.0 <= args.prediction_threshold_percentile < 100.0
    ):
        parser.error("--prediction-threshold-percentile must be in [0, 100).")
    return args, training_args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args, training_args = parse_args(argv)
    effective_training_args = [
        *training_args,
        "--view-indices",
        *(str(int(index)) for index in args.view_indices),
        "--early-stop-checks",
        str(int(args.early_stop_checks)),
        "--record-optimization-time",
    ]
    if args.prediction_threshold is not None:
        effective_training_args.extend(
            ("--prediction-threshold", str(float(args.prediction_threshold)))
        )
    else:
        effective_training_args.extend(
            (
                "--prediction-threshold-percentile",
                str(float(args.prediction_threshold_percentile)),
            )
        )
    if args.output_mode == "json-only":
        effective_training_args.extend(
            ("--evaluation-cache-only", "--no-volume-gif")
        )
    input_dir = Path(args.input_dir).expanduser().resolve()
    split_json = Path(args.split_json).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    ground_truth_dir = Path(args.ground_truth_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded_references = load_split_case_references(split_json, args.split)
    reference_split_labels: Dict[str, str] = {}
    if _normalise_key(args.split) in {"valtest", "validationtest"}:
        for split_name, split_label in (("val", "validation"), ("test", "test")):
            for reference in load_split_case_references(split_json, split_name):
                reference_split_labels.setdefault(reference, split_label)
    else:
        normalised_split = _normalise_key(args.split)
        split_label = (
            "validation"
            if normalised_split in {"val", "valid", "validation", "dev"}
            else "train"
            if normalised_split in {"train", "training"}
            else "test"
            if normalised_split in {"test", "testing"}
            else args.split
        )
        reference_split_labels = {
            reference: split_label for reference in loaded_references
        }
    references = select_case_references(
        loaded_references,
        args.eval_case_ids,
        args.num_eval_cases,
    )
    projection_index = NpzIndex(input_dir)
    ground_truth_index = NpzIndex(ground_truth_dir)
    training_script = Path(__file__).resolve().with_name("train_stage2_npz.py")
    records: List[Dict[str, object]] = []
    failed = False

    evaluation_config = {
        **vars(args),
        "input_dir": str(input_dir),
        "split_json": str(split_json),
        "output_dir": str(output_dir),
        "ground_truth_dir": str(ground_truth_dir),
        "training_args": effective_training_args,
        "resolved_split_cases": references,
    }
    if args.output_mode == "full":
        metrics_dir = output_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        write_json(metrics_dir / "evaluation_config.json", evaluation_config)

    print(f"Evaluating split {args.split!r}: {len(references)} cases", flush=True)
    for case_number, reference in enumerate(references, start=1):
        case_timer_start = time.perf_counter()
        should_train = False
        cache_created_by_run = False
        case_output_dir: Optional[Path] = None
        record: Dict[str, object] = {
            "split": reference_split_labels.get(reference, args.split),
            "evaluation_split": args.split,
            "split_case_id": reference,
            "case_name": reference,
            "status": "failed",
        }
        try:
            projection_path = projection_index.resolve([reference], "projection")
            sample_name = projection_sample_name(projection_path)
            record["case_name"] = sample_name
            record["projection_npz"] = str(projection_path)
            (
                ground_truth_references,
                vessel_type,
                source_case_id,
            ) = ground_truth_case_references(
                projection_path,
                sample_name,
                reference,
            )
            ground_truth_path = ground_truth_index.resolve(
                ground_truth_references,
                "ground-truth",
            )
            record["ground_truth_npz"] = str(ground_truth_path)
            record["vessel_type"] = vessel_type
            record["source_case_id"] = source_case_id
            record["ground_truth_match_references"] = ground_truth_references
            case_output_dir = output_dir / "cases" / projection_path.stem
            cache_had_files = bool(
                case_output_dir.is_dir() and any(case_output_dir.iterdir())
            )
            case_output_dir.mkdir(parents=True, exist_ok=True)
            reconstruction_path = case_output_dir / "reconstructed_volume_zyx.npy"
            record["reconstruction_volume"] = str(reconstruction_path)

            print(
                f"[{case_number}/{len(references)}] {sample_name} "
                f"({projection_path.name})",
                flush=True,
            )
            should_train = not args.skip_reconstruction and not (
                args.reuse_existing and reconstruction_path.is_file()
            )
            if (
                args.output_mode == "json-only"
                and should_train
                and cache_had_files
                and not args.keep_case_cache
            ):
                raise FileExistsError(
                    f"JSON-only cleanup will not overwrite and remove a non-empty "
                    f"existing cache: {case_output_dir}. Use a fresh output directory, "
                    "--reuse-existing, or --keep-case-cache."
                )
            create_visualization_artifacts = bool(
                args.output_mode == "full"
                and (
                    args.max_visualizations is None
                    or case_number <= int(args.max_visualizations)
                )
            )
            record["visualization_artifacts_enabled"] = (
                create_visualization_artifacts
            )
            case_training_args = list(effective_training_args)
            if args.output_mode == "full" and not create_visualization_artifacts:
                case_training_args.extend(("--evaluation-cache-only", "--no-volume-gif"))
            if should_train:
                cache_created_by_run = not cache_had_files
                reconstruction_timer_start = time.perf_counter()
                run_reconstruction(
                    training_script,
                    projection_path,
                    case_output_dir,
                    case_training_args,
                )
                record["reconstruction_wall_time_seconds"] = float(
                    time.perf_counter() - reconstruction_timer_start
                )
            else:
                record["reconstruction_wall_time_seconds"] = 0.0
            record["used_existing_reconstruction"] = bool(not should_train)
            if not reconstruction_path.is_file():
                raise FileNotFoundError(
                    f"Expected reconstructed volume was not produced: {reconstruction_path}"
                )

            record.update(load_case_optimization_timing(case_output_dir))
            metrics_timer_start = time.perf_counter()
            prediction = _squeeze_volume(
                np.load(reconstruction_path, allow_pickle=False),
                str(reconstruction_path),
            )
            prediction_shape_before_alignment = tuple(prediction.shape)
            offset_xyz_m, offset_found = load_projection_center_offset_m(
                projection_path,
                key=args.projection_offset_key,
            )
            if args.projection_offset_mode == "required" and not offset_found:
                raise KeyError(
                    f"{projection_path} has no required projection offset key "
                    f"{args.projection_offset_key!r}."
                )
            applied_offset_xyz_m = (
                offset_xyz_m
                if offset_found and args.projection_offset_mode != "ignore"
                else np.zeros(3, dtype=np.float64)
            )
            apply_offset = bool(
                offset_found
                and args.projection_offset_mode != "ignore"
                and np.any(np.abs(offset_xyz_m) > 0.0)
            )

            (
                ground_truth,
                ground_truth_selected_key,
                ground_truth_effective_axis_order,
                ground_truth_spacing_xyz_m,
            ) = load_ground_truth_volume(
                ground_truth_path,
                args.ground_truth_key,
                args.ground_truth_axis_order,
                args.ground_truth_spacing_key,
                args.ground_truth_spacing_units,
            )
            ground_truth_original_shape_zyx = tuple(ground_truth.shape)
            roi_mask = load_optional_mask(
                ground_truth_path,
                args.evaluation_mask_key,
                ground_truth_effective_axis_order,
            )
            ground_truth_alignment = args.ground_truth_alignment
            if ground_truth_alignment == "auto":
                ground_truth_alignment = (
                    "physical"
                    if ground_truth_spacing_xyz_m is not None
                    or prediction.shape != ground_truth.shape
                    else "same-grid"
                )
            if (
                ground_truth_alignment == "physical"
                and prediction.shape != ground_truth.shape
                and not offset_found
                and args.projection_offset_mode != "ignore"
            ):
                raise KeyError(
                    "Physical alignment of different-sized prediction and GT grids "
                    f"requires {args.projection_offset_key!r} in {projection_path}. "
                    "Use --projection-offset-mode ignore only if the GT physical "
                    "origin already describes a centered coordinate frame."
                )

            applied_shift_zyx_voxels = np.zeros(3, dtype=np.float64)
            spacing_zyx_m: Optional[np.ndarray] = None
            volume_extent_m_for_offset: Optional[float] = None
            volume_extent_source: Optional[str] = None
            if ground_truth_alignment == "physical" or apply_offset:
                volume_extent_m_for_offset, volume_extent_source = (
                    load_reconstruction_volume_extent_m(
                        case_output_dir,
                        args.evaluation_volume_extent_m,
                    )
                )
            if volume_extent_m_for_offset is not None:
                applied_shift_zyx_voxels, spacing_zyx_m = (
                    projection_offset_to_voxel_shift_zyx(
                        applied_offset_xyz_m,
                        prediction.shape,
                        volume_extent_m_for_offset,
                    )
                )

            if ground_truth_alignment == "physical":
                if ground_truth_spacing_xyz_m is None:
                    raise KeyError(
                        f"Physical GT alignment requires {args.ground_truth_spacing_key!r} "
                        f"in {ground_truth_path}."
                    )
                ground_truth = resample_ground_truth_to_prediction_grid(
                    ground_truth,
                    ground_truth_spacing_xyz_m,
                    prediction.shape,
                    float(volume_extent_m_for_offset),
                    applied_offset_xyz_m,
                    ground_truth_origin_xyz_m=args.ground_truth_origin_m,
                    ground_truth_direction_sign_xyz=(
                        args.ground_truth_direction_signs
                    ),
                    interpolation=args.ground_truth_interpolation,
                )
                if roi_mask is not None:
                    roi_mask = resample_ground_truth_to_prediction_grid(
                        roi_mask.astype(np.float32),
                        ground_truth_spacing_xyz_m,
                        prediction.shape,
                        float(volume_extent_m_for_offset),
                        applied_offset_xyz_m,
                        ground_truth_origin_xyz_m=args.ground_truth_origin_m,
                        ground_truth_direction_sign_xyz=(
                            args.ground_truth_direction_signs
                        ),
                        interpolation="nearest",
                    ) > 0.5
                print(
                    "  Resampled physical GT to prediction grid: "
                    f"{ground_truth_original_shape_zyx} -> {prediction.shape}; "
                    f"center_xyz_m={applied_offset_xyz_m.tolist()}",
                    flush=True,
                )
            elif ground_truth_alignment == "same-grid" and apply_offset:
                prediction = translate_volume_zyx(
                    prediction,
                    applied_shift_zyx_voxels,
                    interpolation=args.offset_interpolation,
                )
                print(
                    "  Reversed projection centering: "
                    f"offset_xyz_m={offset_xyz_m.tolist()} -> "
                    f"shift_zyx_voxels={applied_shift_zyx_voxels.tolist()}",
                    flush=True,
                )
            metrics, arrays = compute_volume_metrics(
                prediction,
                ground_truth,
                prediction_threshold=(
                    0.0
                    if args.prediction_threshold is None
                    else args.prediction_threshold
                ),
                ground_truth_threshold=args.ground_truth_threshold,
                normalisation=args.normalisation,
                metric_mask=args.metric_mask,
                roi_mask=roi_mask,
                ssim_window_size=args.ssim_window_size,
                prediction_threshold_percentile=(
                    args.prediction_threshold_percentile
                ),
            )
            arrays["projection_center_offset_xyz_m"] = offset_xyz_m.astype(np.float64)
            arrays["applied_prediction_shift_zyx_voxels"] = (
                applied_shift_zyx_voxels.astype(np.float64)
            )
            if ground_truth_spacing_xyz_m is not None:
                arrays["ground_truth_spacing_xyz_m"] = (
                    ground_truth_spacing_xyz_m.astype(np.float64)
                )
            arrays["ground_truth_origin_xyz_m"] = np.asarray(
                args.ground_truth_origin_m,
                dtype=np.float64,
            )
            record.update(metrics)
            record.update(
                {
                    "status": "completed",
                    "normalisation": args.normalisation,
                    "metric_mask": args.metric_mask,
                    "ground_truth_axis_order_requested": (
                        args.ground_truth_axis_order
                    ),
                    "ground_truth_effective_axis_order": (
                        ground_truth_effective_axis_order
                    ),
                    "ground_truth_key": args.ground_truth_key,
                    "ground_truth_selected_key": ground_truth_selected_key,
                    "ground_truth_alignment": ground_truth_alignment,
                    "ground_truth_original_shape_zyx": list(
                        ground_truth_original_shape_zyx
                    ),
                    "prediction_shape_before_alignment_zyx": list(
                        prediction_shape_before_alignment
                    ),
                    "ground_truth_spacing_xyz_m": (
                        None
                        if ground_truth_spacing_xyz_m is None
                        else ground_truth_spacing_xyz_m.tolist()
                    ),
                    "ground_truth_origin_xyz_m": list(
                        args.ground_truth_origin_m
                    ),
                    "ground_truth_direction_signs_xyz": list(
                        args.ground_truth_direction_signs
                    ),
                    "ground_truth_interpolation": (
                        args.ground_truth_interpolation
                    ),
                    "evaluation_mask_key": args.evaluation_mask_key,
                    "projection_center_offset_found": bool(offset_found),
                    "projection_offset_mode": args.projection_offset_mode,
                    "projection_offset_applied": bool(apply_offset),
                    "projection_offset_key": args.projection_offset_key,
                    "projection_center_offset_xyz_m": offset_xyz_m.tolist(),
                    "applied_prediction_translation_xyz_m": (
                        offset_xyz_m.tolist() if apply_offset else [0.0, 0.0, 0.0]
                    ),
                    "applied_prediction_shift_zyx_voxels": (
                        applied_shift_zyx_voxels.tolist()
                    ),
                    "voxel_spacing_zyx_m": (
                        None if spacing_zyx_m is None else spacing_zyx_m.tolist()
                    ),
                    "volume_extent_m_for_offset": volume_extent_m_for_offset,
                    "volume_extent_source": volume_extent_source,
                    "offset_interpolation": args.offset_interpolation,
                }
            )
            record["metrics_wall_time_seconds"] = float(
                time.perf_counter() - metrics_timer_start
            )

            if args.output_mode == "full" or args.save_evaluation_arrays:
                if args.output_mode == "full":
                    case_metrics_dir = (
                        output_dir / "metrics" / "cases" / projection_path.stem
                    )
                    arrays_path = case_metrics_dir / "evaluation_arrays.npz"
                else:
                    case_metrics_dir = output_dir / "voxel_masks"
                    arrays_path = case_metrics_dir / f"{projection_path.stem}.npz"
                case_metrics_dir.mkdir(parents=True, exist_ok=True)
                if not args.save_ssim_map:
                    arrays.pop("ssim_map_valid_zyx")
                np.savez_compressed(arrays_path, **arrays)
                record["evaluation_arrays"] = str(arrays_path)
            if args.output_mode == "full":
                write_json(case_metrics_dir / "metrics.json", record)

            case_cache_removed = False
            if (
                args.output_mode == "json-only"
                and should_train
                and cache_created_by_run
                and not args.keep_case_cache
                and case_output_dir is not None
            ):
                remove_temporary_case_cache(
                    case_output_dir,
                    output_dir / "cases",
                )
                case_cache_removed = True
            record["case_cache_removed"] = case_cache_removed
            record["case_wall_time_seconds"] = float(
                time.perf_counter() - case_timer_start
            )
            print(
                f"  Dice={float(record['masked_dice_3d']):.6f} "
                f"(prediction {record['prediction_threshold_mode']} "
                f"threshold={float(record['prediction_threshold']):.7g}) "
                f"SSIM={float(record['ssim_3d']):.6f} "
                f"masked SSIM={float(record['masked_ssim_3d']):.6f} "
                f"case time={float(record['case_wall_time_seconds']):.1f}s",
                flush=True,
            )
        except Exception as error:
            failed = True
            record["error"] = f"{type(error).__name__}: {error}"
            if (
                args.output_mode == "json-only"
                and should_train
                and cache_created_by_run
                and not args.keep_case_cache
                and case_output_dir is not None
            ):
                try:
                    remove_temporary_case_cache(
                        case_output_dir,
                        output_dir / "cases",
                    )
                    record["case_cache_removed"] = True
                except Exception as cleanup_error:
                    record["cache_cleanup_error"] = (
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
            record["case_wall_time_seconds"] = float(
                time.perf_counter() - case_timer_start
            )
            print(f"  ERROR: {record['error']}", file=sys.stderr, flush=True)
            records.append(record)
            write_aggregate_reports(
                output_dir,
                records,
                args.split,
                len(references),
                output_mode=args.output_mode,
                evaluation_config=evaluation_config,
            )
            if not args.continue_on_error:
                return 1
            continue

        records.append(record)
        write_aggregate_reports(
            output_dir,
            records,
            args.split,
            len(references),
            output_mode=args.output_mode,
            evaluation_config=evaluation_config,
        )
        timing = summarise_timings(records, len(references))
        average_case_seconds = float(timing["average_case_seconds"])
        remaining_seconds = float(timing["estimated_remaining_seconds"])
        print(
            f"  Rolling average={average_case_seconds:.1f}s/case; "
            f"estimated remaining={remaining_seconds / 3600.0:.2f}h",
            flush=True,
        )

    cache_root = output_dir / "cases"
    if args.output_mode == "json-only" and cache_root.is_dir():
        try:
            cache_root.rmdir()
        except OSError:
            pass
    if args.output_mode == "json-only":
        final_path = output_dir / "evaluation_results.json"
    else:
        final_path = output_dir / "metrics" / "summary_metrics.json"
    print(f"Saved final evaluation JSON: {final_path}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
