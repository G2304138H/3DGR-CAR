"""Paired projection/volume dataset for Gaussian Center Predictor training.

Each projection view is one monocular sample.  Multiple views of the same case
therefore share an aligned volume and point cloud but have view-specific rays
and first-hit depth targets.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from gcp_targets import (
    build_view_targets,
    load_ground_truth_npz,
    resample_volume_to_centered_cube,
    scale_cone_vector_for_detector,
    volume_to_normalized_points,
)
from stage2_npz_data import (
    DEFAULT_SOURCE_ORIGIN_DISTANCE_M,
    Stage2ProjectionCase,
    load_stage2_projection_case,
)

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # NumPy preprocessing and its tests remain usable without PyTorch.
    torch = None

    class Dataset:  # type: ignore[no-redef]
        pass


@dataclass(frozen=True)
class GCPCasePair:
    """One projection archive paired with its dense ground-truth volume."""

    projection_path: Path
    ground_truth_path: Path
    case_name: str = ""
    vessel_type: Optional[str] = None
    case_id: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "projection_path", Path(self.projection_path).expanduser().resolve()
        )
        object.__setattr__(
            self,
            "ground_truth_path",
            Path(self.ground_truth_path).expanduser().resolve(),
        )


def _npz_scalar(archive: np.lib.npyio.NpzFile, key: str) -> Optional[str]:
    if key not in archive.files:
        return None
    value = np.asarray(archive[key])
    if value.size != 1:
        return None
    return str(value.reshape(()).item()).strip()


def _projection_identity(path: Path) -> tuple[str, Optional[str], Optional[str]]:
    with np.load(path, allow_pickle=False) as archive:
        if "images" not in archive.files:
            raise KeyError(f"{path} is not a projection archive: missing 'images'.")
        images = np.asarray(archive["images"])
        if images.ndim != 3:
            raise ValueError(f"{path}:images must have shape [V,H,W].")
        case_name = _npz_scalar(archive, "sample_name") or path.stem
        vessel_type = _npz_scalar(archive, "vessel_type")
        case_id = _npz_scalar(archive, "case_id")
    if case_id is None:
        match = re.search(r"([0-9]+)$", case_name)
        case_id = match.group(1) if match else None
    if vessel_type is None:
        match = re.match(r"([a-zA-Z]+)[_-]", case_name)
        vessel_type = match.group(1).lower() if match else None
    return case_name, vessel_type, case_id


def _normalise_identifier(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _npz_paths(path: str | Path) -> list[Path]:
    root = Path(path).expanduser().resolve()
    if root.is_file():
        if root.suffix.lower() != ".npz":
            raise ValueError(f"Expected an NPZ path, got {root}.")
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(root)
    return sorted(candidate.resolve() for candidate in root.rglob("*.npz"))


def _case_id_aliases(case_id: Optional[str]) -> set[str]:
    if case_id is None:
        return set()
    aliases = {_normalise_identifier(case_id)}
    if str(case_id).isdigit():
        aliases.add(str(int(case_id)))
    return aliases


def _resolve_ground_truth_path(
    ground_truth_paths: Sequence[Path],
    ground_truth_root: Path,
    sample_name: str,
    vessel_type: Optional[str],
    case_id: Optional[str],
) -> Path:
    """Resolve a volume deterministically, preferring vessel/case directory layout."""

    id_aliases = _case_id_aliases(case_id)
    sample_key = _normalise_identifier(sample_name)
    vessel_key = _normalise_identifier(vessel_type or "")
    ranked: list[tuple[int, Path]] = []
    for candidate in ground_truth_paths:
        stem_key = _normalise_identifier(candidate.stem)
        parent_key = _normalise_identifier(candidate.parent.name)
        score: Optional[int] = None
        if (
            vessel_key
            and id_aliases
            and parent_key == vessel_key
            and stem_key in id_aliases
        ):
            score = 0
        elif (
            id_aliases
            and candidate.parent == ground_truth_root
            and stem_key in id_aliases
        ):
            score = 1
        elif stem_key == sample_key:
            score = 2
        elif id_aliases and stem_key in id_aliases:
            score = 3
        if score is not None:
            ranked.append((score, candidate))

    if not ranked:
        raise FileNotFoundError(
            f"No ground-truth NPZ matches projection case {sample_name!r} "
            f"(vessel_type={vessel_type!r}, case_id={case_id!r})."
        )
    best_score = min(score for score, _ in ranked)
    best = sorted(path for score, path in ranked if score == best_score)
    if len(best) != 1:
        raise RuntimeError(
            f"Ambiguous ground truth for {sample_name!r}: "
            f"{[str(path) for path in best]}."
        )
    return best[0]


def discover_case_pairs(
    projection_dir: str | Path,
    ground_truth_dir: str | Path,
    case_names: Optional[Iterable[str]] = None,
) -> list[GCPCasePair]:
    """Discover projection/GT pairs from file or directory roots.

    Filtering accepts a projection's ``sample_name``, filename stem, or
    ``case_id``.  Pairing first prefers ``GT_ROOT/vessel_type/case_id.npz``,
    then ``GT_ROOT/case_id.npz``, then exact sample-name and recursive ID
    matches. Missing and ambiguous matches are errors rather than silent drops.
    """

    projection_paths = _npz_paths(projection_dir)
    ground_truth_paths = _npz_paths(ground_truth_dir)
    ground_truth_root = Path(ground_truth_dir).expanduser().resolve()
    if ground_truth_root.is_file():
        ground_truth_root = ground_truth_root.parent
    requested = (
        {_normalise_identifier(name) for name in case_names}
        if case_names is not None
        else None
    )
    pairs: list[GCPCasePair] = []
    matched_filters: set[str] = set()

    for projection_path in projection_paths:
        try:
            sample_name, vessel_type, case_id = _projection_identity(projection_path)
        except KeyError:
            # A shared directory may contain GT archives as well as projections.
            continue
        identifiers = {
            _normalise_identifier(sample_name),
            _normalise_identifier(projection_path.stem),
        } | _case_id_aliases(case_id)
        if requested is not None and identifiers.isdisjoint(requested):
            continue
        if requested is not None:
            matched_filters.update(identifiers & requested)
        ground_truth_path = _resolve_ground_truth_path(
            ground_truth_paths=ground_truth_paths,
            ground_truth_root=ground_truth_root,
            sample_name=sample_name,
            vessel_type=vessel_type,
            case_id=case_id,
        )
        pairs.append(
            GCPCasePair(
                projection_path=projection_path,
                ground_truth_path=ground_truth_path,
                case_name=sample_name,
                vessel_type=vessel_type,
                case_id=case_id,
            )
        )

    if requested is not None:
        missing = sorted(requested - matched_filters)
        if missing:
            raise KeyError(f"Requested cases were not found: {missing}.")
    if not pairs:
        raise ValueError("No paired projection cases were discovered.")
    names = [pair.case_name for pair in pairs]
    if len(set(names)) != len(names):
        raise ValueError(f"Projection sample names must be unique, got {names}.")
    return sorted(pairs, key=lambda pair: pair.case_name)


def resize_projection_bilinear(
    image: np.ndarray, output_shape: int | Sequence[int]
) -> np.ndarray:
    """Resize one 2D image with PyTorch-compatible half-pixel bilinear sampling."""

    values = np.asarray(image, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError(f"image must be a finite 2D array, got {values.shape}.")
    if isinstance(output_shape, (int, np.integer)):
        target = (int(output_shape), int(output_shape))
    else:
        target = tuple(int(value) for value in output_shape)
    if len(target) != 2 or any(value <= 0 for value in target):
        raise ValueError(f"output_shape must be positive (H,W), got {target}.")
    if tuple(values.shape) == target:
        return np.ascontiguousarray(values, dtype=np.float32)

    input_height, input_width = values.shape
    output_height, output_width = target
    source_y = (
        np.arange(output_height, dtype=np.float64) + 0.5
    ) * input_height / output_height - 0.5
    source_x = (
        np.arange(output_width, dtype=np.float64) + 0.5
    ) * input_width / output_width - 0.5
    y0_unclipped = np.floor(source_y).astype(np.int64)
    x0_unclipped = np.floor(source_x).astype(np.int64)
    y_weight = (source_y - y0_unclipped).astype(np.float32)
    x_weight = (source_x - x0_unclipped).astype(np.float32)
    y0 = np.clip(y0_unclipped, 0, input_height - 1)
    x0 = np.clip(x0_unclipped, 0, input_width - 1)
    y1 = np.clip(y0_unclipped + 1, 0, input_height - 1)
    x1 = np.clip(x0_unclipped + 1, 0, input_width - 1)

    upper = (
        values[y0[:, None], x0[None, :]] * (1.0 - x_weight)[None, :]
        + values[y0[:, None], x1[None, :]] * x_weight[None, :]
    )
    lower = (
        values[y1[:, None], x0[None, :]] * (1.0 - x_weight)[None, :]
        + values[y1[:, None], x1[None, :]] * x_weight[None, :]
    )
    resized = upper * (1.0 - y_weight)[:, None] + lower * y_weight[:, None]
    return np.ascontiguousarray(resized, dtype=np.float32)


class PairedGCPDataset(Dataset):
    """Index every projection view in paired cases as a monocular GCP sample."""

    def __init__(
        self,
        pairs: Sequence[GCPCasePair],
        volume_size: int = 128,
        image_size: int = 128,
        volume_extent_m: Optional[float] = None,
        downsample_factor: int = 2,
        cache_dir: Optional[str | Path] = None,
        max_points: Optional[int] = None,
        source_origin_distance_m: float = DEFAULT_SOURCE_ORIGIN_DISTANCE_M,
    ) -> None:
        if not pairs:
            raise ValueError("pairs must contain at least one GCPCasePair.")
        self.pairs = tuple(pairs)
        self.volume_size = int(volume_size)
        self.image_size = int(image_size)
        self.volume_extent_m = (
            None if volume_extent_m is None else float(volume_extent_m)
        )
        self.downsample_factor = int(downsample_factor)
        self.cache_dir = (
            None if cache_dir is None else Path(cache_dir).expanduser().resolve()
        )
        self.max_points = None if max_points is None else int(max_points)
        self.source_origin_distance_m = float(source_origin_distance_m)
        if self.volume_size < 2 or self.image_size <= 0:
            raise ValueError("volume_size must be >=2 and image_size must be positive.")
        if self.downsample_factor <= 0 or self.image_size % self.downsample_factor != 0:
            raise ValueError("downsample_factor must divide image_size exactly.")
        if self.volume_extent_m is not None and (
            not np.isfinite(self.volume_extent_m) or self.volume_extent_m <= 0.0
        ):
            raise ValueError("volume_extent_m must be finite and positive.")
        if self.max_points is not None and self.max_points <= 0:
            raise ValueError("max_points must be positive when provided.")
        if (
            not np.isfinite(self.source_origin_distance_m)
            or self.source_origin_distance_m <= 0.0
        ):
            raise ValueError("source_origin_distance_m must be finite and positive.")
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._sample_index: list[tuple[int, int]] = []
        for pair_index, pair in enumerate(self.pairs):
            if not pair.projection_path.is_file():
                raise FileNotFoundError(pair.projection_path)
            if not pair.ground_truth_path.is_file():
                raise FileNotFoundError(pair.ground_truth_path)
            with np.load(pair.projection_path, allow_pickle=False) as archive:
                if "images" not in archive.files:
                    raise KeyError(f"{pair.projection_path} has no 'images' key.")
                images = np.asarray(archive["images"])
                if images.ndim != 3 or images.shape[0] <= 0:
                    raise ValueError(
                        f"{pair.projection_path}:images must have shape [V,H,W]."
                    )
                number_of_views = int(images.shape[0])
            self._sample_index.extend(
                (pair_index, view_index) for view_index in range(number_of_views)
            )

        self._projection_lru: OrderedDict[int, Stage2ProjectionCase] = OrderedDict()
        self._aligned_lru: OrderedDict[int, tuple[np.ndarray, np.ndarray, float]] = (
            OrderedDict()
        )

    @property
    def predictor_shape(self) -> tuple[int, int]:
        size = self.image_size // self.downsample_factor
        return size, size

    def __len__(self) -> int:
        return len(self._sample_index)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_projection_lru"] = OrderedDict()
        state["_aligned_lru"] = OrderedDict()
        return state

    @staticmethod
    def _lru_insert(cache: OrderedDict, key: int, value: Any, capacity: int = 2) -> Any:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > capacity:
            cache.popitem(last=False)
        return value

    def _projection_case(self, pair_index: int) -> Stage2ProjectionCase:
        if pair_index in self._projection_lru:
            case = self._projection_lru[pair_index]
            self._projection_lru.move_to_end(pair_index)
            return case
        case = load_stage2_projection_case(
            str(self.pairs[pair_index].projection_path),
            source_origin_distance_m=self.source_origin_distance_m,
        )
        return self._lru_insert(self._projection_lru, pair_index, case)

    def _aligned_case(self, pair_index: int) -> tuple[np.ndarray, np.ndarray, float]:
        if pair_index in self._aligned_lru:
            value = self._aligned_lru[pair_index]
            self._aligned_lru.move_to_end(pair_index)
            return value
        case = self._projection_case(pair_index)
        loaded = load_ground_truth_npz(self.pairs[pair_index].ground_truth_path)
        extent = (
            float(self.volume_extent_m)
            if self.volume_extent_m is not None
            else float(case.isocenter_fov_m)
        )
        center_offset = (
            np.zeros(3, dtype=np.float64)
            if case.projection_center_offset_m is None
            else np.asarray(case.projection_center_offset_m, dtype=np.float64)
        )
        aligned = resample_volume_to_centered_cube(
            volume_zyx=loaded.volume_zyx,
            spacing_xyz_m=loaded.spacing_xyz_m,
            output_shape_zyx=self.volume_size,
            volume_extent_m=extent,
            projection_center_offset_xyz_m=center_offset,
        )
        aligned = np.ascontiguousarray(aligned > 0.5, dtype=np.float32)
        seed_material = str(self.pairs[pair_index].projection_path).encode("utf-8")
        seed = int.from_bytes(
            hashlib.blake2b(seed_material, digest_size=8).digest(), "little"
        )
        points = volume_to_normalized_points(
            aligned, max_points=self.max_points, rng=np.random.default_rng(seed),
        )
        if points.shape[0] == 0:
            raise ValueError(
                f"Aligned ground truth for {self.pairs[pair_index].case_name!r} "
                "contains no foreground voxels. Check extent, spacing, and offset."
            )
        value = (aligned, points, extent)
        return self._lru_insert(self._aligned_lru, pair_index, value)

    def _cache_path(self, pair_index: int, view_index: int) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        pair = self.pairs[pair_index]
        payload = {
            "schema": 1,
            "projection": str(pair.projection_path),
            "projection_stat": [
                pair.projection_path.stat().st_size,
                pair.projection_path.stat().st_mtime_ns,
            ],
            "ground_truth": str(pair.ground_truth_path),
            "ground_truth_stat": [
                pair.ground_truth_path.stat().st_size,
                pair.ground_truth_path.stat().st_mtime_ns,
            ],
            "view_index": int(view_index),
            "volume_size": self.volume_size,
            "image_size": self.image_size,
            "volume_extent_m": self.volume_extent_m,
            "downsample_factor": self.downsample_factor,
            "max_points": self.max_points,
            "source_origin_distance_m": self.source_origin_distance_m,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()[:20]
        safe_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", pair.case_name or "case")
        return self.cache_dir / f"{safe_name}_view{view_index:03d}_{digest}.npz"

    @staticmethod
    def _load_cached(path: Path) -> Optional[dict[str, np.ndarray]]:
        if not path.is_file():
            return None
        try:
            with np.load(path, allow_pickle=False) as archive:
                required = {
                    "image",
                    "volume",
                    "point_cloud",
                    "depth",
                    "depth_mask",
                    "ray_valid_mask",
                    "ray_entry_zyx",
                    "ray_exit_zyx",
                    "cone_vector",
                    "volume_extent_m",
                }
                if not required.issubset(archive.files):
                    return None
                return {key: np.ascontiguousarray(archive[key]) for key in required}
        except (OSError, ValueError, EOFError):
            return None

    @staticmethod
    def _write_cached(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_name: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=path.stem + ".", suffix=".npz", dir=path.parent, delete=False,
            ) as temporary:
                temporary_name = temporary.name
            np.savez_compressed(temporary_name, **arrays)
            os.replace(temporary_name, path)
            temporary_name = None
        finally:
            if temporary_name is not None:
                try:
                    os.unlink(temporary_name)
                except FileNotFoundError:
                    pass

    def _sample_numpy(self, pair_index: int, view_index: int) -> dict[str, np.ndarray]:
        cache_path = self._cache_path(pair_index, view_index)
        if cache_path is not None:
            cached = self._load_cached(cache_path)
            if cached is not None:
                return cached

        case = self._projection_case(pair_index)
        aligned, points, extent = self._aligned_case(pair_index)
        image = resize_projection_bilinear(
            case.images[view_index], (self.image_size, self.image_size)
        )
        original_vector = case.cone_vectors([view_index])[0]
        predictor_vector = scale_cone_vector_for_detector(
            original_vector,
            original_shape=case.detector_shape,
            target_shape=self.predictor_shape,
        )
        view_targets = build_view_targets(
            volume_zyx=aligned,
            cone_vector=predictor_vector,
            detector_shape=self.predictor_shape,
            volume_extent_m=extent,
        )
        arrays = {
            "image": image[None, ...].astype(np.float32, copy=False),
            "volume": aligned[None, ...].astype(np.float32, copy=False),
            "point_cloud": points.astype(np.float32, copy=False),
            "depth": view_targets["depth"][None, ...].astype(np.float32, copy=False),
            "depth_mask": view_targets["depth_mask"][None, ...].astype(
                bool, copy=False
            ),
            "ray_valid_mask": view_targets["ray_valid_mask"][None, ...].astype(
                bool, copy=False
            ),
            "ray_entry_zyx": view_targets["ray_entry_zyx"].astype(
                np.float32, copy=False
            ),
            "ray_exit_zyx": view_targets["ray_exit_zyx"].astype(np.float32, copy=False),
            "cone_vector": predictor_vector.astype(np.float32, copy=False),
            "volume_extent_m": np.asarray(extent, dtype=np.float32),
        }
        arrays = {key: np.ascontiguousarray(value) for key, value in arrays.items()}
        if cache_path is not None:
            self._write_cached(cache_path, arrays)
        return arrays

    def __getitem__(self, index: int) -> dict[str, Any]:
        pair_index, view_index = self._sample_index[int(index)]
        pair = self.pairs[pair_index]
        case = self._projection_case(pair_index)
        arrays = self._sample_numpy(pair_index, view_index)
        numeric: dict[str, Any]
        if torch is None:
            numeric = arrays
        else:
            numeric = {
                key: torch.from_numpy(
                    value.copy() if not value.flags.writeable else value
                )
                for key, value in arrays.items()
            }
        numeric.update(
            {
                "case_name": pair.case_name or case.sample_name,
                "view_index": int(view_index),
                "clinical_view": str(case.clinical_views[view_index]),
                "projection_path": str(pair.projection_path),
                "ground_truth_path": str(pair.ground_truth_path),
                "vessel_type": pair.vessel_type or "",
                "case_id": pair.case_id or "",
            }
        )
        return numeric


def gcp_collate(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Stack fixed fields and pad variable point clouds with ``point_mask``."""

    if not batch:
        raise ValueError("Cannot collate an empty batch.")
    point_counts = [int(sample["point_cloud"].shape[0]) for sample in batch]
    maximum_points = max(point_counts)
    if torch is not None and isinstance(batch[0]["point_cloud"], torch.Tensor):
        reference = batch[0]["point_cloud"]
        points = reference.new_zeros((len(batch), maximum_points, 3))
        point_mask = torch.zeros(
            (len(batch), maximum_points), dtype=torch.bool, device=reference.device
        )
        for batch_index, (sample, count) in enumerate(zip(batch, point_counts)):
            points[batch_index, :count] = sample["point_cloud"]
            point_mask[batch_index, :count] = True

        def stack(values: Sequence[Any], axis: int = 0) -> Any:
            return torch.stack(tuple(values), dim=axis)

    else:
        points = np.zeros((len(batch), maximum_points, 3), dtype=np.float32)
        point_mask = np.zeros((len(batch), maximum_points), dtype=bool)
        for batch_index, (sample, count) in enumerate(zip(batch, point_counts)):
            points[batch_index, :count] = np.asarray(sample["point_cloud"])
            point_mask[batch_index, :count] = True

        def stack(values: Sequence[Any], axis: int = 0) -> Any:
            return np.stack(tuple(values), axis=axis)

    result: dict[str, Any] = {"point_cloud": points, "point_mask": point_mask}
    metadata_keys = {
        "case_name",
        "view_index",
        "clinical_view",
        "projection_path",
        "ground_truth_path",
        "vessel_type",
        "case_id",
    }
    for key in batch[0]:
        if key == "point_cloud":
            continue
        if key in metadata_keys:
            result[key] = [sample[key] for sample in batch]
        else:
            result[key] = stack([sample[key] for sample in batch], axis=0)
    return result


__all__ = [
    "GCPCasePair",
    "PairedGCPDataset",
    "discover_case_pairs",
    "gcp_collate",
    "resize_projection_bilinear",
]
