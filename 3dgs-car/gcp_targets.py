"""Ground-truth and ray targets for the monocular Gaussian Center Predictor.

Coordinate conventions are deliberately explicit:

* Input ImageCAS ``vol`` arrays are stored as ``[X, Y, Z]`` and are converted
  to NumPy/PyTorch ``[Z, Y, X]`` volumes on load.
* Projection geometry is one ASTRA ``cone_vec`` row in metres, ordered as
  ``source_xyz, detector_centre_xyz, detector_u_xyz, detector_v_xyz``. ``u``
  advances one detector column and ``v`` advances one detector row.
* Point clouds and ray endpoints are returned as normalized ``[Z, Y, X]``
  coordinates.  Zero and one are the physical cube faces at
  ``-volume_extent_m / 2`` and ``+volume_extent_m / 2`` respectively.
* A depth is the fraction from the ray's cube entry point to its cube exit
  point.  Background depth is one; ``depth_mask`` disambiguates a real hit at
  the exit face from background.

These conventions match the repository's ZYX volumes and normalized Gaussian
centres while retaining ASTRA's XYZ camera convention at the API boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np


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


@dataclass(frozen=True)
class GroundTruthVolume:
    """A safely loaded ground-truth mask and its physical metadata."""

    path: Path
    volume_zyx: np.ndarray
    spacing_xyz_m: np.ndarray
    volume_key: str
    source_axis_order: str


def _squeeze_numeric_volume(value: np.ndarray, source: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "biuf":
        raise TypeError(f"{source} must be numeric, got dtype {array.dtype}.")
    array = np.squeeze(array)
    if array.ndim != 3:
        raise ValueError(f"{source} must reduce to a 3D array, got {array.shape}.")
    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"{source} contains NaN or infinity.")
    return np.ascontiguousarray(array)


def _choose_volume_key(
    archive: np.lib.npyio.NpzFile, path: Path, volume_key: Optional[str],
) -> str:
    if volume_key is not None:
        if volume_key not in archive.files:
            raise KeyError(
                f"{path} has no volume key {volume_key!r}; available keys: "
                f"{archive.files}."
            )
        return volume_key

    case_insensitive = {name.lower(): name for name in archive.files}
    for candidate in VOLUME_KEY_CANDIDATES:
        if candidate.lower() in case_insensitive:
            return case_insensitive[candidate.lower()]

    compatible = []
    for name in archive.files:
        value = np.asarray(archive[name])
        if value.dtype.kind in "biuf" and np.squeeze(value).ndim == 3:
            compatible.append(name)
    if len(compatible) != 1:
        raise KeyError(
            f"Could not identify exactly one 3D volume in {path}; candidates are "
            f"{compatible}. Pass volume_key explicitly."
        )
    return compatible[0]


def load_ground_truth_npz(
    path: str | Path,
    volume_key: Optional[str] = None,
    axis_order: str = "auto",
    spacing_key: str = "spacing",
    spacing_units: str = "mm",
) -> GroundTruthVolume:
    """Load a GT NPZ as a finite ZYX volume with XYZ spacing in metres.

    With ``axis_order='auto'``, the ImageCAS key ``vol`` is interpreted as XYZ;
    other historical keys are interpreted as ZYX.  Explicit ``xyz`` or ``zyx``
    should be used for a dataset that follows a different convention.
    """

    npz_path = Path(path).expanduser().resolve()
    if axis_order not in {"auto", "xyz", "zyx"}:
        raise ValueError("axis_order must be 'auto', 'xyz', or 'zyx'.")
    units = str(spacing_units).strip().lower()
    if units not in {"m", "mm"}:
        raise ValueError("spacing_units must be 'm' or 'mm'.")

    with np.load(npz_path, allow_pickle=False) as archive:
        selected_key = _choose_volume_key(archive, npz_path, volume_key)
        source_order = (
            ("xyz" if selected_key.lower() == "vol" else "zyx")
            if axis_order == "auto"
            else axis_order
        )
        volume = _squeeze_numeric_volume(
            archive[selected_key], f"{npz_path}:{selected_key}"
        )
        if spacing_key not in archive.files:
            raise KeyError(f"{npz_path} has no required spacing key {spacing_key!r}.")
        spacing_stored = np.asarray(archive[spacing_key], dtype=np.float64).reshape(-1)

    if spacing_stored.shape != (3,) or not np.isfinite(spacing_stored).all():
        raise ValueError(
            f"{npz_path}:{spacing_key} must contain three finite values, got "
            f"{spacing_stored}."
        )
    if np.any(spacing_stored <= 0.0):
        raise ValueError(f"{npz_path}:{spacing_key} must be strictly positive.")

    if source_order == "xyz":
        volume = np.transpose(volume, (2, 1, 0))
        spacing_xyz = spacing_stored
    else:
        spacing_xyz = spacing_stored[::-1]
    spacing_xyz_m = spacing_xyz * (1.0e-3 if units == "mm" else 1.0)

    return GroundTruthVolume(
        path=npz_path,
        volume_zyx=np.ascontiguousarray(volume, dtype=np.float32),
        spacing_xyz_m=np.ascontiguousarray(spacing_xyz_m, dtype=np.float64),
        volume_key=selected_key,
        source_axis_order=source_order,
    )


def _shape_zyx(shape: int | Sequence[int], name: str) -> Tuple[int, int, int]:
    if isinstance(shape, (int, np.integer)):
        result = (int(shape),) * 3
    else:
        result = tuple(int(value) for value in shape)
    if len(result) != 3 or any(value < 2 for value in result):
        raise ValueError(f"{name} must contain three dimensions >= 2, got {result}.")
    return result


def _resample_axis_nearest_zero(
    volume: np.ndarray, source_coordinates: np.ndarray, axis: int
) -> np.ndarray:
    size = int(volume.shape[axis])
    source_coordinates = np.asarray(source_coordinates, dtype=np.float64).reshape(-1)
    indices = np.floor(source_coordinates + 0.5).astype(np.int64)
    valid = (indices >= 0) & (indices < size)
    sampled = np.take(volume, np.clip(indices, 0, size - 1), axis=axis)
    broadcast_shape = [1, 1, 1]
    broadcast_shape[axis] = int(indices.size)
    return np.ascontiguousarray(
        sampled * valid.reshape(broadcast_shape), dtype=np.float32
    )


def resample_volume_to_centered_cube(
    volume_zyx: np.ndarray,
    spacing_xyz_m: Sequence[float],
    output_shape_zyx: int | Sequence[int],
    volume_extent_m: float,
    projection_center_offset_xyz_m: Sequence[float],
    origin_xyz_m: Sequence[float] = (0.0, 0.0, 0.0),
    direction_sign_xyz: Sequence[int] = (1, 1, 1),
) -> np.ndarray:
    """Nearest-neighbour sample a physical GT mask on a centered cubic grid.

    The projection-data builder applies ``centered = world - center_offset``.
    Consequently a centered output coordinate ``q`` samples the source volume
    at ``q + center_offset``.  Output coordinates include both cube endpoints,
    matching ``torch.linspace`` in the reconstruction code.
    """

    volume = _squeeze_numeric_volume(volume_zyx, "volume_zyx")
    output_shape = _shape_zyx(output_shape_zyx, "output_shape_zyx")
    extent = float(volume_extent_m)
    if not np.isfinite(extent) or extent <= 0.0:
        raise ValueError("volume_extent_m must be finite and positive.")

    spacing = np.asarray(spacing_xyz_m, dtype=np.float64).reshape(-1)
    center = np.asarray(projection_center_offset_xyz_m, dtype=np.float64).reshape(-1)
    origin = np.asarray(origin_xyz_m, dtype=np.float64).reshape(-1)
    signs = np.asarray(direction_sign_xyz, dtype=np.float64).reshape(-1)
    for label, value in (
        ("spacing_xyz_m", spacing),
        ("projection_center_offset_xyz_m", center),
        ("origin_xyz_m", origin),
        ("direction_sign_xyz", signs),
    ):
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError(f"{label} must contain three finite XYZ values.")
    if np.any(spacing <= 0.0):
        raise ValueError("spacing_xyz_m must be strictly positive.")
    if not np.all(np.isin(signs, (-1.0, 1.0))):
        raise ValueError("direction_sign_xyz values must be -1 or +1.")

    output_shape_xyz = output_shape[::-1]
    source_coordinates_xyz = []
    for axis, output_size in enumerate(output_shape_xyz):
        centered_coordinates = np.linspace(
            -extent / 2.0, extent / 2.0, output_size, dtype=np.float64
        )
        world_coordinates = centered_coordinates + center[axis]
        source_coordinates_xyz.append(
            (world_coordinates - origin[axis]) / (signs[axis] * spacing[axis])
        )

    resampled = volume
    source_coordinates_zyx = source_coordinates_xyz[::-1]
    # Reducing X and Y before Z limits peak memory for 512x512x275 masks.
    for axis in (2, 1, 0):
        resampled = _resample_axis_nearest_zero(
            resampled, source_coordinates_zyx[axis], axis
        )
    return np.ascontiguousarray(resampled, dtype=np.float32)


def volume_to_normalized_points(
    volume_zyx: np.ndarray,
    threshold: float = 0.5,
    max_points: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Return foreground voxel centres as normalized ZYX points in [0, 1]."""

    volume = _squeeze_numeric_volume(volume_zyx, "volume_zyx")
    indices = np.argwhere(volume > float(threshold))
    if max_points is not None:
        limit = int(max_points)
        if limit <= 0:
            raise ValueError("max_points must be positive when provided.")
        if indices.shape[0] > limit:
            generator = rng if rng is not None else np.random.default_rng(0)
            selection = generator.choice(indices.shape[0], size=limit, replace=False)
            indices = indices[np.sort(selection)]
    scale = np.asarray(volume.shape, dtype=np.float32) - 1.0
    points = indices.astype(np.float32) / scale[None, :]
    return np.ascontiguousarray(points, dtype=np.float32)


def scale_cone_vector_for_detector(
    cone_vector: np.ndarray, original_shape: Sequence[int], target_shape: Sequence[int],
) -> np.ndarray:
    """Scale ASTRA detector steps while preserving the detector field of view."""

    vector = np.asarray(cone_vector, dtype=np.float64).reshape(-1)
    if vector.shape != (12,) or not np.isfinite(vector).all():
        raise ValueError("cone_vector must contain 12 finite values.")
    original = tuple(int(value) for value in original_shape)
    target = tuple(int(value) for value in target_shape)
    if len(original) != 2 or len(target) != 2 or min(*original, *target) <= 0:
        raise ValueError("original_shape and target_shape must be positive (H, W).")
    scaled = vector.copy()
    scaled[6:9] *= float(original[1]) / float(target[1])
    scaled[9:12] *= float(original[0]) / float(target[0])
    return np.ascontiguousarray(scaled, dtype=np.float32)


def detector_ray_box_intersections(
    cone_vector: np.ndarray, detector_shape: Sequence[int], volume_extent_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Intersect every ASTRA cone-vector detector ray with the centered cube.

    Returns ``(entry_zyx, exit_zyx, valid_mask)`` with shapes ``[H,W,3]``,
    ``[H,W,3]``, and ``[H,W]``. Endpoints are normalized to [0,1].
    """

    vector = np.asarray(cone_vector, dtype=np.float64).reshape(-1)
    if vector.shape != (12,) or not np.isfinite(vector).all():
        raise ValueError("cone_vector must contain 12 finite values.")
    shape = tuple(int(value) for value in detector_shape)
    if len(shape) != 2 or any(value <= 0 for value in shape):
        raise ValueError(f"detector_shape must be positive (H,W), got {shape}.")
    extent = float(volume_extent_m)
    if not np.isfinite(extent) or extent <= 0.0:
        raise ValueError("volume_extent_m must be finite and positive.")

    height, width = shape
    source = vector[0:3]
    detector = vector[3:6]
    u_step = vector[6:9]
    v_step = vector[9:12]
    columns = np.arange(width, dtype=np.float64) - (width - 1.0) / 2.0
    rows = np.arange(height, dtype=np.float64) - (height - 1.0) / 2.0
    pixels = (
        detector[None, None, :]
        + columns[None, :, None] * u_step[None, None, :]
        + rows[:, None, None] * v_step[None, None, :]
    )
    directions = pixels - source[None, None, :]

    half_extent = extent / 2.0
    near = np.full((height, width), -np.inf, dtype=np.float64)
    far = np.full((height, width), np.inf, dtype=np.float64)
    invalid = np.zeros((height, width), dtype=bool)
    epsilon = np.finfo(np.float64).eps * 16.0
    for axis in range(3):
        direction = directions[..., axis]
        parallel = np.abs(direction) <= epsilon
        if parallel.any() and not (-half_extent <= source[axis] <= half_extent):
            invalid |= parallel
        nonparallel = ~parallel
        if nonparallel.any():
            first = np.empty_like(direction)
            second = np.empty_like(direction)
            first[nonparallel] = (-half_extent - source[axis]) / direction[nonparallel]
            second[nonparallel] = (half_extent - source[axis]) / direction[nonparallel]
            axis_near = np.minimum(first[nonparallel], second[nonparallel])
            axis_far = np.maximum(first[nonparallel], second[nonparallel])
            near[nonparallel] = np.maximum(near[nonparallel], axis_near)
            far[nonparallel] = np.minimum(far[nonparallel], axis_far)

    near = np.maximum(near, 0.0)
    valid = (~invalid) & np.isfinite(near) & np.isfinite(far) & (far >= near)
    safe_near = np.where(valid, near, 0.0)
    safe_far = np.where(valid, far, 0.0)
    entry_xyz = source[None, None, :] + safe_near[..., None] * directions
    exit_xyz = source[None, None, :] + safe_far[..., None] * directions
    entry_xyz[~valid] = 0.0
    exit_xyz[~valid] = 0.0

    entry_zyx = ((entry_xyz + half_extent) / extent)[..., ::-1]
    exit_zyx = ((exit_xyz + half_extent) / extent)[..., ::-1]
    entry_zyx = np.clip(entry_zyx, 0.0, 1.0)
    exit_zyx = np.clip(exit_zyx, 0.0, 1.0)
    return (
        np.ascontiguousarray(entry_zyx, dtype=np.float32),
        np.ascontiguousarray(exit_zyx, dtype=np.float32),
        np.ascontiguousarray(valid),
    )


def raycast_first_hit_nearest(
    volume_zyx: np.ndarray,
    ray_entry_zyx: np.ndarray,
    ray_exit_zyx: np.ndarray,
    ray_valid_mask: Optional[np.ndarray] = None,
    num_samples: Optional[int] = None,
    threshold: float = 0.5,
    chunk_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbour first-hit ray cast through a normalized ZYX volume."""

    volume = _squeeze_numeric_volume(volume_zyx, "volume_zyx")
    entry = np.asarray(ray_entry_zyx, dtype=np.float32)
    exit = np.asarray(ray_exit_zyx, dtype=np.float32)
    if entry.shape != exit.shape or entry.ndim != 3 or entry.shape[-1] != 3:
        raise ValueError("ray_entry_zyx and ray_exit_zyx must share shape [H,W,3].")
    if not np.isfinite(entry).all() or not np.isfinite(exit).all():
        raise ValueError("Ray endpoints contain NaN or infinity.")
    spatial_shape = entry.shape[:2]
    if ray_valid_mask is None:
        valid = np.ones(spatial_shape, dtype=bool)
    else:
        valid = np.asarray(ray_valid_mask, dtype=bool)
        if valid.shape != spatial_shape:
            raise ValueError(
                f"ray_valid_mask must have shape {spatial_shape}, got {valid.shape}."
            )
    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be positive.")

    flat_entry = entry.reshape(-1, 3)
    flat_exit = exit.reshape(-1, 3)
    flat_valid = valid.reshape(-1)
    depth = np.ones(flat_valid.shape, dtype=np.float32)
    hit_mask = np.zeros(flat_valid.shape, dtype=bool)

    voxel_scale = np.asarray(volume.shape, dtype=np.float32) - 1.0
    if num_samples is None:
        valid_indices = np.flatnonzero(flat_valid)
        if valid_indices.size:
            lengths = np.linalg.norm(
                (flat_exit[valid_indices] - flat_entry[valid_indices])
                * voxel_scale[None, :],
                axis=1,
            )
            sample_count = max(2, int(np.ceil(float(lengths.max()) * 2.0)) + 1)
        else:
            sample_count = 2
    else:
        sample_count = int(num_samples)
        if sample_count < 2:
            raise ValueError("num_samples must be at least 2.")
    fractions = np.linspace(0.0, 1.0, sample_count, dtype=np.float32)

    valid_indices = np.flatnonzero(flat_valid)
    for start in range(0, valid_indices.size, int(chunk_size)):
        indices = valid_indices[start : start + int(chunk_size)]
        starts = flat_entry[indices, None, :]
        differences = (flat_exit[indices] - flat_entry[indices])[:, None, :]
        positions = starts + fractions[None, :, None] * differences
        voxel_indices = np.rint(positions * voxel_scale[None, None, :]).astype(np.int64)
        for axis, size in enumerate(volume.shape):
            np.clip(voxel_indices[..., axis], 0, size - 1, out=voxel_indices[..., axis])
        occupied = volume[
            voxel_indices[..., 0], voxel_indices[..., 1], voxel_indices[..., 2],
        ] > float(threshold)
        has_hit = occupied.any(axis=1)
        if has_hit.any():
            first_hit = occupied.argmax(axis=1)
            hit_indices = indices[has_hit]
            depth[hit_indices] = fractions[first_hit[has_hit]]
            hit_mask[hit_indices] = True

    return (
        np.ascontiguousarray(depth.reshape(spatial_shape), dtype=np.float32),
        np.ascontiguousarray(hit_mask.reshape(spatial_shape)),
    )


def build_view_targets(
    volume_zyx: np.ndarray,
    cone_vector: np.ndarray,
    detector_shape: Sequence[int],
    volume_extent_m: float,
    num_depth_samples: Optional[int] = None,
) -> dict[str, np.ndarray]:
    """Build normalized ray endpoints and a first-hit depth target for one view."""

    entry, exit, valid = detector_ray_box_intersections(
        cone_vector=cone_vector,
        detector_shape=detector_shape,
        volume_extent_m=volume_extent_m,
    )
    depth, depth_mask = raycast_first_hit_nearest(
        volume_zyx=volume_zyx,
        ray_entry_zyx=entry,
        ray_exit_zyx=exit,
        ray_valid_mask=valid,
        num_samples=num_depth_samples,
    )
    return {
        "depth": depth,
        "depth_mask": depth_mask,
        "ray_valid_mask": valid,
        "ray_entry_zyx": entry,
        "ray_exit_zyx": exit,
    }


__all__ = [
    "GroundTruthVolume",
    "VOLUME_KEY_CANDIDATES",
    "build_view_targets",
    "detector_ray_box_intersections",
    "load_ground_truth_npz",
    "raycast_first_hit_nearest",
    "resample_volume_to_centered_cube",
    "scale_cone_vector_for_detector",
    "volume_to_normalized_points",
]
