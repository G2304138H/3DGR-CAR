"""Render a reconstructed density volume with the Stage-2 monitor camera.

The camera sequence, shared GT/prediction centering, equal-axis framing, figure
size, DPI, and GIF timing mirror ``methods/src/visualization.py``'s
``save_3d_overlay_gif``.  The reference artery is used only to establish the
shared transform and axes; the GIF itself draws only the reconstructed volume
isosurface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np


MONITOR_SURFACE_CIRCLE_POINTS = 24
DEFAULT_VOLUME_GIF_POSITIVE_PERCENTILE = 97.0


def _set_equal_3d_axes(ax: Any, points: np.ndarray) -> None:
    """Copy the monitor overlay's equal-axis calculation exactly."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if pts.size == 0:
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_zlim(-1.0, 1.0)
        return
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = 0.5 * (mins + maxs)
    half = float(max(0.5 * np.max(maxs - mins), 1e-3))
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)


def _center_surface_overlay(
    gt_surfs_raw: list[np.ndarray],
    pred_surfs_raw: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Apply the same shared GT-centred transform as the monitor overlay."""
    if not gt_surfs_raw and not pred_surfs_raw:
        return [], [], np.zeros((0, 3), dtype=np.float32)
    ref_surfs = gt_surfs_raw if gt_surfs_raw else pred_surfs_raw
    ref_points = np.concatenate(
        [np.asarray(surface).reshape(-1, 3) for surface in ref_surfs], axis=0
    )
    ref_points = ref_points[np.all(np.isfinite(ref_points), axis=1)]
    center = (
        ref_points.mean(axis=0)
        if ref_points.size
        else np.zeros((3,), dtype=np.float32)
    )
    gt_surfs = [
        np.asarray(surface, dtype=np.float32) - center for surface in gt_surfs_raw
    ]
    pred_surfs = [
        np.asarray(surface, dtype=np.float32) - center
        for surface in pred_surfs_raw
    ]
    all_surfs = gt_surfs + pred_surfs
    points = (
        np.concatenate([surface.reshape(-1, 3) for surface in all_surfs], axis=0)
        if all_surfs
        else np.zeros((0, 3), dtype=np.float32)
    )
    return gt_surfs, pred_surfs, points


def _estimate_derivatives(centerline: np.ndarray) -> np.ndarray:
    derivatives = np.zeros_like(centerline)
    if centerline.shape[0] < 2:
        return derivatives
    derivatives[0] = centerline[1] - centerline[0]
    derivatives[-1] = centerline[-1] - centerline[-2]
    if centerline.shape[0] > 2:
        derivatives[1:-1] = 0.5 * (centerline[2:] - centerline[:-2])
    return derivatives


def _fallback_normal(tangent: np.ndarray) -> np.ndarray:
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm <= 1.0e-12:
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    tangent_hat = tangent / tangent_norm
    axis = np.eye(3, dtype=np.float64)[int(np.argmin(np.abs(tangent_hat)))]
    normal = axis - float(np.dot(axis, tangent_hat)) * tangent_hat
    normal_norm = float(np.linalg.norm(normal))
    return normal / max(normal_norm, 1.0e-12)


def _build_tube_surface(
    centerline: np.ndarray,
    radius: np.ndarray,
    num_circle_points: int = MONITOR_SURFACE_CIRCLE_POINTS,
) -> np.ndarray:
    """Build the same 24-point tube/end-cap layout used by the monitor."""
    points = np.asarray(centerline, dtype=np.float64)
    radii = np.asarray(radius, dtype=np.float64).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] != radii.size:
        raise ValueError(
            "Expected centerline [N,3] and radius [N], got "
            f"{points.shape} and {radii.shape}."
        )
    derivatives = _estimate_derivatives(points)
    keep = np.flatnonzero(np.sum(np.abs(derivatives), axis=1) != 0.0)
    if keep.size < 2:
        raise ValueError("Reference artery branch has fewer than two valid points.")
    points = points[keep]
    derivatives = derivatives[keep]
    radii = np.clip(radii[keep], 1.0e-5, None)

    angles = np.linspace(0.0, 2.0 * np.pi, int(num_circle_points))
    cos_angles = np.cos(angles)[:, None]
    sin_angles = np.sin(angles)[:, None]
    normal = np.zeros((3,), dtype=np.float64)
    normal[int(np.argmin(np.abs(points[1])))] = 1.0
    rings: list[np.ndarray] = []

    for index in range(points.shape[0]):
        tangent = derivatives[index]
        conormal = np.cross(normal, tangent)
        conormal_norm = float(np.linalg.norm(conormal))
        if conormal_norm <= 1.0e-12:
            normal = _fallback_normal(tangent)
            conormal = np.cross(normal, tangent)
            conormal_norm = float(np.linalg.norm(conormal))
        conormal = conormal / max(conormal_norm, 1.0e-12)
        normal = np.cross(tangent, conormal)
        normal = normal / max(float(np.linalg.norm(normal)), 1.0e-12)

        if index == 0:
            ring_radii: Sequence[float] = np.linspace(
                0.0, radii[index], 50
            )[1:]
        elif index == points.shape[0] - 1:
            ring_radii = np.flip(np.linspace(0.0, radii[index], 50)[1:])
        else:
            ring_radii = (float(radii[index]),)
        for ring_radius in ring_radii:
            ring = (
                points[index][None, :]
                + float(ring_radius) * cos_angles * normal[None, :]
                + float(ring_radius) * sin_angles * conormal[None, :]
            )
            rings.append(ring)
    return np.asarray(rings, dtype=np.float32)


def _reference_surfaces_from_npz(npz_path: Path) -> list[np.ndarray]:
    """Load projected GT artery branches and build 24-point tube surfaces."""
    with np.load(npz_path, allow_pickle=False) as data:
        if "artery" in data.files:
            vessel_m = np.asarray(data["artery"], dtype=np.float32)
        elif "reconstructed_vessel_code_mm" in data.files:
            vessel_m = (
                np.asarray(data["reconstructed_vessel_code_mm"], dtype=np.float32)
                * np.float32(1.0e-3)
            )
        else:
            return []
        if vessel_m.ndim != 3 or vessel_m.shape[-1] < 4:
            return []

        point_valid = (
            np.asarray(data["point_valid_mask"], dtype=bool)
            if "point_valid_mask" in data.files
            else np.isfinite(vessel_m[..., :4]).all(axis=-1)
        )
        branch_indices = (
            np.asarray(data["projected_branch_indices"], dtype=np.int64).reshape(-1)
            if "projected_branch_indices" in data.files
            else np.flatnonzero(np.any(point_valid, axis=1))
        )
        projection_center = (
            np.asarray(data["projection_center_offset"], dtype=np.float32).reshape(3)
            if "projection_center_offset" in data.files
            else np.zeros((3,), dtype=np.float32)
        )

    surfaces: list[np.ndarray] = []
    for branch_index in branch_indices:
        index = int(branch_index)
        if index < 0 or index >= vessel_m.shape[0]:
            continue
        valid = np.asarray(point_valid[index], dtype=bool)
        valid &= np.isfinite(vessel_m[index, :, :4]).all(axis=1)
        valid &= vessel_m[index, :, 3] > 0.0
        if int(valid.sum()) < 2:
            continue
        surface = _build_tube_surface(
            vessel_m[index, valid, :3],
            vessel_m[index, valid, 3],
            num_circle_points=MONITOR_SURFACE_CIRCLE_POINTS,
        )
        surfaces.append(surface - projection_center.reshape(1, 1, 3))
    return surfaces


def _extract_volume_isosurface(
    volume_zyx: np.ndarray,
    volume_extent_m: float,
    isovalue: Optional[float],
) -> tuple[np.ndarray, np.ndarray, float]:
    try:
        from skimage import measure
    except ImportError as error:
        raise RuntimeError(
            "scikit-image is required for reconstructed_volume.gif; install "
            "requirements-stage2.txt again."
        ) from error

    volume = np.asarray(volume_zyx, dtype=np.float32)
    if volume.ndim != 3 or min(volume.shape) < 2:
        raise ValueError(f"Expected a 3D volume with dimensions >=2, got {volume.shape}.")
    volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)
    value_min = float(volume.min())
    value_max = float(volume.max())
    if value_max <= value_min:
        raise ValueError("Cannot render an isosurface from a constant volume.")
    if isovalue is not None:
        level = float(isovalue)
    else:
        positive = volume[volume > 0.0]
        if positive.size == 0:
            raise ValueError(
                "Cannot select the default GIF isovalue because the volume has "
                "no positive voxels."
            )
        level = float(
            np.percentile(positive, DEFAULT_VOLUME_GIF_POSITIVE_PERCENTILE)
        )
        # A very sparse or quantized volume can place P97 exactly at an endpoint,
        # while marching_cubes requires a strictly interior level.
        level = max(level, float(np.nextafter(value_min, value_max)))
        level = min(level, float(np.nextafter(value_max, value_min)))
    if not value_min < level < value_max:
        raise ValueError(
            f"Volume GIF isovalue must be inside ({value_min:.7g}, {value_max:.7g}), "
            f"got {level:.7g}."
        )

    vertices_zyx, faces, _, _ = measure.marching_cubes(
        volume,
        level=level,
        allow_degenerate=False,
    )
    denominators = np.maximum(np.asarray(volume.shape, dtype=np.float32) - 1.0, 1.0)
    normalized_zyx = vertices_zyx / denominators[None, :]
    vertices_xyz_m = np.stack(
        [
            (normalized_zyx[:, 2] - 0.5) * float(volume_extent_m),
            (normalized_zyx[:, 1] - 0.5) * float(volume_extent_m),
            (normalized_zyx[:, 0] - 0.5) * float(volume_extent_m),
        ],
        axis=1,
    )
    return (
        np.asarray(vertices_xyz_m, dtype=np.float32),
        np.asarray(faces, dtype=np.int32),
        level,
    )


def save_reconstructed_volume_gif(
    *,
    volume_zyx: np.ndarray,
    volume_extent_m: float,
    source_npz: Path,
    out_path: Path,
    num_frames: int = 24,
    fps: int = 5,
    isovalue: Optional[float] = None,
    title: str = "Reconstructed volume",
) -> float:
    """Save the prediction-only volume GIF with overlay-synchronized framing."""
    try:
        import imageio.v2 as imageio
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "imageio and matplotlib are required for reconstructed_volume.gif; "
            "install requirements-stage2.txt again."
        ) from error

    out_path.parent.mkdir(parents=True, exist_ok=True)
    gt_surfs_raw = _reference_surfaces_from_npz(Path(source_npz))
    pred_vertices_raw, faces, resolved_isovalue = _extract_volume_isosurface(
        volume_zyx=volume_zyx,
        volume_extent_m=float(volume_extent_m),
        isovalue=isovalue,
    )
    _, pred_surfs, points_for_axes = _center_surface_overlay(
        gt_surfs_raw,
        [pred_vertices_raw],
    )
    pred_vertices = pred_surfs[0]

    frames: list[np.ndarray] = []
    frame_count = max(1, int(num_frames))
    for frame_index in range(frame_count):
        fig = plt.figure(figsize=(5.2, 5.2), dpi=120)
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_trisurf(
            pred_vertices[:, 0],
            pred_vertices[:, 1],
            pred_vertices[:, 2],
            triangles=faces,
            color="red",
            alpha=0.5,
            shade=True,
            linewidth=0.0,
            antialiased=False,
        )
        _set_equal_3d_axes(ax, points_for_axes)
        ax.set_title(title)
        ax.view_init(
            elev=22.0,
            azim=360.0 * float(frame_index) / float(frame_count),
        )
        ax.set_axis_off()
        fig.tight_layout(pad=0.0)
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(
            height, width, 4
        )[..., :3]
        frames.append(frame.copy())
        plt.close(fig)
    imageio.mimsave(out_path, frames, fps=max(1, int(fps)), loop=0)
    return float(resolved_isovalue)
