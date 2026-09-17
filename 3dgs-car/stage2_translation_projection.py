"""Re-render one Stage-2 view after a fixed physical artery translation.

The Stage-2 ImageCAS files embed the exact artery used to make the stored
projection images.  This module reproduces the dataset projection convention
without changing the stored camera angles or the canonical 3D target.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from stage2_npz_data import stage2_angles_to_camera_frames


MM_TO_M = 1.0e-3
DEFAULT_RENDERER_NUM_CIRCLE_POINTS = 120
DEFAULT_CLEAN_RERENDER_MIN_DICE = 0.98
DEFAULT_VISIBILITY_WARNING_THRESHOLD = 0.95


def _scalar(data: Mapping[str, np.ndarray], key: str) -> object:
    return np.asarray(data[key]).reshape(()).item()


def _estimate_derivatives(points: np.ndarray) -> np.ndarray:
    derivatives = np.zeros_like(points)
    if points.shape[0] < 2:
        return derivatives
    derivatives[0] = points[1] - points[0]
    derivatives[-1] = points[-1] - points[-2]
    if points.shape[0] > 2:
        derivatives[1:-1] = 0.5 * (points[2:] - points[:-2])
    return derivatives


def _robust_tube_surface(
    centerline_xyz_m: np.ndarray,
    radii_m: np.ndarray,
    num_circle_points: int,
) -> np.ndarray:
    """Build the same capped-ring representation used by Stage-2 rendering."""

    centerline = np.asarray(centerline_xyz_m, dtype=np.float64)
    radii = np.asarray(radii_m, dtype=np.float64).reshape(-1)
    if centerline.ndim != 2 or centerline.shape[1] != 3 or centerline.shape[0] < 2:
        raise ValueError(
            f"Tube centerline must have shape [N>=2,3], got {centerline.shape}."
        )
    if radii.shape[0] != centerline.shape[0]:
        raise ValueError("Tube radii and centerline lengths differ.")
    if not np.isfinite(centerline).all() or not np.isfinite(radii).all():
        raise ValueError("Tube inputs contain NaN or infinity.")

    derivatives = _estimate_derivatives(centerline)
    norms = np.linalg.norm(derivatives, axis=1)
    valid_tangents = np.flatnonzero(norms > 1.0e-12)
    if valid_tangents.size == 0:
        raise ValueError("Cannot render a centerline with no non-zero tangent.")
    for index in np.flatnonzero(norms <= 1.0e-12):
        nearest = valid_tangents[np.argmin(np.abs(valid_tangents - index))]
        derivatives[index] = derivatives[nearest]
        norms[index] = norms[nearest]
    tangents = derivatives / norms[:, None]

    angles = np.linspace(
        0.0,
        2.0 * np.pi,
        int(num_circle_points),
        dtype=np.float64,
    )
    cos_angles = np.cos(angles)[:, None]
    sin_angles = np.sin(angles)[:, None]
    coordinate_axes = np.eye(3, dtype=np.float64)
    previous_normal = None
    rings: List[np.ndarray] = []
    for index, tangent in enumerate(tangents):
        normal = None
        if previous_normal is not None:
            transported = previous_normal - float(np.dot(previous_normal, tangent)) * tangent
            transported_norm = float(np.linalg.norm(transported))
            if transported_norm > 1.0e-12:
                normal = transported / transported_norm
        if normal is None:
            for axis_index in np.argsort(np.abs(coordinate_axes @ tangent)):
                candidate = coordinate_axes[int(axis_index)]
                candidate = candidate - float(np.dot(candidate, tangent)) * tangent
                candidate_norm = float(np.linalg.norm(candidate))
                if candidate_norm > 1.0e-12:
                    normal = candidate / candidate_norm
                    break
        if normal is None:
            raise ValueError(f"Could not construct a tube frame at point {index}.")
        conormal = np.cross(normal, tangent)
        conormal /= max(float(np.linalg.norm(conormal)), 1.0e-12)
        normal = np.cross(tangent, conormal)
        normal /= max(float(np.linalg.norm(normal)), 1.0e-12)
        previous_normal = normal

        if index == 0:
            ring_radii: Sequence[float] = np.linspace(
                0.0, radii[index], 50, dtype=np.float64
            )[1:]
        elif index == centerline.shape[0] - 1:
            ring_radii = np.flip(
                np.linspace(0.0, radii[index], 50, dtype=np.float64)[1:]
            )
        else:
            ring_radii = (float(radii[index]),)
        for ring_radius in ring_radii:
            rings.append(
                centerline[index][None, :]
                + float(ring_radius) * cos_angles * normal[None, :]
                + float(ring_radius) * sin_angles * conormal[None, :]
            )
    surface = np.asarray(rings, dtype=np.float32)
    if not np.isfinite(surface).all():
        raise ValueError("Generated tube surface contains NaN or infinity.")
    return surface


def _tube_surface(
    centerline_xyz_m: np.ndarray,
    radii_m: np.ndarray,
    num_circle_points: int,
) -> np.ndarray:
    """Use the legacy Stage-2 moving frame, with its finite fallback."""

    centerline = np.asarray(centerline_xyz_m, dtype=np.float64)
    radii = np.asarray(radii_m, dtype=np.float64).reshape(-1)
    try:
        derivatives = _estimate_derivatives(centerline)
        keep = np.flatnonzero(np.sum(np.abs(derivatives), axis=1) != 0.0)
        points = centerline[keep]
        derivatives = derivatives[keep]
        if points.shape[0] < 2:
            raise ValueError("Legacy tube frame has fewer than two valid points.")
        normal = np.zeros(3, dtype=np.float64)
        normal[int(np.argmin(np.abs(points[1])))] = 1.0
        angles = np.linspace(
            0.0, 2.0 * np.pi, int(num_circle_points), dtype=np.float64
        )
        cos_angles = np.cos(angles)[:, None]
        sin_angles = np.sin(angles)[:, None]
        rings: List[np.ndarray] = []
        for local_index, (point, derivative) in enumerate(
            zip(points, derivatives)
        ):
            conormal = np.cross(normal, derivative)
            conormal /= np.linalg.norm(conormal)
            normal = np.cross(derivative, conormal)
            normal /= np.linalg.norm(normal)
            if local_index == 0:
                ring_radii: Sequence[float] = np.linspace(
                    0.0, radii[local_index], 50, dtype=np.float64
                )[1:]
            elif local_index == points.shape[0] - 1:
                ring_radii = np.flip(
                    np.linspace(
                        0.0, radii[local_index], 50, dtype=np.float64
                    )[1:]
                )
            else:
                ring_radii = (float(radii[local_index]),)
            for ring_radius in ring_radii:
                rings.append(
                    point[None, :]
                    + float(ring_radius) * cos_angles * normal[None, :]
                    + float(ring_radius) * sin_angles * conormal[None, :]
                )
        surface = np.asarray(rings, dtype=np.float32)
        if surface.ndim != 3 or surface.shape[-1] != 3:
            raise ValueError("Legacy tube frame produced an invalid shape.")
        if not np.isfinite(surface).all():
            raise ValueError("Legacy tube frame produced NaN or infinity.")
        return surface
    except Exception:
        return _robust_tube_surface(
            centerline_xyz_m,
            radii_m,
            num_circle_points,
        )


def _projected_artery(
    data: Mapping[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if "artery" not in data:
        raise KeyError(
            "Translational calibration evaluation requires the embedded 'artery' array."
        )
    artery = np.asarray(data["artery"], dtype=np.float32)
    if artery.ndim != 3 or artery.shape[-1] < 4:
        raise ValueError(f"artery must have shape [B,N,4+], got {artery.shape}.")
    branch_indices = (
        np.asarray(data["projected_branch_indices"], dtype=np.int64).reshape(-1)
        if "projected_branch_indices" in data
        else np.arange(artery.shape[0], dtype=np.int64)
    )
    if branch_indices.size == 0:
        raise ValueError("projected_branch_indices is empty.")
    if int(branch_indices.min()) < 0 or int(branch_indices.max()) >= artery.shape[0]:
        raise IndexError("projected_branch_indices contains an invalid artery branch.")
    center_offset = (
        np.asarray(data["projection_center_offset"], dtype=np.float32).reshape(3)
        if "projection_center_offset" in data
        else np.zeros(3, dtype=np.float32)
    )
    return artery, branch_indices, center_offset


def _surfaces_and_centerlines(
    data: Mapping[str, np.ndarray],
    *,
    translation_xyz_m: np.ndarray,
    num_circle_points: int,
) -> Tuple[List[np.ndarray], np.ndarray]:
    artery, branch_indices, center_offset = _projected_artery(data)
    surfaces: List[np.ndarray] = []
    centerlines: List[np.ndarray] = []
    translation = np.asarray(translation_xyz_m, dtype=np.float32).reshape(3)
    for branch_index in branch_indices:
        branch = artery[int(branch_index)]
        valid = np.logical_and(
            np.any(np.abs(branch[:, :3]) > 0.0, axis=1),
            branch[:, 3] > 0.0,
        )
        valid &= np.isfinite(branch[:, :4]).all(axis=1)
        if int(valid.sum()) < 2:
            continue
        centerline = (
            branch[valid, :3]
            - center_offset.reshape(1, 3)
            + translation.reshape(1, 3)
        )
        surface = _tube_surface(
            branch[valid, :3],
            np.clip(branch[valid, 3], 1.0e-5, None),
            num_circle_points,
        )
        surface = surface - center_offset.reshape(1, 1, 3) + translation.reshape(
            1, 1, 3
        )
        centerlines.append(centerline.astype(np.float32))
        surfaces.append(surface.astype(np.float32))
    if not surfaces:
        raise ValueError("No valid projected artery branches could be rendered.")
    return surfaces, np.concatenate(centerlines, axis=0)


def _project_points_to_pixels(
    points_xyz_m: np.ndarray,
    *,
    source_xyz_m: np.ndarray,
    detector_xyz_m: np.ndarray,
    detector_u_axis: np.ndarray,
    detector_v_axis: np.ndarray,
    image_dim: int,
    sensor_width_m: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reproduce the legacy ray-plane intersection and pixel conversion."""

    points = np.asarray(points_xyz_m, dtype=np.float64).reshape(-1, 3)
    source = np.asarray(source_xyz_m, dtype=np.float64).reshape(3)
    detector = np.asarray(detector_xyz_m, dtype=np.float64).reshape(3)
    axis_u = np.asarray(detector_u_axis, dtype=np.float64).reshape(3)
    axis_v = np.asarray(detector_v_axis, dtype=np.float64).reshape(3)
    normal = np.cross(axis_u, axis_v)
    ray_to_source = source.reshape(1, 3) - points
    denominators = ray_to_source @ normal
    ray_valid = np.abs(denominators) > 1.0e-6
    pixels = np.full((points.shape[0], 2), np.nan, dtype=np.float64)
    if not np.any(ray_valid):
        return pixels, ray_valid

    valid_points = points[ray_valid]
    scale = -((valid_points - detector.reshape(1, 3)) @ normal) / denominators[
        ray_valid
    ]
    plane_points = valid_points + scale[:, None] * ray_to_source[ray_valid]

    dimension = float(image_dim)
    origin_factor = (1.0 - dimension / 2.0) / dimension
    maximum_factor = (dimension - dimension / 2.0) / dimension
    local_origin = detector + float(sensor_width_m) * origin_factor * (
        axis_u + axis_v
    )
    local_u_max = detector + float(sensor_width_m) * (
        maximum_factor * axis_u + origin_factor * axis_v
    )
    local_v_max = detector + float(sensor_width_m) * (
        origin_factor * axis_u + maximum_factor * axis_v
    )
    relative = plane_points - local_origin.reshape(1, 3)
    u_projection = (relative @ axis_u)[:, None] * axis_u.reshape(1, 3)
    v_projection = (relative @ axis_v)[:, None] * axis_v.reshape(1, 3)
    u_reference = local_u_max - local_origin
    v_reference = local_v_max - local_origin
    x_sign = np.sign(u_projection @ u_reference)
    y_sign = np.sign(v_projection @ v_reference)
    x = x_sign * dimension * np.linalg.norm(u_projection, axis=1) / np.linalg.norm(
        u_reference
    )
    y = y_sign * dimension * np.linalg.norm(v_projection, axis=1) / np.linalg.norm(
        v_reference
    )
    pixels[ray_valid] = np.stack((x, dimension - y), axis=1)
    return pixels, ray_valid


def _in_detector(pixels: np.ndarray, image_dim: int) -> np.ndarray:
    return np.isfinite(pixels).all(axis=1) & np.all(pixels > 0.0, axis=1) & np.all(
        pixels < float(image_dim), axis=1
    )


def _render_surface_mask(
    surfaces: Sequence[np.ndarray],
    *,
    source_xyz_m: np.ndarray,
    detector_xyz_m: np.ndarray,
    detector_u_axis: np.ndarray,
    detector_v_axis: np.ndarray,
    image_dim: int,
    sensor_width_m: float,
    mask_render_mode: str,
) -> np.ndarray:
    try:
        from skimage import draw, filters
        from skimage import morphology as morph
    except ImportError as error:
        raise RuntimeError(
            "scikit-image is required for translated Stage-2 reprojection."
        ) from error

    mask = np.zeros((image_dim, image_dim), dtype=np.bool_)
    if mask_render_mode == "point":
        points = np.concatenate([surface.reshape(-1, 3) for surface in surfaces], axis=0)
        pixels, _ = _project_points_to_pixels(
            points,
            source_xyz_m=source_xyz_m,
            detector_xyz_m=detector_xyz_m,
            detector_u_axis=detector_u_axis,
            detector_v_axis=detector_v_axis,
            image_dim=image_dim,
            sensor_width_m=sensor_width_m,
        )
        pixels = np.round(pixels)
        pixels = pixels[_in_detector(pixels, image_dim)].astype(np.int64)
        if pixels.size:
            mask[image_dim - pixels[:, 1], pixels[:, 0]] = True
    elif mask_render_mode == "filled":
        for surface in surfaces:
            previous_rows = None
            previous_columns = None
            for ring in surface.reshape(-1, surface.shape[-2], 3):
                pixels, _ = _project_points_to_pixels(
                    ring,
                    source_xyz_m=source_xyz_m,
                    detector_xyz_m=detector_xyz_m,
                    detector_u_axis=detector_u_axis,
                    detector_v_axis=detector_v_axis,
                    image_dim=image_dim,
                    sensor_width_m=sensor_width_m,
                )
                pixels = np.round(pixels)
                pixels = pixels[_in_detector(pixels, image_dim)]
                if pixels.shape[0] < 3:
                    continue
                rows = np.clip(
                    (image_dim - pixels[:, 1]).astype(np.int32), 0, image_dim - 1
                )
                columns = np.clip(
                    pixels[:, 0].astype(np.int32), 0, image_dim - 1
                )
                polygon_rows, polygon_columns = draw.polygon(
                    rows, columns, shape=mask.shape
                )
                mask[polygon_rows, polygon_columns] = True
                if previous_rows is not None and previous_columns is not None:
                    ring_length = min(len(rows), len(previous_rows))
                    for point_index in range(0, ring_length, 2):
                        line_rows, line_columns = draw.line(
                            int(previous_rows[point_index]),
                            int(previous_columns[point_index]),
                            int(rows[point_index]),
                            int(columns[point_index]),
                        )
                        valid = (
                            (line_rows >= 0)
                            & (line_rows < image_dim)
                            & (line_columns >= 0)
                            & (line_columns < image_dim)
                        )
                        mask[line_rows[valid], line_columns[valid]] = True
                previous_rows = rows
                previous_columns = columns
    else:
        raise ValueError(
            f"Unsupported mask_render_mode={mask_render_mode!r}; expected filled or point."
        )
    closed = morph.closing(mask, morph.disk(2))
    return (filters.gaussian(closed, sigma=0.5) > 0.25).astype(np.float32)


def _dice(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first) > 0.5
    b = np.asarray(second) > 0.5
    denominator = int(a.sum()) + int(b.sum())
    if denominator == 0:
        return 1.0
    return float(2.0 * np.logical_and(a, b).sum() / denominator)


def _render_view(
    data: Mapping[str, np.ndarray],
    *,
    view_index: int,
    translation_xyz_m: np.ndarray,
    num_circle_points: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    images = np.asarray(data["images"], dtype=np.float32)
    theta = np.asarray(data["theta_deg"], dtype=np.float32).reshape(-1)
    phi = np.asarray(data["phi_deg"], dtype=np.float32).reshape(-1)
    if not 0 <= int(view_index) < images.shape[0]:
        raise IndexError(f"View index {view_index} is outside [0,{images.shape[0] - 1}].")
    image_dim = int(images.shape[-1])
    if images.ndim != 3 or images.shape[1] != image_dim:
        raise ValueError(f"images must have shape [V,H,H], got {images.shape}.")
    sid_m = float(_scalar(data, "sid"))
    spacing = float(_scalar(data, "imager_pixel_spacing"))
    units = (
        str(_scalar(data, "imager_pixel_spacing_units")).strip().lower()
        if "imager_pixel_spacing_units" in data
        else "mm"
    )
    spacing_m = spacing * 1.0e-3 if units.startswith("mm") else spacing
    source_distance_m = 0.75
    source, detector, axis_u, axis_v = stage2_angles_to_camera_frames(
        theta_deg=np.asarray([theta[int(view_index)]], dtype=np.float32),
        phi_deg=np.asarray([phi[int(view_index)]], dtype=np.float32),
        sid_m=sid_m,
        source_origin_distance_m=source_distance_m,
    )
    surfaces, centerline = _surfaces_and_centerlines(
        data,
        translation_xyz_m=np.asarray(translation_xyz_m, dtype=np.float32),
        num_circle_points=num_circle_points,
    )
    sensor_width_m = spacing_m * image_dim
    mode = (
        str(_scalar(data, "mask_render_mode")).strip().lower()
        if "mask_render_mode" in data
        else "filled"
    )
    rendered = _render_surface_mask(
        surfaces,
        source_xyz_m=source[0],
        detector_xyz_m=detector[0],
        detector_u_axis=axis_u[0],
        detector_v_axis=axis_v[0],
        image_dim=image_dim,
        sensor_width_m=sensor_width_m,
        mask_render_mode=mode,
    )

    centerline_pixels, _ = _project_points_to_pixels(
        centerline,
        source_xyz_m=source[0],
        detector_xyz_m=detector[0],
        detector_u_axis=axis_u[0],
        detector_v_axis=axis_v[0],
        image_dim=image_dim,
        sensor_width_m=sensor_width_m,
    )
    surface_points = np.concatenate(
        [surface.reshape(-1, 3) for surface in surfaces], axis=0
    )
    surface_pixels, _ = _project_points_to_pixels(
        surface_points,
        source_xyz_m=source[0],
        detector_xyz_m=detector[0],
        detector_u_axis=axis_u[0],
        detector_v_axis=axis_v[0],
        image_dim=image_dim,
        sensor_width_m=sensor_width_m,
    )
    diagnostics = {
        "visible_centerline_fraction": float(
            np.mean(_in_detector(centerline_pixels, image_dim))
        ),
        "visible_vessel_surface_fraction": float(
            np.mean(_in_detector(surface_pixels, image_dim))
        ),
        "foreground_pixel_ratio": float(np.mean(rendered > 0.5)),
    }
    return rendered, diagnostics


def create_translated_stage2_case(
    source_path: Path,
    output_path: Path,
    *,
    selected_view_indices: Sequence[int],
    translation_xyz_mm: Sequence[float],
    num_circle_points: int = DEFAULT_RENDERER_NUM_CIRCLE_POINTS,
    clean_rerender_min_dice: float = DEFAULT_CLEAN_RERENDER_MIN_DICE,
    visibility_warning_threshold: float = DEFAULT_VISIBILITY_WARNING_THRESHOLD,
    fail_below_visibility_threshold: bool = False,
) -> Dict[str, Any]:
    """Write an NPZ where only the second selected input image is replaced."""

    selected = [int(index) for index in selected_view_indices]
    if len(selected) != 2 or selected[0] == selected[1]:
        raise ValueError(
            "Translated calibration evaluation requires exactly two distinct selected views."
        )
    translation_mm = np.asarray(translation_xyz_mm, dtype=np.float64).reshape(-1)
    if translation_mm.shape != (3,) or not np.isfinite(translation_mm).all():
        raise ValueError("translation_xyz_mm must contain three finite values.")
    if int(num_circle_points) < 8:
        raise ValueError("num_circle_points must be at least 8.")
    minimum_dice = float(clean_rerender_min_dice)
    if not 0.0 <= minimum_dice <= 1.0:
        raise ValueError("clean_rerender_min_dice must be in [0,1].")
    visibility_threshold = float(visibility_warning_threshold)
    if not 0.0 <= visibility_threshold <= 1.0:
        raise ValueError("visibility_warning_threshold must be in [0,1].")
    if not isinstance(fail_below_visibility_threshold, bool):
        raise ValueError("fail_below_visibility_threshold must be boolean.")

    with np.load(source_path, allow_pickle=False) as loaded:
        payload = {key: np.asarray(loaded[key]) for key in loaded.files}
    _, projected_branch_indices, projection_center_offset = _projected_artery(
        payload
    )
    images = np.asarray(payload["images"], dtype=np.float32).copy()
    view2_index = selected[1]
    if max(selected) >= images.shape[0] or min(selected) < 0:
        raise IndexError(
            f"Selected views {selected} are invalid for {images.shape[0]} stored views."
        )
    zero_image, zero_diagnostics = _render_view(
        payload,
        view_index=view2_index,
        translation_xyz_m=np.zeros(3, dtype=np.float32),
        num_circle_points=int(num_circle_points),
    )
    clean_dice = _dice(images[view2_index], zero_image)
    if clean_dice < minimum_dice:
        raise RuntimeError(
            "Zero-translation view-2 re-render failed the clean control: "
            f"Dice={clean_dice:.6f} < {minimum_dice:.6f}. Check the renderer, "
            "projected branch subset, centring offset, and circle-point setting."
        )

    translated_image, translated_diagnostics = _render_view(
        payload,
        view_index=view2_index,
        translation_xyz_m=translation_mm * MM_TO_M,
        num_circle_points=int(num_circle_points),
    )
    below_visibility_threshold = bool(
        translated_diagnostics["visible_centerline_fraction"]
        < visibility_threshold
    )
    if below_visibility_threshold and fail_below_visibility_threshold:
        raise RuntimeError(
            "Translated view-2 centreline visibility is below the configured "
            f"threshold: {translated_diagnostics['visible_centerline_fraction']:.6f} "
            f"< {visibility_threshold:.6f}."
        )
    original_foreground = float(np.mean(images[view2_index] > 0.5))
    images[view2_index] = translated_image
    payload["images"] = images
    payload.update(
        {
            "translation_calibration_applied": np.asarray(True),
            "translation_calibration_selected_view_indices": np.asarray(
                selected, dtype=np.int32
            ),
            "translation_calibration_changed_view_index": np.asarray(
                view2_index, dtype=np.int32
            ),
            "translation_calibration_vector_xyz_mm": translation_mm.astype(
                np.float32
            ),
            "translation_calibration_vector_xyz_m": (
                translation_mm * MM_TO_M
            ).astype(np.float32),
            "translation_calibration_magnitude_mm": np.asarray(
                float(np.linalg.norm(translation_mm)), dtype=np.float32
            ),
            "translation_calibration_delta_theta_deg": np.asarray(
                0.0, dtype=np.float32
            ),
            "translation_calibration_delta_phi_deg": np.asarray(
                0.0, dtype=np.float32
            ),
            "translation_calibration_convention": np.asarray(
                "artery_plus_delta_t_equivalent_camera_isocentre_minus_delta_t"
            ),
            "translation_calibration_renderer_num_circle_points": np.asarray(
                int(num_circle_points), dtype=np.int32
            ),
            "translation_calibration_clean_rerender_dice": np.asarray(
                clean_dice, dtype=np.float32
            ),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)

    theta = np.asarray(payload["theta_deg"], dtype=np.float64).reshape(-1)
    phi = np.asarray(payload["phi_deg"], dtype=np.float64).reshape(-1)
    return {
        "translation_calibration_type": (
            "fixed_two_view_positive_direction_translational_stress_test"
        ),
        "translation_coordinate_convention": {
            "+x": "patient left",
            "+y": "patient anterior, away from table",
            "+z": "patient superior, toward head",
        },
        "translation_application_convention": (
            "artery + delta_t; equivalent to source-detector system/isocentre - delta_t"
        ),
        "translation_selected_view_indices": selected,
        "translation_unchanged_view_index": selected[0],
        "translation_changed_view_index": view2_index,
        "translation_vector_xyz_mm": translation_mm.tolist(),
        "translation_equivalent_source_detector_isocentre_vector_xyz_mm": (
            -translation_mm
        ).tolist(),
        "translation_total_magnitude_mm": float(np.linalg.norm(translation_mm)),
        "translation_delta_theta_deg": 0.0,
        "translation_delta_phi_deg": 0.0,
        "translation_nominal_theta_deg": [float(theta[index]) for index in selected],
        "translation_nominal_phi_deg": [float(phi[index]) for index in selected],
        "translation_angles_and_view_features_unchanged": True,
        "translation_ground_truth_unchanged": True,
        "translation_projected_branch_indices": (
            projected_branch_indices.astype(int).tolist()
        ),
        "translation_projection_center_offset_xyz_m": (
            projection_center_offset.astype(float).tolist()
        ),
        "translation_original_view2_foreground_pixel_ratio": original_foreground,
        "translation_zero_rerender_view2_foreground_pixel_ratio": zero_diagnostics[
            "foreground_pixel_ratio"
        ],
        "translation_translated_view2_foreground_pixel_ratio": translated_diagnostics[
            "foreground_pixel_ratio"
        ],
        "translation_zero_visible_centerline_fraction": zero_diagnostics[
            "visible_centerline_fraction"
        ],
        "translation_visible_centerline_fraction": translated_diagnostics[
            "visible_centerline_fraction"
        ],
        "translation_zero_visible_vessel_surface_fraction": zero_diagnostics[
            "visible_vessel_surface_fraction"
        ],
        "translation_visible_vessel_surface_fraction": translated_diagnostics[
            "visible_vessel_surface_fraction"
        ],
        "translation_visibility_warning_threshold": visibility_threshold,
        "translation_below_visibility_warning_threshold": (
            below_visibility_threshold
        ),
        "translation_clean_rerender_dice": clean_dice,
        "translation_clean_rerender_minimum_dice": minimum_dice,
        "translation_renderer_num_circle_points": int(num_circle_points),
        "translation_image_feature_cache_used": False,
        "translation_image_features_recomputed_by_gcp": True,
        "translation_effective_input_npz": str(output_path),
    }


def translation_vector_mm(family: str, magnitude_mm: float) -> Tuple[float, float, float]:
    """Return one fixed-positive-direction vector with the requested norm."""

    magnitude = float(magnitude_mm)
    if not math.isfinite(magnitude) or magnitude <= 0.0:
        raise ValueError("Translation magnitude must be a positive finite number.")
    normalized = str(family).strip().upper()
    if normalized == "Y":
        return 0.0, magnitude, 0.0
    if normalized == "XZ":
        component = magnitude / math.sqrt(2.0)
        return component, 0.0, component
    if normalized == "XYZ":
        component = magnitude / math.sqrt(3.0)
        return component, component, component
    raise ValueError(f"Unsupported translation family {family!r}; use Y, XZ, or XYZ.")
