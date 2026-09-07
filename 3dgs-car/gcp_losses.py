"""Differentiable supervision terms for the Gaussian-centre predictor."""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass(frozen=True)
class GCPLossWeights:
    """Explicit weights for every term in :func:`gcp_loss`."""

    chamfer: float = 1.0
    silog: float = 1.0
    depth_l1: float = 1.0
    depth_gradient: float = 1.0
    cldice: float = 1.0

    def __post_init__(self) -> None:
        for field in fields(self):
            value = float(getattr(self, field.name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"Loss weight {field.name} must be finite and non-negative, got {value}."
                )

    @classmethod
    def from_value(
        cls, value: Optional[Union["GCPLossWeights", Mapping[str, float]]],
    ) -> "GCPLossWeights":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("weights must be GCPLossWeights, a mapping, or None.")
        expected = {field.name for field in fields(cls)}
        supplied = set(value)
        if supplied != expected:
            missing = sorted(expected.difference(supplied))
            unexpected = sorted(supplied.difference(expected))
            raise ValueError(
                "A loss-weight mapping must specify every term; "
                f"missing={missing}, unexpected={unexpected}."
            )
        return cls(**{name: float(value[name]) for name in expected})


def _require_point_tensor(name: str, points: Tensor) -> None:
    if not torch.is_tensor(points) or points.ndim != 3 or points.shape[-1] != 3:
        shape = (
            tuple(points.shape) if torch.is_tensor(points) else type(points).__name__
        )
        raise ValueError(f"{name} must have shape [B,N,3], got {shape}.")
    if not points.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype.")


def chamfer_distance(
    predicted_points: Tensor,
    target_points: Tensor,
    target_mask: Optional[Tensor] = None,
    *,
    predicted_point_mask: Optional[Tensor] = None,
    chunk_size: int = 1024,
    squared: bool = True,
) -> Tensor:
    """Return a batch-mean symmetric Chamfer distance.

    Either tensor may contain padding or geometrically invalid points. Entries
    masked false do not participate in either Chamfer direction. Pairwise
    distances are constructed in chunks, avoiding a full ``N x M`` allocation.
    """

    _require_point_tensor("predicted_points", predicted_points)
    _require_point_tensor("target_points", target_points)
    if predicted_points.shape[0] != target_points.shape[0]:
        raise ValueError("predicted_points and target_points must share a batch size.")
    if predicted_points.shape[1] == 0 or target_points.shape[1] == 0:
        raise ValueError("Chamfer distance requires non-empty point dimensions.")
    if predicted_points.device != target_points.device:
        raise ValueError(
            "predicted_points and target_points must be on the same device."
        )
    if predicted_points.dtype != target_points.dtype:
        raise ValueError("predicted_points and target_points must have the same dtype.")
    if (
        not isinstance(chunk_size, int)
        or isinstance(chunk_size, bool)
        or chunk_size <= 0
    ):
        raise ValueError("chunk_size must be a positive integer.")

    batch_size, num_predicted, _ = predicted_points.shape
    num_target = target_points.shape[1]
    if predicted_point_mask is None:
        predicted_point_mask = torch.ones(
            (batch_size, num_predicted),
            dtype=torch.bool,
            device=predicted_points.device,
        )
    else:
        if tuple(predicted_point_mask.shape) != (batch_size, num_predicted):
            raise ValueError(
                "predicted_point_mask must have shape "
                f"{(batch_size, num_predicted)}, got "
                f"{tuple(predicted_point_mask.shape)}."
            )
        predicted_point_mask = predicted_point_mask.to(
            device=predicted_points.device, dtype=torch.bool,
        )
    if target_mask is None:
        target_mask = torch.ones(
            (batch_size, num_target), dtype=torch.bool, device=target_points.device,
        )
    else:
        if tuple(target_mask.shape) != (batch_size, num_target):
            raise ValueError(
                "target_mask must have shape "
                f"{(batch_size, num_target)}, got {tuple(target_mask.shape)}."
            )
        target_mask = target_mask.to(device=target_points.device, dtype=torch.bool)
    if not torch.all(predicted_point_mask.any(dim=1)):
        raise ValueError(
            "Each batch item must contain at least one valid predicted point."
        )
    if not torch.all(target_mask.any(dim=1)):
        raise ValueError(
            "Each batch item must contain at least one valid target point."
        )
    if not torch.isfinite(predicted_points[predicted_point_mask]).all():
        raise ValueError("Valid predicted_points contain NaN or infinity.")
    if not torch.isfinite(target_points[target_mask]).all():
        raise ValueError("Valid target_points contain NaN or infinity.")

    # NaN padding is allowed because masked points are replaced before cdist.
    safe_predicted = torch.where(
        predicted_point_mask.unsqueeze(-1),
        predicted_points,
        torch.zeros_like(predicted_points),
    )
    safe_targets = torch.where(
        target_mask.unsqueeze(-1), target_points, torch.zeros_like(target_points),
    )
    predicted_to_target = []
    for start in range(0, num_predicted, chunk_size):
        predicted_chunk = safe_predicted[:, start : start + chunk_size]
        nearest = predicted_points.new_full(predicted_chunk.shape[:2], float("inf"),)
        for target_start in range(0, num_target, chunk_size):
            target_end = min(target_start + chunk_size, num_target)
            distances = torch.cdist(
                predicted_chunk, safe_targets[:, target_start:target_end], p=2,
            )
            if squared:
                distances = distances.square()
            distances = distances.masked_fill(
                ~target_mask[:, None, target_start:target_end], float("inf"),
            )
            nearest = torch.minimum(nearest, distances.min(dim=-1).values)
        predicted_to_target.append(nearest)
    predicted_nearest = torch.cat(predicted_to_target, dim=1)
    predicted_counts = predicted_point_mask.sum(dim=1)
    predicted_term = torch.where(
        predicted_point_mask, predicted_nearest, torch.zeros_like(predicted_nearest),
    ).sum(dim=1) / predicted_counts.clamp_min(1)

    target_sums = predicted_points.new_zeros(batch_size)
    target_counts = predicted_points.new_zeros(batch_size)
    for start in range(0, num_target, chunk_size):
        end = min(start + chunk_size, num_target)
        chunk_mask = target_mask[:, start:end]
        target_chunk = safe_targets[:, start:end]
        nearest = predicted_points.new_full(target_chunk.shape[:2], float("inf"))
        for predicted_start in range(0, num_predicted, chunk_size):
            distances = torch.cdist(
                target_chunk,
                safe_predicted[:, predicted_start : predicted_start + chunk_size],
                p=2,
            )
            if squared:
                distances = distances.square()
            distances = distances.masked_fill(
                ~predicted_point_mask[
                    :, None, predicted_start : predicted_start + chunk_size
                ],
                float("inf"),
            )
            nearest = torch.minimum(nearest, distances.min(dim=-1).values)
        target_sums = target_sums + (nearest * chunk_mask).sum(dim=1)
        target_counts = target_counts + chunk_mask.sum(dim=1)
    target_term = target_sums / target_counts.clamp_min(1.0)
    return (predicted_term + target_term).mean()


def _as_single_channel(name: str, value: Tensor) -> Tensor:
    if value.ndim == 3:
        value = value.unsqueeze(1)
    if value.ndim != 4 or value.shape[1] != 1:
        raise ValueError(f"{name} must have shape [B,1,H,W] or [B,H,W].")
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype.")
    return value


def _resize_depth_and_mask(
    target_depth: Tensor, depth_mask: Tensor, output_size: Tuple[int, int],
) -> Tuple[Tensor, Tensor]:
    if target_depth.shape[-2:] == output_size:
        return target_depth, depth_mask
    mask_float = depth_mask.to(dtype=target_depth.dtype)
    mask_fraction = F.interpolate(mask_float, size=output_size, mode="area")
    weighted_depth = F.interpolate(
        torch.where(depth_mask, target_depth, torch.zeros_like(target_depth)),
        size=output_size,
        mode="area",
    )
    resized_mask = mask_fraction > 0.0
    resized_depth = weighted_depth / mask_fraction.clamp_min(
        torch.finfo(target_depth.dtype).eps
    )
    return resized_depth, resized_mask


def _masked_batch_mean(values: Tensor, mask: Tensor) -> Tensor:
    flattened_values = values.reshape(values.shape[0], -1)
    flattened_mask = mask.reshape(mask.shape[0], -1)
    counts = flattened_mask.sum(dim=1)
    means = torch.where(
        flattened_mask, flattened_values, torch.zeros_like(flattened_values),
    ).sum(dim=1) / counts.clamp_min(1)
    valid_batches = counts > 0
    if not bool(valid_batches.any()):
        return values.sum() * 0.0
    return means[valid_batches].mean()


def depth_loss_components(
    predicted_depth: Tensor,
    target_depth: Tensor,
    depth_mask: Optional[Tensor] = None,
    *,
    silog_variance: float = 0.85,
    epsilon: float = 1.0e-6,
) -> Dict[str, Tensor]:
    """Compute SILog, masked-L1, and masked depth-gradient-L1 terms."""

    predicted_depth = _as_single_channel("predicted_depth", predicted_depth)
    target_depth = _as_single_channel("target_depth", target_depth)
    if predicted_depth.shape[0] != target_depth.shape[0]:
        raise ValueError("predicted_depth and target_depth must share a batch size.")
    if predicted_depth.device != target_depth.device:
        raise ValueError("predicted_depth and target_depth must be on the same device.")
    if predicted_depth.dtype != target_depth.dtype:
        raise ValueError("predicted_depth and target_depth must have the same dtype.")
    if not 0.0 <= silog_variance <= 1.0:
        raise ValueError("silog_variance must be in [0, 1].")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")

    if depth_mask is None:
        # The paper's released toy targets encode background rays as depth 1.
        depth_mask = (
            torch.isfinite(target_depth)
            & (target_depth > 0.0)
            & (target_depth < 1.0 - epsilon)
        )
    else:
        if depth_mask.ndim == 3:
            depth_mask = depth_mask.unsqueeze(1)
        if depth_mask.shape != target_depth.shape:
            raise ValueError("depth_mask must match target_depth before resizing.")
        depth_mask = depth_mask.to(device=target_depth.device, dtype=torch.bool)
        depth_mask = depth_mask & torch.isfinite(target_depth)

    target_depth, depth_mask = _resize_depth_and_mask(
        target_depth, depth_mask, predicted_depth.shape[-2:],
    )
    valid = depth_mask & torch.isfinite(predicted_depth) & (target_depth > 0.0)

    absolute_error = (predicted_depth - target_depth).abs()
    masked_l1 = _masked_batch_mean(absolute_error, valid)

    log_difference = (
        predicted_depth.clamp_min(epsilon).log() - target_depth.clamp_min(epsilon).log()
    )
    flat_log = log_difference.reshape(log_difference.shape[0], -1)
    flat_valid = valid.reshape(valid.shape[0], -1)
    counts = flat_valid.sum(dim=1)
    valid_log = torch.where(flat_valid, flat_log, torch.zeros_like(flat_log))
    means = valid_log.sum(dim=1) / counts.clamp_min(1)
    mean_squares = valid_log.square().sum(dim=1) / counts.clamp_min(1)
    silog_per_batch = (
        (mean_squares - silog_variance * means.square())
        .clamp_min(0.0)
        .add(epsilon)
        .sqrt()
    )
    valid_batches = counts > 0
    silog = (
        silog_per_batch[valid_batches].mean()
        if bool(valid_batches.any())
        else predicted_depth.sum() * 0.0
    )

    predicted_dx = predicted_depth[..., :, 1:] - predicted_depth[..., :, :-1]
    target_dx = target_depth[..., :, 1:] - target_depth[..., :, :-1]
    mask_dx = valid[..., :, 1:] & valid[..., :, :-1]
    predicted_dy = predicted_depth[..., 1:, :] - predicted_depth[..., :-1, :]
    target_dy = target_depth[..., 1:, :] - target_depth[..., :-1, :]
    mask_dy = valid[..., 1:, :] & valid[..., :-1, :]

    batch_size = predicted_depth.shape[0]
    gradient_sum = predicted_depth.new_zeros(batch_size)
    gradient_count = predicted_depth.new_zeros(batch_size)
    for difference, pair_mask in (
        ((predicted_dx - target_dx).abs(), mask_dx),
        ((predicted_dy - target_dy).abs(), mask_dy),
    ):
        gradient_sum = gradient_sum + torch.where(
            pair_mask, difference, torch.zeros_like(difference),
        ).reshape(batch_size, -1).sum(dim=1)
        gradient_count = gradient_count + pair_mask.reshape(batch_size, -1).sum(dim=1)
    valid_gradient_batches = gradient_count > 0
    gradient_per_batch = gradient_sum / gradient_count.clamp_min(1)
    gradient_l1 = (
        gradient_per_batch[valid_gradient_batches].mean()
        if bool(valid_gradient_batches.any())
        else predicted_depth.sum() * 0.0
    )

    return {
        "silog": silog,
        "depth_l1": masked_l1,
        "depth_gradient": gradient_l1,
    }


def depth_loss(
    predicted_depth: Tensor,
    target_depth: Tensor,
    depth_mask: Optional[Tensor] = None,
    *,
    silog_weight: float = 1.0,
    masked_l1_weight: float = 1.0,
    gradient_l1_weight: float = 1.0,
    silog_variance: float = 0.85,
    epsilon: float = 1.0e-6,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Return the weighted depth loss and its three unweighted components."""

    for name, weight in (
        ("silog_weight", silog_weight),
        ("masked_l1_weight", masked_l1_weight),
        ("gradient_l1_weight", gradient_l1_weight),
    ):
        if weight < 0.0:
            raise ValueError(f"{name} must be non-negative.")
    components = depth_loss_components(
        predicted_depth,
        target_depth,
        depth_mask,
        silog_variance=silog_variance,
        epsilon=epsilon,
    )
    total = (
        silog_weight * components["silog"]
        + masked_l1_weight * components["depth_l1"]
        + gradient_l1_weight * components["depth_gradient"]
    )
    return total, components


def _normalise_grid_size(grid_size: Union[int, Sequence[int]]) -> Tuple[int, int, int]:
    if isinstance(grid_size, int) and not isinstance(grid_size, bool):
        dimensions = (grid_size, grid_size, grid_size)
    else:
        dimensions = tuple(int(size) for size in grid_size)
    if len(dimensions) != 3 or any(size <= 0 for size in dimensions):
        raise ValueError(
            f"grid_size must contain three positive sizes, got {dimensions}."
        )
    return dimensions


def trilinear_point_splat(
    points: Tensor,
    grid_size: Union[int, Sequence[int]],
    point_mask: Optional[Tensor] = None,
    point_weights: Optional[Tensor] = None,
    *,
    clamp_points: bool = True,
    saturate: bool = True,
) -> Tensor:
    """Trilinearly splat normalised ZYX points into ``[B,1,D,H,W]``.

    With ``saturate=True``, accumulated density is converted to soft occupancy
    as ``1 - exp(-density)``.  The operation is differentiable with respect to
    point coordinates and weights almost everywhere.
    """

    _require_point_tensor("points", points)
    depth, height, width = _normalise_grid_size(grid_size)
    batch_size, num_points, _ = points.shape
    if point_mask is None:
        point_mask = torch.ones(
            (batch_size, num_points), dtype=torch.bool, device=points.device,
        )
    else:
        if tuple(point_mask.shape) != (batch_size, num_points):
            raise ValueError("point_mask must have shape [B,N].")
        point_mask = point_mask.to(device=points.device, dtype=torch.bool)
    if not torch.isfinite(points[point_mask]).all():
        raise ValueError("Valid points contain NaN or infinity.")

    if point_weights is None:
        point_weights = torch.ones(
            (batch_size, num_points), dtype=points.dtype, device=points.device,
        )
    else:
        if point_weights.ndim == 3 and point_weights.shape[-1] == 1:
            point_weights = point_weights.squeeze(-1)
        if tuple(point_weights.shape) != (batch_size, num_points):
            raise ValueError("point_weights must have shape [B,N] or [B,N,1].")
        point_weights = point_weights.to(device=points.device, dtype=points.dtype)
        if not torch.isfinite(point_weights[point_mask]).all():
            raise ValueError("Valid point_weights contain NaN or infinity.")
        if bool((point_weights[point_mask] < 0.0).any()):
            raise ValueError("Valid point_weights must be non-negative.")

    safe_points = torch.where(
        point_mask.unsqueeze(-1), points, torch.zeros_like(points),
    )
    if clamp_points:
        safe_points = safe_points.clamp(0.0, 1.0)
    elif bool(
        ((safe_points[point_mask] < 0.0) | (safe_points[point_mask] > 1.0)).any()
    ):
        raise ValueError("Valid points must be within [0,1] when clamp_points=False.")

    scale = points.new_tensor((depth - 1, height - 1, width - 1))
    voxel_coordinates = safe_points * scale
    lower = voxel_coordinates.floor().to(dtype=torch.long)
    fraction = voxel_coordinates - lower.to(dtype=points.dtype)
    upper_limit = torch.tensor(
        (depth - 1, height - 1, width - 1), device=points.device, dtype=torch.long,
    )

    flat_occupancy = points.new_zeros((batch_size, depth * height * width))
    base_weight = point_weights * point_mask.to(dtype=points.dtype)
    for z_upper in (0, 1):
        for y_upper in (0, 1):
            for x_upper in (0, 1):
                corner_selector = points.new_tensor((z_upper, y_upper, x_upper))
                corner_index = torch.minimum(
                    lower + corner_selector.to(dtype=torch.long), upper_limit,
                )
                interpolation_weight = torch.where(
                    corner_selector.bool(), fraction, 1.0 - fraction,
                ).prod(dim=-1)
                contribution = base_weight * interpolation_weight
                flat_index = (
                    corner_index[..., 0] * height * width
                    + corner_index[..., 1] * width
                    + corner_index[..., 2]
                )
                flat_occupancy = flat_occupancy.scatter_add(1, flat_index, contribution)

    occupancy = flat_occupancy.reshape(batch_size, 1, depth, height, width)
    if saturate:
        occupancy = -torch.expm1(-occupancy.clamp_min(0.0))
    return occupancy


def _soft_erode_3d(volume: Tensor) -> Tensor:
    eroded_z = -F.max_pool3d(-volume, (3, 1, 1), stride=1, padding=(1, 0, 0))
    eroded_y = -F.max_pool3d(-volume, (1, 3, 1), stride=1, padding=(0, 1, 0))
    eroded_x = -F.max_pool3d(-volume, (1, 1, 3), stride=1, padding=(0, 0, 1))
    return torch.minimum(torch.minimum(eroded_z, eroded_y), eroded_x)


def _soft_dilate_3d(volume: Tensor) -> Tensor:
    return F.max_pool3d(volume, kernel_size=3, stride=1, padding=1)


def _soft_open_3d(volume: Tensor) -> Tensor:
    return _soft_dilate_3d(_soft_erode_3d(volume))


def soft_skeletonize_3d(volume: Tensor, iterations: int = 3) -> Tensor:
    """Approximate a 3-D morphological skeleton with differentiable pooling."""

    if volume.ndim == 4:
        volume = volume.unsqueeze(1)
    if volume.ndim != 5 or volume.shape[1] != 1:
        raise ValueError("volume must have shape [B,1,D,H,W] or [B,D,H,W].")
    if (
        not isinstance(iterations, int)
        or isinstance(iterations, bool)
        or iterations < 0
    ):
        raise ValueError("iterations must be a non-negative integer.")
    volume = volume.clamp(0.0, 1.0)
    skeleton = F.relu(volume - _soft_open_3d(volume))
    for _ in range(iterations):
        volume = _soft_erode_3d(volume)
        delta = F.relu(volume - _soft_open_3d(volume))
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton.clamp(0.0, 1.0)


def soft_cldice_loss(
    predicted_volume: Tensor,
    target_volume: Tensor,
    *,
    skeleton_iterations: int = 3,
    epsilon: float = 1.0e-6,
) -> Tensor:
    """Return batch-mean soft-clDice loss for 3-D occupancy volumes."""

    if predicted_volume.ndim == 4:
        predicted_volume = predicted_volume.unsqueeze(1)
    if target_volume.ndim == 4:
        target_volume = target_volume.unsqueeze(1)
    if predicted_volume.ndim != 5 or predicted_volume.shape[1] != 1:
        raise ValueError("predicted_volume must have shape [B,1,D,H,W].")
    if not predicted_volume.is_floating_point():
        raise TypeError("predicted_volume must have a floating-point dtype.")
    if target_volume.shape != predicted_volume.shape:
        raise ValueError(
            "target_volume must match predicted_volume, got "
            f"{tuple(target_volume.shape)} and {tuple(predicted_volume.shape)}."
        )
    if predicted_volume.device != target_volume.device:
        raise ValueError(
            "predicted_volume and target_volume must be on the same device."
        )
    if predicted_volume.dtype != target_volume.dtype:
        target_volume = target_volume.to(dtype=predicted_volume.dtype)
    if (
        not torch.isfinite(predicted_volume).all()
        or not torch.isfinite(target_volume).all()
    ):
        raise ValueError("clDice volumes contain NaN or infinity.")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")

    predicted_volume = predicted_volume.clamp(0.0, 1.0)
    target_volume = target_volume.clamp(0.0, 1.0)
    predicted_skeleton = soft_skeletonize_3d(
        predicted_volume, iterations=skeleton_iterations,
    )
    target_skeleton = soft_skeletonize_3d(
        target_volume, iterations=skeleton_iterations,
    )
    reduce_dimensions = tuple(range(1, predicted_volume.ndim))
    topology_precision = (
        (predicted_skeleton * target_volume).sum(dim=reduce_dimensions) + epsilon
    ) / (predicted_skeleton.sum(dim=reduce_dimensions) + epsilon)
    topology_sensitivity = (
        (target_skeleton * predicted_volume).sum(dim=reduce_dimensions) + epsilon
    ) / (target_skeleton.sum(dim=reduce_dimensions) + epsilon)
    cldice = (2.0 * topology_precision * topology_sensitivity + epsilon) / (
        topology_precision + topology_sensitivity + epsilon
    )
    return (1.0 - cldice).mean()


def gcp_loss(
    predicted_centers: Tensor,
    target_points: Tensor,
    target_point_mask: Optional[Tensor],
    predicted_depth: Tensor,
    target_depth: Tensor,
    depth_mask: Optional[Tensor],
    predicted_volume: Tensor,
    target_volume: Tensor,
    weights: Optional[Union[GCPLossWeights, Mapping[str, float]]] = None,
    *,
    predicted_point_mask: Optional[Tensor] = None,
    chamfer_chunk_size: int = 1024,
    skeleton_iterations: int = 3,
    silog_variance: float = 0.85,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Combine all paper-inspired GCP supervision terms.

    Returns ``(total, components)``.  Component tensors are unweighted and keep
    their autograd history; callers should detach them only when logging.
    """

    loss_weights = GCPLossWeights.from_value(weights)
    chamfer = chamfer_distance(
        predicted_centers,
        target_points,
        target_point_mask,
        predicted_point_mask=predicted_point_mask,
        chunk_size=chamfer_chunk_size,
    )
    depth_components = depth_loss_components(
        predicted_depth, target_depth, depth_mask, silog_variance=silog_variance,
    )
    cldice = soft_cldice_loss(
        predicted_volume, target_volume, skeleton_iterations=skeleton_iterations,
    )
    components = {
        "chamfer": chamfer,
        **depth_components,
        "cldice": cldice,
    }
    total = (
        loss_weights.chamfer * components["chamfer"]
        + loss_weights.silog * components["silog"]
        + loss_weights.depth_l1 * components["depth_l1"]
        + loss_weights.depth_gradient * components["depth_gradient"]
        + loss_weights.cldice * components["cldice"]
    )
    return total, components


__all__ = [
    "GCPLossWeights",
    "chamfer_distance",
    "depth_loss",
    "depth_loss_components",
    "gcp_loss",
    "soft_cldice_loss",
    "soft_skeletonize_3d",
    "trilinear_point_splat",
]
