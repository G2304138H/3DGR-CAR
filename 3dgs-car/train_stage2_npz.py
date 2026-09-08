#!/usr/bin/env python3
"""Reconstruct one Stage-2 NPZ case with initialized 3D Gaussians.

This entry point is deliberately separate from ``train.py``.  The released demo
uses a one-angle circular ODL geometry and synthesizes its own projections from
a ground-truth volume.  Stage-2 NPZ files instead contain binary masks rendered
with arbitrary two-angle camera poses.  Here those poses are represented as
ASTRA ``cone_vec`` geometry, used consistently for initialization, optimization,
and novel-view rendering.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from arguments_init import OptimizationParams
from gaussian_model_anisotropic import GaussianModelAnisotropic
from stage2_npz_data import (
    DEFAULT_SOURCE_ORIGIN_DISTANCE_M,
    Stage2ProjectionCase,
    load_stage2_projection_case,
    validate_view_indices,
)
from volume_gif import (
    DEFAULT_VOLUME_GIF_POSITIVE_PERCENTILE,
    save_reconstructed_volume_gif,
)


def create_grid_3d(depth: int, height: int, width: int, device: torch.device) -> torch.Tensor:
    axes = (
        torch.linspace(0.0, 1.0, steps=depth, device=device),
        torch.linspace(0.0, 1.0, steps=height, device=device),
        torch.linspace(0.0, 1.0, steps=width, device=device),
    )
    grid_z, grid_y, grid_x = torch.meshgrid(*axes, indexing="ij")
    return torch.stack([grid_z, grid_y, grid_x], dim=-1).unsqueeze(0)


class _AstraProjectionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, volume: torch.Tensor, projector: "AstraConeVecProjector") -> torch.Tensor:
        ctx.projector = projector
        return projector._forward_tensor(volume)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return ctx.projector._backward_tensor(grad_output), None


class AstraConeVecProjector:
    """Differentiable ASTRA cone-vector forward projector for a batch of one."""

    def __init__(
        self,
        vectors: np.ndarray,
        detector_shape: Sequence[int],
        volume_size: int,
        volume_extent_m: float,
        gpu_index: int = 0,
    ) -> None:
        try:
            import astra
        except ImportError as error:
            raise RuntimeError(
                "ASTRA Toolbox is required. Install it in the 3DGR-CAR environment, "
                "preferably with `conda install -c astra-toolbox astra-toolbox`."
            ) from error

        self.astra = astra
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.detector_rows = int(detector_shape[0])
        self.detector_cols = int(detector_shape[1])
        self.volume_size = int(volume_size)
        self.volume_extent_m = float(volume_extent_m)
        self.gpu_index = int(gpu_index)
        if self.vectors.ndim != 2 or self.vectors.shape[1] != 12:
            raise ValueError(f"ASTRA cone vectors must have shape [V,12], got {self.vectors.shape}.")
        if self.volume_size <= 0 or self.volume_extent_m <= 0.0:
            raise ValueError("volume_size and volume_extent_m must be positive.")

        half_extent = self.volume_extent_m / 2.0
        self.volume_geometry = astra.create_vol_geom(
            self.volume_size,
            self.volume_size,
            self.volume_size,
            -half_extent,
            half_extent,
            -half_extent,
            half_extent,
            -half_extent,
            half_extent,
        )
        self.projection_geometry = astra.create_proj_geom(
            "cone_vec",
            self.detector_rows,
            self.detector_cols,
            self.vectors,
        )
        self.projector_id = astra.create_projector(
            "cuda3d", self.projection_geometry, self.volume_geometry,
            options={"GPUIndex": self.gpu_index},
        )
        projector3d = getattr(astra, "projector3d", None)
        self._has_direct_dlpack = bool(
            projector3d is not None
            and hasattr(projector3d, "direct_FP")
            and hasattr(projector3d, "direct_BP")
        )
        self._warned_fallback = False

    @property
    def num_views(self) -> int:
        return int(self.vectors.shape[0])

    def close(self) -> None:
        if getattr(self, "projector_id", None) is not None:
            self.astra.projector3d.delete(self.projector_id)
            self.projector_id = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __call__(self, volume: torch.Tensor) -> torch.Tensor:
        return _AstraProjectionFunction.apply(volume, self)

    def _validate_volume(self, volume: torch.Tensor) -> None:
        expected = (1, self.volume_size, self.volume_size, self.volume_size)
        if tuple(volume.shape) != expected:
            raise ValueError(f"Volume must have shape {expected}, got {tuple(volume.shape)}.")
        if volume.dtype != torch.float32:
            raise TypeError(f"ASTRA volume must be float32, got {volume.dtype}.")

    def _validate_projections(self, projections: torch.Tensor) -> None:
        expected = (1, self.num_views, self.detector_rows, self.detector_cols)
        if tuple(projections.shape) != expected:
            raise ValueError(f"Projections must have shape {expected}, got {tuple(projections.shape)}.")
        if projections.dtype != torch.float32:
            raise TypeError(f"ASTRA projections must be float32, got {projections.dtype}.")

    def _forward_tensor(self, volume: torch.Tensor) -> torch.Tensor:
        self._validate_volume(volume)
        # ASTRA consumes the tensor through DLPack and does not participate in
        # PyTorch autograd.  The surrounding _AstraProjectionFunction supplies
        # the matching backprojection explicitly, so exporting a detached view
        # here is both required by torch.__dlpack__ and preserves gradients via
        # that custom backward method.
        volume_zyx = volume[0].detach().contiguous()
        if self._has_direct_dlpack and volume.is_cuda:
            # ASTRA's projection kernels may accumulate into the supplied output.
            astra_projection = torch.zeros(
                (self.detector_rows, self.num_views, self.detector_cols),
                dtype=torch.float32,
                device=volume.device,
            )
            self.astra.projector3d.direct_FP(
                self.projector_id, volume_zyx, out=astra_projection
            )
            return astra_projection.permute(1, 0, 2).unsqueeze(0).contiguous()

        self._warn_cpu_fallback()
        result = self._forward_numpy(volume_zyx.detach().cpu().numpy())
        return torch.from_numpy(result).to(device=volume.device).unsqueeze(0)

    def _backward_tensor(self, projections: torch.Tensor) -> torch.Tensor:
        self._validate_projections(projections)
        astra_projection = (
            projections[0].detach().permute(1, 0, 2).contiguous()
        )
        if self._has_direct_dlpack and projections.is_cuda:
            volume = torch.zeros(
                (self.volume_size, self.volume_size, self.volume_size),
                dtype=torch.float32,
                device=projections.device,
            )
            self.astra.projector3d.direct_BP(
                self.projector_id, astra_projection, out=volume
            )
            return volume.unsqueeze(0)

        self._warn_cpu_fallback()
        result = self._backward_numpy(astra_projection.detach().cpu().numpy())
        return torch.from_numpy(result).to(device=projections.device).unsqueeze(0)

    def _warn_cpu_fallback(self) -> None:
        if not self._warned_fallback:
            print(
                "[WARN] This ASTRA version lacks direct DLPack FP/BP. Optimization "
                "will copy arrays between PyTorch and ASTRA every iteration and will "
                "be slow. ASTRA >=2.4 is recommended."
            )
            self._warned_fallback = True

    def _forward_numpy(self, volume_zyx: np.ndarray) -> np.ndarray:
        astra = self.astra
        volume_id = astra.data3d.create("-vol", self.volume_geometry, volume_zyx)
        projection_id = astra.data3d.create("-sino", self.projection_geometry)
        algorithm_id = None
        try:
            config = astra.astra_dict("FP3D_CUDA")
            config["VolumeDataId"] = volume_id
            config["ProjectionDataId"] = projection_id
            config["option"] = {"GPUindex": self.gpu_index}
            algorithm_id = astra.algorithm.create(config)
            astra.algorithm.run(algorithm_id)
            projection = astra.data3d.get(projection_id)
        finally:
            if algorithm_id is not None:
                astra.algorithm.delete(algorithm_id)
            astra.data3d.delete([volume_id, projection_id])
        return np.ascontiguousarray(np.transpose(projection, (1, 0, 2)), dtype=np.float32)

    def _backward_numpy(self, astra_projection: np.ndarray) -> np.ndarray:
        astra = self.astra
        projection_id = astra.data3d.create("-sino", self.projection_geometry, astra_projection)
        volume_id = astra.data3d.create("-vol", self.volume_geometry)
        algorithm_id = None
        try:
            config = astra.astra_dict("BP3D_CUDA")
            config["ProjectionDataId"] = projection_id
            config["ReconstructionDataId"] = volume_id
            config["option"] = {"GPUindex": self.gpu_index}
            algorithm_id = astra.algorithm.create(config)
            astra.algorithm.run(algorithm_id)
            volume = astra.data3d.get(volume_id)
        finally:
            if algorithm_id is not None:
                astra.algorithm.delete(algorithm_id)
            astra.data3d.delete([projection_id, volume_id])
        return np.ascontiguousarray(volume, dtype=np.float32)

    def reconstruct(self, projections: torch.Tensor, method: str = "fdk") -> torch.Tensor:
        """Return an FDK or simple-backprojection initialization volume."""
        self._validate_projections(projections)
        astra_projection = np.ascontiguousarray(
            projections[0].detach().cpu().numpy().transpose(1, 0, 2),
            dtype=np.float32,
        )
        astra = self.astra
        projection_id = astra.data3d.create("-sino", self.projection_geometry, astra_projection)
        volume_id = astra.data3d.create("-vol", self.volume_geometry)
        algorithm_id = None
        algorithm_name = "FDK_CUDA" if method == "fdk" else "BP3D_CUDA"
        try:
            config = astra.astra_dict(algorithm_name)
            config["ProjectionDataId"] = projection_id
            config["ReconstructionDataId"] = volume_id
            config["option"] = {"GPUindex": self.gpu_index}
            if method == "fdk":
                config["option"]["FilterType"] = "ram-lak"
            algorithm_id = astra.algorithm.create(config)
            astra.algorithm.run(algorithm_id)
            volume = astra.data3d.get(volume_id)
        except Exception as error:
            raise RuntimeError(
                f"ASTRA {algorithm_name} failed for the selected cone-vector views. "
                "FDK assumes a circular cone-beam scan and two tilted views are a "
                "very sparse approximation. Try `--init-method bp` only if ASTRA "
                "rejects this geometry."
            ) from error
        finally:
            if algorithm_id is not None:
                astra.algorithm.delete(algorithm_id)
            astra.data3d.delete([projection_id, volume_id])
        return torch.from_numpy(np.ascontiguousarray(volume, dtype=np.float32)).to(
            device=projections.device
        ).unsqueeze(0)


def normalize_initial_volume(volume: torch.Tensor) -> torch.Tensor:
    volume = torch.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)
    volume = torch.clamp(volume, min=0.0)
    positive = volume[volume > 0.0]
    if positive.numel() == 0:
        raise RuntimeError("FBP/backprojection contains no positive voxels.")
    scale = torch.quantile(positive, 0.995).clamp_min(1.0e-8)
    return torch.clamp(volume / scale, 0.0, 1.0)


def silhouette_from_line_integrals(line_integrals: torch.Tensor, gain: float) -> torch.Tensor:
    return 1.0 - torch.exp(-float(gain) * torch.clamp(line_integrals, min=0.0))


def select_gcp_initialization_view(view_indices: Sequence[int]) -> int:
    """Return the one view used by the monocular GCP."""
    flattened = np.asarray(view_indices, dtype=np.int64).reshape(-1)
    if flattened.size == 0:
        raise ValueError("At least one view index is required for GCP initialization.")
    return int(flattened[0])


def resolve_projection_loss_alpha(
    init_method: str,
    requested_alpha: Optional[float],
) -> float:
    """Resolve the global-MSE weight without changing legacy FDK/BP defaults."""
    alpha = (
        (0.5 if str(init_method) == "gcp" else 1.0)
        if requested_alpha is None
        else float(requested_alpha)
    )
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("projection loss alpha must be in [0, 1].")
    return alpha


def fixed_view_direction_records(
    case: Stage2ProjectionCase,
    view_indices: Sequence[int],
    *,
    theta_change_deg: float,
    phi_change_deg: float,
) -> Tuple[np.ndarray, np.ndarray, list[Dict[str, Any]]]:
    """Resolve one fixed assumed-pose error for every selected input view."""

    indices = validate_view_indices(view_indices, case.num_views)
    theta_change = float(theta_change_deg)
    phi_change = float(phi_change_deg)
    if not math.isfinite(theta_change) or not math.isfinite(phi_change):
        raise ValueError("View-direction changes must be finite degrees.")
    evaluated_theta = np.asarray(
        case.theta_deg[indices] + theta_change, dtype=np.float32
    )
    evaluated_phi = np.asarray(
        case.phi_deg[indices] + phi_change, dtype=np.float32
    )
    records = [
        {
            "model_view_position": int(position),
            "selected_view_index": int(index),
            "original_theta_deg": float(case.theta_deg[index]),
            "theta_change_deg": theta_change,
            "evaluated_theta_deg": float(evaluated_theta[position]),
            "original_phi_deg": float(case.phi_deg[index]),
            "phi_change_deg": phi_change,
            "evaluated_phi_deg": float(evaluated_phi[position]),
        }
        for position, index in enumerate(indices)
    ]
    return evaluated_theta, evaluated_phi, records


def build_centerline_skeleton_masks(
    target: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Skeletonize binary 2D targets once and return masks on ``target.device``."""
    if target.ndim != 4:
        raise ValueError(
            f"Projection target must have shape [B,V,H,W], got {tuple(target.shape)}."
        )
    try:
        from skimage.morphology import skeletonize
    except ImportError as error:
        raise RuntimeError(
            "scikit-image is required when the centerline-weighted projection "
            "loss is enabled. Install requirements-stage2.txt or set "
            "--projection-loss-alpha 1."
        ) from error

    binary = target.detach().cpu().numpy() > float(threshold)
    skeletons = np.zeros_like(binary, dtype=np.float32)
    for batch_index in range(binary.shape[0]):
        for view_index in range(binary.shape[1]):
            skeletons[batch_index, view_index] = skeletonize(
                binary[batch_index, view_index]
            ).astype(np.float32)
    return torch.from_numpy(skeletons).to(
        device=target.device, dtype=target.dtype
    )


def masked_centerline_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    centerline_masks: torch.Tensor,
) -> torch.Tensor:
    """Mean squared residual over centerline pixels only."""
    if prediction.shape != target.shape or prediction.shape != centerline_masks.shape:
        raise ValueError(
            "Prediction, target, and centerline masks must have identical shapes, got "
            f"{tuple(prediction.shape)}, {tuple(target.shape)}, and "
            f"{tuple(centerline_masks.shape)}."
        )
    mask = centerline_masks.to(dtype=prediction.dtype)
    denominator = mask.sum()
    if float(denominator.detach().item()) <= 0.0:
        return F.mse_loss(prediction, target)
    return ((prediction - target).square() * mask).sum() / denominator


def projection_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    alpha: float,
    centerline_masks: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combine image MSE and the paper's centerline-weighted MSE term."""
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1].")
    image_mse = F.mse_loss(prediction, target)
    if alpha == 1.0:
        return image_mse, image_mse, image_mse
    if centerline_masks is None:
        raise ValueError("centerline_masks are required when alpha is less than 1.")
    centerline_mse = masked_centerline_mse(
        prediction, target, centerline_masks
    )
    total = alpha * image_mse + (1.0 - alpha) * centerline_mse
    return total, image_mse, centerline_mse


@torch.no_grad()
def predict_gcp_initial_centers(
    *,
    checkpoint_path: str | Path,
    case: Stage2ProjectionCase,
    view_indices: Sequence[int],
    volume_extent_m: float,
    device: torch.device,
    expected_model_config: Optional[Mapping[str, Any]] = None,
    selected_cone_vectors: Optional[np.ndarray] = None,
    evaluated_theta_deg: Optional[Sequence[float]] = None,
    evaluated_phi_deg: Optional[Sequence[float]] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Run the monocular GCP on exactly the first selected Stage-2 view."""
    # These imports are intentionally lazy: legacy FDK/BP runs should not load
    # the learned initializer or its target-generation helpers.
    from gcp_model import (
        GCPModelConfig,
        lift_depth_offsets_to_centers,
        load_gcp_checkpoint,
    )
    from gcp_targets import (
        detector_ray_box_intersections,
        scale_cone_vector_for_detector,
    )

    initialization_view_index = select_gcp_initialization_view(view_indices)
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    model = load_gcp_checkpoint(checkpoint, device=device)
    loaded_model_config = model.config.to_dict()
    if expected_model_config is not None:
        configured_model_config = GCPModelConfig.from_dict(
            expected_model_config
        ).to_dict()
        if configured_model_config != loaded_model_config:
            mismatches = {
                key: {
                    "configured": configured_model_config[key],
                    "checkpoint": loaded_model_config[key],
                }
                for key in configured_model_config
                if configured_model_config[key] != loaded_model_config[key]
            }
            raise ValueError(
                "Configured GCP model parameters do not match the checkpoint: "
                f"{mismatches}."
            )
    input_size = int(model.config.image_size)
    if int(model.config.in_channels) != 1:
        raise ValueError(
            "Stage-2 GCP initialization requires a one-channel checkpoint, got "
            f"in_channels={model.config.in_channels}."
        )

    image = torch.as_tensor(
        np.ascontiguousarray(case.images[initialization_view_index]),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0).unsqueeze(0)
    original_detector_shape = tuple(int(value) for value in image.shape[-2:])
    if original_detector_shape != (input_size, input_size):
        image = F.interpolate(
            image,
            size=(input_size, input_size),
            mode="bilinear",
            align_corners=False,
        )

    prediction = model(image)
    output_shape = tuple(int(value) for value in prediction["depth"].shape[-2:])
    selected_indices = validate_view_indices(view_indices, case.num_views)
    if selected_cone_vectors is None:
        initialization_cone_vector = case.cone_vectors(
            np.asarray([initialization_view_index], dtype=np.int64)
        )[0]
    else:
        selected_vectors = np.asarray(selected_cone_vectors, dtype=np.float32)
        if selected_vectors.shape != (selected_indices.size, 12):
            raise ValueError(
                "selected_cone_vectors must align with view_indices and have "
                f"shape {(selected_indices.size, 12)}, got "
                f"{selected_vectors.shape}."
            )
        initialization_cone_vector = selected_vectors[0]
    output_cone_vector = scale_cone_vector_for_detector(
        initialization_cone_vector,
        original_shape=original_detector_shape,
        target_shape=output_shape,
    )
    entry_zyx, exit_zyx, valid_mask = detector_ray_box_intersections(
        output_cone_vector,
        detector_shape=output_shape,
        volume_extent_m=float(volume_extent_m),
    )
    prediction_dtype = prediction["depth"].dtype
    entry = torch.from_numpy(entry_zyx).to(
        device=device, dtype=prediction_dtype
    ).unsqueeze(0)
    exit = torch.from_numpy(exit_zyx).to(
        device=device, dtype=prediction_dtype
    ).unsqueeze(0)
    centers = lift_depth_offsets_to_centers(
        prediction, entry, exit, clamp=True
    )[0]
    valid = torch.from_numpy(valid_mask.reshape(-1)).to(device=device)
    centers = centers[valid]
    if centers.shape[0] == 0:
        raise RuntimeError(
            "No rays from the GCP output grid intersect the reconstruction cube. "
            "Check the projection geometry and --volume-extent-m."
        )

    evaluated_theta = (
        float(case.theta_deg[initialization_view_index])
        if evaluated_theta_deg is None
        else float(tuple(evaluated_theta_deg)[0])
    )
    evaluated_phi = (
        float(case.phi_deg[initialization_view_index])
        if evaluated_phi_deg is None
        else float(tuple(evaluated_phi_deg)[0])
    )
    metadata = {
        "checkpoint": str(checkpoint),
        "initialization_view_index": int(initialization_view_index),
        "initialization_theta_deg": float(case.theta_deg[initialization_view_index]),
        "initialization_phi_deg": float(case.phi_deg[initialization_view_index]),
        "evaluated_initialization_theta_deg": evaluated_theta,
        "evaluated_initialization_phi_deg": evaluated_phi,
        "source_detector_shape": list(original_detector_shape),
        "input_shape": [input_size, input_size],
        "input_resize_mode": "bilinear_align_corners_false",
        "output_grid_shape": list(output_shape),
        "valid_output_rays": int(valid.sum().item()),
        "total_output_rays": int(valid.numel()),
        "center_coordinate_order": "normalized_zyx",
        "model_config": loaded_model_config,
        "configured_model_parameters": (
            None
            if expected_model_config is None
            else dict(expected_model_config)
        ),
    }
    return centers, metadata


@torch.no_grad()
def estimate_silhouette_gain(
    projector: AstraConeVecProjector,
    initial_volume: torch.Tensor,
    target_masks: torch.Tensor,
    target_level: float = 0.95,
) -> float:
    line_integrals = projector._forward_tensor(initial_volume)
    foreground_values = line_integrals[target_masks > 0.5]
    foreground_values = foreground_values[foreground_values > 0.0]
    if foreground_values.numel() == 0:
        return 1.0
    reference = float(torch.quantile(foreground_values, 0.5).item())
    return -math.log(max(1.0 - float(target_level), 1.0e-6)) / max(reference, 1.0e-8)


def cpu_state_dict(gaussians: GaussianModelAnisotropic) -> Dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in gaussians.state_dict().items()
    }


def restore_cpu_state(gaussians: GaussianModelAnisotropic, state: Dict[str, torch.Tensor], device: torch.device) -> None:
    gaussians.load_state_dict({key: value.to(device=device) for key, value in state.items()})


def export_gaussians(
    gaussians: GaussianModelAnisotropic,
    output_dir: Path,
    case: Stage2ProjectionCase,
    view_indices: np.ndarray,
    cone_vectors: np.ndarray,
    evaluated_theta_deg: np.ndarray,
    evaluated_phi_deg: np.ndarray,
    volume_extent_m: float,
    silhouette_gain: Optional[float],
) -> None:
    xyz_zyx = gaussians.get_xyz.detach().cpu()
    density = gaussians.get_density.detach().cpu()
    scale_zyx = gaussians.get_scaling.detach().cpu()
    rotation_wxyz = gaussians.get_rotation.detach().cpu()
    raw_state = cpu_state_dict(gaussians)

    world_xyz_m = torch.stack(
        [
            (xyz_zyx[:, 2] - 0.5) * float(volume_extent_m),
            (xyz_zyx[:, 1] - 0.5) * float(volume_extent_m),
            (xyz_zyx[:, 0] - 0.5) * float(volume_extent_m),
        ],
        dim=1,
    )
    checkpoint = {
        "format": "3dgr_car_stage2_gaussians_v1",
        "model_state": raw_state,
        "xyz_normalized_zyx": xyz_zyx,
        "xyz_world_m": world_xyz_m,
        "density": density,
        "scale_normalized_zyx": scale_zyx,
        "rotation_wxyz": rotation_wxyz,
        "sample_name": case.sample_name,
        "source_npz": str(case.path),
        "view_indices": torch.from_numpy(view_indices.copy()),
        "theta_deg": torch.from_numpy(case.theta_deg[view_indices].copy()),
        "phi_deg": torch.from_numpy(case.phi_deg[view_indices].copy()),
        "evaluated_theta_deg": torch.from_numpy(evaluated_theta_deg.copy()),
        "evaluated_phi_deg": torch.from_numpy(evaluated_phi_deg.copy()),
        "cone_vectors": torch.from_numpy(cone_vectors.copy()),
        "volume_extent_m": float(volume_extent_m),
        "projection_center_offset_m": (
            None
            if case.projection_center_offset_m is None
            else torch.from_numpy(case.projection_center_offset_m.copy())
        ),
        "silhouette_gain": silhouette_gain,
    }
    torch.save(checkpoint, output_dir / "gaussians.pt")
    np.savez_compressed(
        output_dir / "gaussians.npz",
        xyz_normalized_zyx=xyz_zyx.numpy(),
        xyz_world_m=world_xyz_m.numpy(),
        density=density.numpy(),
        scale_normalized_zyx=scale_zyx.numpy(),
        rotation_wxyz=rotation_wxyz.numpy(),
        view_indices=view_indices.astype(np.int32),
        theta_deg=case.theta_deg[view_indices],
        phi_deg=case.phi_deg[view_indices],
        evaluated_theta_deg=evaluated_theta_deg,
        evaluated_phi_deg=evaluated_phi_deg,
        cone_vectors=cone_vectors,
        volume_extent_m=np.asarray(volume_extent_m, dtype=np.float32),
    )


def save_volume(
    output_dir: Path,
    volume: torch.Tensor,
    volume_extent_m: float,
    save_nifti: bool = True,
) -> None:
    volume_np = volume[0].detach().cpu().numpy().astype(np.float32)
    np.save(output_dir / "reconstructed_volume_zyx.npy", volume_np)
    if not save_nifti:
        return
    try:
        import nibabel as nib

        spacing_m = float(volume_extent_m) / float(volume_np.shape[0])
        affine = np.diag([spacing_m * 1000.0, spacing_m * 1000.0, spacing_m * 1000.0, 1.0])
        nib.save(
            nib.Nifti1Image(np.transpose(volume_np, (2, 1, 0)), affine),
            output_dir / "reconstructed_volume_xyz.nii.gz",
        )
    except ImportError:
        print("[WARN] nibabel is unavailable; saved only reconstructed_volume_zyx.npy.")


def reprojection_metrics(
    target: np.ndarray,
    predictions: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_np = np.asarray(target, dtype=np.float32)
    prediction_np = np.asarray(predictions, dtype=np.float32)
    if target_np.shape != prediction_np.shape or target_np.ndim != 3:
        raise ValueError(
            "Reprojection targets and predictions must have matching [V,H,W] "
            f"shapes, got {target_np.shape} and {prediction_np.shape}."
        )
    absolute_error = np.abs(prediction_np - target_np).astype(np.float32)
    mse_per_view = np.mean(
        (prediction_np - target_np) ** 2, axis=(1, 2)
    ).astype(np.float32)
    intersection = np.sum(prediction_np * target_np, axis=(1, 2))
    dice_per_view = (
        (2.0 * intersection + 1.0e-6)
        / (
            np.sum(prediction_np, axis=(1, 2))
            + np.sum(target_np, axis=(1, 2))
            + 1.0e-6
        )
    ).astype(np.float32)
    return absolute_error, mse_per_view, dice_per_view


def save_reprojection_montage(
    *,
    output_path: Path,
    view_indices: np.ndarray,
    target: np.ndarray,
    predictions: np.ndarray,
    absolute_error: np.ndarray,
    mse_per_view: np.ndarray,
    dice_per_view: np.ndarray,
    target_title: str,
    prediction_title: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            f"[WARN] matplotlib is unavailable; did not save {output_path.name}."
        )
        return

    figure, axes = plt.subplots(
        len(view_indices),
        3,
        figsize=(9.0, 3.0 * len(view_indices)),
        squeeze=False,
    )
    column_titles = (target_title, prediction_title, "Absolute error")
    for column, title in enumerate(column_titles):
        axes[0, column].set_title(title)
    for row, view_index in enumerate(view_indices):
        images = (target[row], predictions[row], absolute_error[row])
        for column, image in enumerate(images):
            axes[row, column].imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
            axes[row, column].axis("off")
        axes[row, 0].set_ylabel(
            f"view {int(view_index)}\n"
            f"MSE={float(mse_per_view[row]):.4g}\n"
            f"soft Dice={float(dice_per_view[row]):.4f}"
        )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_input_view_reprojections(
    output_dir: Path,
    case: Stage2ProjectionCase,
    view_indices: np.ndarray,
    target: torch.Tensor,
    predictions: torch.Tensor,
    line_integrals: torch.Tensor,
    evaluated_theta_deg: np.ndarray,
    evaluated_phi_deg: np.ndarray,
) -> None:
    target_np = target[0].detach().cpu().numpy().astype(np.float32)
    prediction_np = predictions[0].detach().cpu().numpy().astype(np.float32)
    line_integrals_np = line_integrals[0].detach().cpu().numpy().astype(np.float32)
    absolute_error, mse_per_view, dice_per_view = reprojection_metrics(
        target_np, prediction_np
    )

    np.savez_compressed(
        output_dir / "input_view_reprojections.npz",
        view_indices=view_indices.astype(np.int32),
        theta_deg=case.theta_deg[view_indices],
        phi_deg=case.phi_deg[view_indices],
        evaluated_theta_deg=evaluated_theta_deg,
        evaluated_phi_deg=evaluated_phi_deg,
        target=target_np,
        predictions=prediction_np,
        line_integrals=line_integrals_np,
        absolute_error=absolute_error,
        mse_per_view=mse_per_view,
        soft_dice_per_view=dice_per_view,
    )

    save_reprojection_montage(
        output_path=output_dir / "input_view_reprojections.png",
        view_indices=view_indices,
        target=target_np,
        predictions=prediction_np,
        absolute_error=absolute_error,
        mse_per_view=mse_per_view,
        dice_per_view=dice_per_view,
        target_title="Input target",
        prediction_title="Final reprojection",
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> Tuple[argparse.Namespace, OptimizationParams]:
    parser = argparse.ArgumentParser(
        description="Run initialized 3D Gaussian reconstruction on a Stage-2 NPZ case."
    )
    optimization = OptimizationParams(parser)
    parser.set_defaults(iterations=8000, densify_until_iter=8000)
    parser.add_argument("--input", required=True, help="Stage-2 case NPZ, e.g. rca_0001.npz")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--view-indices",
        nargs="+",
        type=int,
        default=[0, 1],
        help="One or more stored view indices used for reconstruction.",
    )
    parser.add_argument(
        "--novel-view-indices",
        nargs="*",
        type=int,
        default=None,
        help="Defaults to all views not selected for reconstruction; pass no values to disable.",
    )
    parser.add_argument("--volume-size", type=int, default=128)
    parser.add_argument("--volume-extent-m", type=float, default=None)
    parser.add_argument(
        "--source-origin-distance-m",
        type=float,
        default=DEFAULT_SOURCE_ORIGIN_DISTANCE_M,
        help="The Stage-2 renderer hard-codes this to 0.75 m.",
    )
    parser.add_argument(
        "--fallback-detector-pixel-spacing-mm",
        type=float,
        default=None,
        help=(
            "Detector pixel spacing in mm used only when the projection NPZ "
            "has no imager_pixel_spacing key."
        ),
    )
    parser.add_argument(
        "--fallback-sid-m",
        type=float,
        default=None,
        help="SID in metres used only when the projection NPZ has no sid key.",
    )
    parser.add_argument(
        "--view-direction-theta-change-deg",
        type=float,
        default=0.0,
        help=(
            "Evaluation-only fixed theta error applied to the assumed geometry "
            "of every selected input view. Projection images are unchanged."
        ),
    )
    parser.add_argument(
        "--view-direction-phi-change-deg",
        type=float,
        default=0.0,
        help=(
            "Evaluation-only fixed phi error applied to the assumed geometry "
            "of every selected input view. Projection images are unchanged."
        ),
    )
    parser.add_argument("--num-init-gaussians", type=int, default=10000)
    parser.add_argument("--air-threshold", type=float, default=0.05)
    parser.add_argument("--initial-density", type=float, default=0.04)
    parser.add_argument("--initial-sigma", type=float, default=0.01)
    parser.add_argument("--init-method", choices=["fdk", "bp", "gcp"], default="fdk")
    parser.add_argument(
        "--gcp-checkpoint",
        default=None,
        help=(
            "Trained monocular Gaussian Center Predictor checkpoint. Required "
            "when --init-method gcp; only the first --view-indices value is "
            "used by the predictor."
        ),
    )
    parser.add_argument(
        "--expected-gcp-model-config-json",
        default=None,
        help=(
            "Optional JSON object describing the expected GCP architecture. "
            "Evaluation fails if it differs from the loaded checkpoint."
        ),
    )
    parser.add_argument("--target-type", choices=["auto", "mask", "line-integral"], default="auto")
    parser.add_argument("--silhouette-gain", type=float, default=None)
    parser.add_argument("--silhouette-target-level", type=float, default=0.95)
    parser.add_argument(
        "--projection-loss-alpha",
        type=float,
        default=None,
        help=(
            "Weight of full-image MSE; the remaining weight is centerline MSE. "
            "Defaults to 0.5 for GCP and 1.0 for FDK/BP."
        ),
    )
    parser.add_argument(
        "--centerline-threshold",
        type=float,
        default=0.5,
        help="Threshold used to binarize target projections before 2D skeletonization.",
    )
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--early-stop-checks", type=int, default=7)
    parser.add_argument("--no-densify", action="store_true")
    parser.add_argument(
        "--record-optimization-time",
        "--record_optimization_time",
        dest="record_optimization_time",
        action="store_true",
        help=(
            "Record CUDA-synchronized wall time for the Gaussian optimization "
            "loop only in optimization_timing.json."
        ),
    )
    parser.add_argument(
        "--monitor-gif-frames",
        "--monitor_gif_frames",
        dest="monitor_gif_frames",
        type=int,
        default=24,
    )
    parser.add_argument(
        "--monitor-gif-fps",
        "--monitor_gif_fps",
        dest="monitor_gif_fps",
        type=int,
        default=5,
    )
    prediction_threshold_group = parser.add_mutually_exclusive_group()
    prediction_threshold_group.add_argument(
        "--prediction-threshold",
        "--volume-gif-isovalue",
        dest="prediction_threshold",
        type=float,
        default=None,
        help=(
            "Fixed raw-density foreground threshold and GIF isovalue. "
            "--volume-gif-isovalue is retained as a compatible alias."
        ),
    )
    prediction_threshold_group.add_argument(
        "--prediction-threshold-percentile",
        type=float,
        default=None,
        help=(
            "Per-case percentile of strictly positive final-volume voxels used "
            "as the foreground threshold and GIF isovalue (default: P97)."
        ),
    )
    parser.add_argument("--no-volume-gif", action="store_true")
    parser.add_argument(
        "--evaluation-cache-only",
        action="store_true",
        help=(
            "Write only reconstructed_volume_zyx.npy, timing JSON, and run metadata "
            "for temporary split evaluation; skip NIfTI, Gaussian, reprojection, "
            "montage, novel-view, and GIF artifacts."
        ),
    )
    parser.add_argument("--gpu-index", type=int, default=0)
    args = parser.parse_args(argv)
    if args.expected_gcp_model_config_json is not None:
        try:
            expected_model_config = json.loads(
                args.expected_gcp_model_config_json
            )
        except json.JSONDecodeError as error:
            parser.error(
                "--expected-gcp-model-config-json must contain valid JSON: "
                f"{error}."
            )
        if not isinstance(expected_model_config, Mapping):
            parser.error(
                "--expected-gcp-model-config-json must contain a JSON object."
            )
        args.expected_gcp_model_config = dict(expected_model_config)
    else:
        args.expected_gcp_model_config = None
    if args.init_method == "gcp" and args.gcp_checkpoint is None:
        parser.error("--gcp-checkpoint is required when --init-method gcp.")
    if args.projection_loss_alpha is not None and not (
        0.0 <= float(args.projection_loss_alpha) <= 1.0
    ):
        parser.error("--projection-loss-alpha must be in [0, 1].")
    if not 0.0 <= float(args.centerline_threshold) <= 1.0:
        parser.error("--centerline-threshold must be in [0, 1].")
    for option, value in (
        (
            "--fallback-detector-pixel-spacing-mm",
            args.fallback_detector_pixel_spacing_mm,
        ),
        ("--fallback-sid-m", args.fallback_sid_m),
    ):
        if value is not None and (
            not math.isfinite(float(value)) or float(value) <= 0.0
        ):
            parser.error(f"{option} must be finite and positive.")
    for option, value in (
        (
            "--view-direction-theta-change-deg",
            args.view_direction_theta_change_deg,
        ),
        (
            "--view-direction-phi-change-deg",
            args.view_direction_phi_change_deg,
        ),
    ):
        if not math.isfinite(float(value)):
            parser.error(f"{option} must be finite.")
    if args.prediction_threshold is None and args.prediction_threshold_percentile is None:
        args.prediction_threshold_percentile = float(
            DEFAULT_VOLUME_GIF_POSITIVE_PERCENTILE
        )
    if (
        args.prediction_threshold_percentile is not None
        and not 0.0 <= args.prediction_threshold_percentile < 100.0
    ):
        parser.error("--prediction-threshold-percentile must be in [0, 100).")
    return args, optimization


def main(argv: Optional[Sequence[str]] = None) -> int:
    args, optimization_group = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("This repository's Gaussian implementation requires CUDA.")
    torch.cuda.set_device(int(args.gpu_index))
    device = torch.device("cuda", int(args.gpu_index))

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    case = load_stage2_projection_case(
        args.input,
        source_origin_distance_m=float(args.source_origin_distance_m),
        fallback_detector_pixel_spacing_mm=(
            args.fallback_detector_pixel_spacing_mm
        ),
        fallback_sid_m=args.fallback_sid_m,
    )
    view_indices = validate_view_indices(args.view_indices, case.num_views)
    target = torch.from_numpy(case.images[view_indices]).to(device=device).unsqueeze(0)

    target_type = str(args.target_type)
    if target_type == "auto":
        target_type = "mask" if case.is_binary_mask else "line-integral"
    volume_extent_m = (
        float(args.volume_extent_m)
        if args.volume_extent_m is not None
        else float(case.isocenter_fov_m)
    )
    theta_change_deg = float(args.view_direction_theta_change_deg)
    phi_change_deg = float(args.view_direction_phi_change_deg)
    evaluated_theta_deg, evaluated_phi_deg, view_direction_records = (
        fixed_view_direction_records(
            case,
            view_indices,
            theta_change_deg=theta_change_deg,
            phi_change_deg=phi_change_deg,
        )
    )
    cone_vectors = case.cone_vectors(
        view_indices,
        theta_change_deg=theta_change_deg,
        phi_change_deg=phi_change_deg,
    )
    projector = AstraConeVecProjector(
        vectors=cone_vectors,
        detector_shape=case.detector_shape,
        volume_size=int(args.volume_size),
        volume_extent_m=volume_extent_m,
        gpu_index=int(args.gpu_index),
    )

    print(f"Case: {case.sample_name}")
    print(f"Views: {view_indices.tolist()}")
    for position, index in enumerate(view_indices):
        print(
            f"  {int(index)}: {case.clinical_views[index]} | "
            f"stored theta={float(case.theta_deg[index]):.1f} deg, "
            f"phi={float(case.phi_deg[index]):.1f} deg | assumed "
            f"theta={float(evaluated_theta_deg[position]):.1f} deg, "
            f"phi={float(evaluated_phi_deg[position]):.1f} deg"
        )
    print(f"Input shape: {tuple(target.shape)}; target type: {target_type}")
    print(f"Reconstruction: {args.volume_size}^3 over {volume_extent_m:.6f} m")

    projection_loss_alpha = resolve_projection_loss_alpha(
        str(args.init_method), args.projection_loss_alpha
    )
    gcp_metadata: Optional[Dict[str, Any]] = None
    initial_volume: Optional[torch.Tensor] = None
    gaussians = GaussianModelAnisotropic()
    if args.init_method in {"fdk", "bp"}:
        initial_volume = normalize_initial_volume(
            projector.reconstruct(target, method=str(args.init_method))
        )
        if not args.evaluation_cache_only:
            np.save(
                output_dir / "initial_fbp_volume_zyx.npy",
                initial_volume[0].cpu().numpy(),
            )
        gaussians.create_from_fbp(
            initial_volume,
            air_threshold=float(args.air_threshold),
            ini_density=float(args.initial_density),
            ini_sigma=float(args.initial_sigma),
            spatial_lr_scale=1.0,
            num_samples=int(args.num_init_gaussians),
        )
    else:
        predicted_centers, gcp_metadata = predict_gcp_initial_centers(
            checkpoint_path=str(args.gcp_checkpoint),
            case=case,
            view_indices=view_indices,
            volume_extent_m=volume_extent_m,
            device=device,
            expected_model_config=args.expected_gcp_model_config,
            selected_cone_vectors=cone_vectors,
            evaluated_theta_deg=evaluated_theta_deg,
            evaluated_phi_deg=evaluated_phi_deg,
        )
        if not args.evaluation_cache_only:
            np.save(
                output_dir / "initial_gcp_centers_normalized_zyx.npy",
                predicted_centers.detach().cpu().numpy(),
            )
        gaussians.create_from_predicted_centers(
            predicted_centers,
            ini_density=float(args.initial_density),
            ini_sigma=float(args.initial_sigma),
            spatial_lr_scale=1.0,
            scale_from_nearest_neighbour=True,
        )
        print(
            "GCP initialization: "
            f"view={gcp_metadata['initialization_view_index']} "
            f"centers={gaussians.get_gaussians_num} "
            f"checkpoint={gcp_metadata['checkpoint']}"
        )
    num_initial_gaussians = int(gaussians.get_gaussians_num)
    gaussians.training_setup(optimization_group.extract(args))

    grid = create_grid_3d(
        int(args.volume_size), int(args.volume_size), int(args.volume_size), device=device
    )
    if initial_volume is None:
        with torch.no_grad():
            initial_volume = gaussians.grid_sample(
                grid, expand=[5, 15, 15]
            ).squeeze(-1)
        if not args.evaluation_cache_only:
            np.save(
                output_dir / "initial_gcp_volume_zyx.npy",
                initial_volume[0].detach().cpu().numpy(),
            )

    silhouette_gain: Optional[float] = None
    if target_type == "mask":
        silhouette_gain = (
            float(args.silhouette_gain)
            if args.silhouette_gain is not None
            else estimate_silhouette_gain(
                projector,
                initial_volume,
                target,
                target_level=float(args.silhouette_target_level),
            )
        )
        print(f"Silhouette gain: {silhouette_gain:.6g}")

    centerline_masks: Optional[torch.Tensor] = None
    if projection_loss_alpha < 1.0:
        centerline_masks = build_centerline_skeleton_masks(
            target, threshold=float(args.centerline_threshold)
        )
        print(
            "Projection loss: "
            f"alpha={projection_loss_alpha:.6g} "
            f"centerline_pixels={int(centerline_masks.sum().item())}"
        )
    else:
        print("Projection loss: full-image MSE (alpha=1)")

    best_loss = float("inf")
    best_iteration = -1
    best_state: Optional[Dict[str, torch.Tensor]] = None
    stale_checks = 0
    post_initialization_start_time = time.perf_counter()
    optimization_iterations_completed = 0
    optimization_elapsed_seconds: Optional[float] = None
    optimization_seconds_per_iteration: Optional[float] = None
    optimization_timer_start: Optional[float] = None
    if args.record_optimization_time:
        torch.cuda.synchronize(device)
        optimization_timer_start = time.perf_counter()

    for iteration in range(int(args.iterations)):
        gaussians.update_learning_rate(iteration)
        gaussian_grid = gaussians.grid_sample(grid, expand=[5, 15, 15])
        reconstructed_volume = gaussian_grid.squeeze(-1)
        predicted_line_integrals = projector(reconstructed_volume)
        if target_type == "mask":
            prediction = silhouette_from_line_integrals(
                predicted_line_integrals, gain=float(silhouette_gain)
            )
        else:
            prediction = predicted_line_integrals
        loss, image_mse, centerline_mse = projection_reconstruction_loss(
            prediction,
            target,
            alpha=projection_loss_alpha,
            centerline_masks=centerline_masks,
        )
        loss.backward()

        if (
            not args.no_densify
            and iteration < int(args.densify_until_iter)
            and iteration > int(args.densify_from_iter)
            and iteration % int(args.densification_interval) == 0
            and gaussians._xyz.grad is not None
        ):
            gaussians.densify_and_prune(
                float(args.densify_grad_threshold), 0.005, 1.5
            )

        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)
        optimization_iterations_completed = iteration + 1

        should_check = iteration == 0 or (iteration + 1) % int(args.log_every) == 0
        if should_check:
            loss_value = float(loss.detach().item())
            image_mse_value = float(image_mse.detach().item())
            centerline_mse_value = float(centerline_mse.detach().item())
            psnr = -10.0 * math.log10(max(image_mse_value, 1.0e-12))
            print(
                f"iteration={iteration + 1} loss={loss_value:.7g} "
                f"image_mse={image_mse_value:.7g} "
                f"centerline_mse={centerline_mse_value:.7g} "
                f"psnr={psnr:.3f} gaussians={gaussians.get_gaussians_num}"
            )
            if loss_value < best_loss:
                best_loss = loss_value
                best_iteration = iteration
                best_state = cpu_state_dict(gaussians)
                stale_checks = 0
            else:
                stale_checks += 1
                if int(args.early_stop_checks) > 0 and stale_checks >= int(args.early_stop_checks):
                    print(f"Early stopping; best checked iteration was {best_iteration + 1}.")
                    break

    if optimization_timer_start is not None:
        torch.cuda.synchronize(device)
        optimization_elapsed_seconds = float(
            time.perf_counter() - optimization_timer_start
        )
        if optimization_iterations_completed > 0:
            optimization_seconds_per_iteration = float(
                optimization_elapsed_seconds / optimization_iterations_completed
            )
        timing = {
            "format": "3dgr_car_optimization_timing_v1",
            "scope": "gaussian_optimization_loop_only",
            "cuda_synchronized": True,
            "sample_name": case.sample_name,
            "view_indices": view_indices.tolist(),
            "init_method": str(args.init_method),
            "gcp_initialization_view_index": (
                None
                if gcp_metadata is None
                else int(gcp_metadata["initialization_view_index"])
            ),
            "projection_loss_alpha": float(projection_loss_alpha),
            "iterations_requested": int(args.iterations),
            "iterations_completed": int(optimization_iterations_completed),
            "early_stopped": bool(
                optimization_iterations_completed < int(args.iterations)
            ),
            "elapsed_seconds": optimization_elapsed_seconds,
            "seconds_per_iteration": optimization_seconds_per_iteration,
            "volume_size": int(args.volume_size),
            "num_initial_gaussians": int(num_initial_gaussians),
            "num_gaussians_after_optimization": int(gaussians.get_gaussians_num),
            "densification_enabled": bool(not args.no_densify),
            "gpu_index": int(args.gpu_index),
            "gpu_name": torch.cuda.get_device_name(int(args.gpu_index)),
            "pytorch_version": str(torch.__version__),
            "pytorch_cuda_version": str(torch.version.cuda),
        }
        timing_path = output_dir / "optimization_timing.json"
        timing_path.write_text(json.dumps(timing, indent=2), encoding="utf-8")
        print(
            "Optimization time: "
            f"{optimization_elapsed_seconds:.6f} s for "
            f"{optimization_iterations_completed} iterations "
            f"({optimization_seconds_per_iteration or 0.0:.6f} s/iteration)"
        )
        print(f"Saved optimization timing: {timing_path}")

    if best_state is None:
        best_state = cpu_state_dict(gaussians)
    restore_cpu_state(gaussians, best_state, device=device)

    with torch.no_grad():
        final_volume = gaussians.grid_sample(grid, expand=[15, 15, 15]).squeeze(-1)
    save_volume(
        output_dir,
        final_volume,
        volume_extent_m=volume_extent_m,
        save_nifti=not args.evaluation_cache_only,
    )
    volume_gif_isovalue: Optional[float] = None
    if not args.evaluation_cache_only:
        export_gaussians(
            gaussians=gaussians,
            output_dir=output_dir,
            case=case,
            view_indices=view_indices,
            cone_vectors=cone_vectors,
            evaluated_theta_deg=evaluated_theta_deg,
            evaluated_phi_deg=evaluated_phi_deg,
            volume_extent_m=volume_extent_m,
            silhouette_gain=silhouette_gain,
        )

    if not args.evaluation_cache_only:
        with torch.no_grad():
            input_line_integrals = projector._forward_tensor(final_volume)
            if target_type == "mask":
                input_reprojections = silhouette_from_line_integrals(
                    input_line_integrals, gain=float(silhouette_gain)
                )
            else:
                input_reprojections = input_line_integrals
        save_input_view_reprojections(
            output_dir=output_dir,
            case=case,
            view_indices=view_indices,
            target=target,
            predictions=input_reprojections,
            line_integrals=input_line_integrals,
            evaluated_theta_deg=evaluated_theta_deg,
            evaluated_phi_deg=evaluated_phi_deg,
        )

    if args.evaluation_cache_only:
        novel_indices = np.empty((0,), dtype=np.int64)
    elif args.novel_view_indices is None:
        selected = set(int(index) for index in view_indices)
        novel_indices = np.asarray(
            [index for index in range(case.num_views) if index not in selected],
            dtype=np.int64,
        )
    else:
        novel_indices = np.asarray(args.novel_view_indices, dtype=np.int64)
    if novel_indices.size:
        novel_indices = validate_view_indices(novel_indices, case.num_views)
        novel_projector = AstraConeVecProjector(
            vectors=case.cone_vectors(novel_indices),
            detector_shape=case.detector_shape,
            volume_size=int(args.volume_size),
            volume_extent_m=volume_extent_m,
            gpu_index=int(args.gpu_index),
        )
        with torch.no_grad():
            novel_line_integrals = novel_projector._forward_tensor(final_volume)
            if target_type == "mask":
                novel_predictions = silhouette_from_line_integrals(
                    novel_line_integrals, gain=float(silhouette_gain)
                )
            else:
                novel_predictions = novel_line_integrals
        novel_target_np = np.asarray(
            case.images[novel_indices], dtype=np.float32
        )
        novel_prediction_np = (
            novel_predictions[0].detach().cpu().numpy().astype(np.float32)
        )
        novel_line_integrals_np = (
            novel_line_integrals[0].detach().cpu().numpy().astype(np.float32)
        )
        novel_absolute_error, novel_mse_per_view, novel_dice_per_view = (
            reprojection_metrics(novel_target_np, novel_prediction_np)
        )
        np.savez_compressed(
            output_dir / "novel_views.npz",
            view_indices=novel_indices.astype(np.int32),
            theta_deg=case.theta_deg[novel_indices],
            phi_deg=case.phi_deg[novel_indices],
            predictions=novel_prediction_np,
            target_masks=novel_target_np,
            line_integrals=novel_line_integrals_np,
            absolute_error=novel_absolute_error,
            mse_per_view=novel_mse_per_view,
            soft_dice_per_view=novel_dice_per_view,
        )
        save_reprojection_montage(
            output_path=output_dir / "novel_view_reprojections.png",
            view_indices=novel_indices,
            target=novel_target_np,
            predictions=novel_prediction_np,
            absolute_error=novel_absolute_error,
            mse_per_view=novel_mse_per_view,
            dice_per_view=novel_dice_per_view,
            target_title="Novel-view target",
            prediction_title="Novel-view reprojection",
        )
        novel_projector.close()

    if not args.no_volume_gif and not args.evaluation_cache_only:
        volume_gif_path = output_dir / "reconstructed_volume.gif"
        volume_gif_isovalue = save_reconstructed_volume_gif(
            volume_zyx=final_volume[0].detach().cpu().numpy(),
            volume_extent_m=volume_extent_m,
            source_npz=case.path,
            out_path=volume_gif_path,
            num_frames=int(args.monitor_gif_frames),
            fps=int(args.monitor_gif_fps),
            isovalue=args.prediction_threshold,
            positive_percentile=(
                DEFAULT_VOLUME_GIF_POSITIVE_PERCENTILE
                if args.prediction_threshold_percentile is None
                else args.prediction_threshold_percentile
            ),
        )
        threshold_description = (
            f"fixed={float(args.prediction_threshold):.7g}"
            if args.prediction_threshold is not None
            else f"P{float(args.prediction_threshold_percentile):g}"
        )
        print(
            f"Saved reconstructed-volume GIF: {volume_gif_path} "
            f"({threshold_description}, effective isovalue="
            f"{float(volume_gif_isovalue):.7g})"
        )

    metadata = {
        "sample_name": case.sample_name,
        "source_npz": str(case.path),
        "view_indices": view_indices.tolist(),
        "theta_deg": case.theta_deg[view_indices].tolist(),
        "phi_deg": case.phi_deg[view_indices].tolist(),
        "evaluation_view_directions": {
            "accurate": bool(theta_change_deg == 0.0 and phi_change_deg == 0.0),
            "distribution": "fixed_per_selected_view",
            "theta_change_deg": theta_change_deg,
            "phi_change_deg": phi_change_deg,
            "num_perturbed_views": (
                0
                if theta_change_deg == 0.0 and phi_change_deg == 0.0
                else int(len(view_indices))
            ),
            "views": view_direction_records,
            "gcp_lifting_geometry": "assumed_geometry_first_selected_view",
            "gaussian_optimization_geometry": "assumed_geometry_all_selected_views",
            "projection_images_changed": False,
            "novel_view_geometry_changed": False,
        },
        "target_type": target_type,
        "init_method": str(args.init_method),
        "requested_num_init_gaussians": int(args.num_init_gaussians),
        "num_initial_gaussians": int(num_initial_gaussians),
        "air_threshold": float(args.air_threshold),
        "initial_density": float(args.initial_density),
        "initial_sigma": float(args.initial_sigma),
        "scale_from_nearest_neighbour": bool(args.init_method == "gcp"),
        "gcp": gcp_metadata,
        "gcp_checkpoint": (
            None if gcp_metadata is None else str(gcp_metadata["checkpoint"])
        ),
        "gcp_initialization_view_index": (
            None
            if gcp_metadata is None
            else int(gcp_metadata["initialization_view_index"])
        ),
        "projection_loss_alpha_requested": args.projection_loss_alpha,
        "projection_loss_alpha": float(projection_loss_alpha),
        "projection_loss_formula": (
            "alpha * full_image_mse + (1 - alpha) * centerline_masked_mse"
        ),
        "centerline_threshold": float(args.centerline_threshold),
        "centerline_mask_pixels": (
            None
            if centerline_masks is None
            else int(centerline_masks.sum().item())
        ),
        "volume_size": int(args.volume_size),
        "volume_extent_m": float(volume_extent_m),
        "source_origin_distance_m": float(case.source_origin_distance_m),
        "fallback_detector_pixel_spacing_mm": (
            None
            if args.fallback_detector_pixel_spacing_mm is None
            else float(args.fallback_detector_pixel_spacing_mm)
        ),
        "fallback_sid_m": (
            None if args.fallback_sid_m is None else float(args.fallback_sid_m)
        ),
        "detector_origin_distance_m": float(case.detector_origin_distance_m),
        "detector_pixel_spacing_m": float(case.detector_pixel_spacing_m),
        "best_checked_iteration": int(best_iteration + 1),
        "best_checked_loss": float(best_loss),
        "post_initialization_pipeline_elapsed_seconds": float(
            time.perf_counter() - post_initialization_start_time
        ),
        "record_optimization_time": bool(args.record_optimization_time),
        "evaluation_cache_only": bool(args.evaluation_cache_only),
        "optimization_iterations_completed": int(
            optimization_iterations_completed
        ),
        "optimization_elapsed_seconds": optimization_elapsed_seconds,
        "optimization_seconds_per_iteration": optimization_seconds_per_iteration,
        "num_gaussians": int(gaussians.get_gaussians_num),
        "silhouette_gain": silhouette_gain,
        "volume_gif_frames": int(args.monitor_gif_frames),
        "volume_gif_fps": int(args.monitor_gif_fps),
        "prediction_threshold": args.prediction_threshold,
        "prediction_threshold_percentile": args.prediction_threshold_percentile,
        "volume_gif_isovalue": volume_gif_isovalue,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    projector.close()
    if not args.evaluation_cache_only:
        print(f"Saved Gaussian checkpoint: {output_dir / 'gaussians.pt'}")
        print(f"Saved portable Gaussian arrays: {output_dir / 'gaussians.npz'}")
    else:
        print("Evaluation cache-only mode: skipped persistent visualization/model artifacts.")
    print(f"Saved reconstructed volume under: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
