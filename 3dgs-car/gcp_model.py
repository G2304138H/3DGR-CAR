"""Monocular Gaussian-centre predictor used to initialise Stage 2.

The predictor follows the representation described in 3DGR-CAR: a 2-D
U-Net maps one projection to a pooled grid of depths and three-dimensional
offsets.  Coordinates and offsets in this module use normalised ``(z, y, x)``
order so the lifted points can be passed directly to the repository's
volumetric Gaussian representation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class GCPModelConfig:
    """Serializable architecture and output-parameterisation settings."""

    image_size: int = 128
    in_channels: int = 1
    base_channels: int = 32
    num_levels: int = 4
    alpha: int = 2
    offset_scale: float = 0.1
    norm_groups: int = 8
    dropout: float = 0.0

    def __post_init__(self) -> None:
        integer_fields = {
            "image_size": self.image_size,
            "in_channels": self.in_channels,
            "base_channels": self.base_channels,
            "num_levels": self.num_levels,
            "alpha": self.alpha,
            "norm_groups": self.norm_groups,
        }
        for name, value in integer_fields.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}.")
        if self.num_levels < 2:
            raise ValueError("num_levels must be at least 2.")
        if self.offset_scale < 0.0:
            raise ValueError("offset_scale must be non-negative.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

    @property
    def downsample_factor(self) -> int:
        """Alias used by inference code when constructing the ray grid."""

        return self.alpha

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "GCPModelConfig":
        if not isinstance(values, Mapping):
            raise TypeError("model_config must be a mapping.")
        config = dict(values)
        if "downsample_factor" in config:
            downsample_factor = config.pop("downsample_factor")
            if "alpha" in config and config["alpha"] != downsample_factor:
                raise ValueError("model_config alpha and downsample_factor disagree.")
            config["alpha"] = downsample_factor
        if "input_channels" in config:
            input_channels = config.pop("input_channels")
            if "in_channels" in config and config["in_channels"] != input_channels:
                raise ValueError(
                    "model_config in_channels and input_channels disagree."
                )
            config["in_channels"] = input_channels

        allowed = {field.name for field in fields(cls)}
        unexpected = sorted(set(config).difference(allowed))
        if unexpected:
            raise ValueError(
                f"Unexpected GCP model configuration fields: {unexpected}."
            )
        return cls(**config)


def _group_count(channels: int, requested_groups: int) -> int:
    for groups in range(min(channels, requested_groups), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class _DoubleConv(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, norm_groups: int, dropout: float,
    ) -> None:
        super().__init__()
        groups = _group_count(out_channels, norm_groups)
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))
        layers.extend(
            [
                nn.Conv2d(
                    out_channels, out_channels, kernel_size=3, padding=1, bias=False,
                ),
                nn.GroupNorm(groups, out_channels),
                nn.SiLU(inplace=True),
            ]
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, image: Tensor) -> Tensor:
        return self.layers(image)


class MonocularGaussianCenterPredictor(nn.Module):
    """Predict depth and bounded ZYX offsets from one 2-D projection.

    ``forward`` returns a mapping with ``depth`` shaped ``[B,1,h,w]`` and
    ``offsets`` shaped ``[B,3,h,w]``.  The offset channels are ordered ZYX.
    Both outputs are produced after the four-channel prediction head is
    average-pooled by ``config.alpha``.
    """

    def __init__(self, config: Optional[GCPModelConfig] = None) -> None:
        super().__init__()
        self.config = config if config is not None else GCPModelConfig()

        channels = [
            self.config.base_channels * (2 ** level)
            for level in range(self.config.num_levels)
        ]
        encoders = []
        input_channels = self.config.in_channels
        for output_channels in channels:
            encoders.append(
                _DoubleConv(
                    input_channels,
                    output_channels,
                    self.config.norm_groups,
                    self.config.dropout,
                )
            )
            input_channels = output_channels
        self.encoders = nn.ModuleList(encoders)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        bottleneck_channels = channels[-1] * 2
        self.bottleneck = _DoubleConv(
            channels[-1],
            bottleneck_channels,
            self.config.norm_groups,
            self.config.dropout,
        )

        upconvolutions = []
        decoders = []
        current_channels = bottleneck_channels
        for skip_channels in reversed(channels):
            upconvolutions.append(
                nn.ConvTranspose2d(
                    current_channels, skip_channels, kernel_size=2, stride=2,
                )
            )
            decoders.append(
                _DoubleConv(
                    skip_channels * 2,
                    skip_channels,
                    self.config.norm_groups,
                    self.config.dropout,
                )
            )
            current_channels = skip_channels
        self.upconvolutions = nn.ModuleList(upconvolutions)
        self.decoders = nn.ModuleList(decoders)
        self.prediction_head = nn.Conv2d(channels[0], 4, kernel_size=1)

    def forward(self, image: Tensor) -> Dict[str, Tensor]:
        if image.ndim != 4:
            raise ValueError(
                f"image must have shape [B,C,H,W], got {tuple(image.shape)}."
            )
        if image.shape[1] != self.config.in_channels:
            raise ValueError(
                f"Expected {self.config.in_channels} input channels, got {image.shape[1]}."
            )
        minimum_unet_size = 2 ** self.config.num_levels
        minimum_size = max(self.config.alpha, minimum_unet_size)
        if image.shape[-2] < minimum_size or image.shape[-1] < minimum_size:
            raise ValueError(
                "Input spatial dimensions are too small for the configured U-Net and "
                f"pooling factor; expected at least {minimum_size}, got "
                f"{tuple(image.shape[-2:])}."
            )

        skips = []
        features = image
        for encoder in self.encoders:
            features = encoder(features)
            skips.append(features)
            features = self.pool(features)
        features = self.bottleneck(features)

        for upconvolution, decoder, skip in zip(
            self.upconvolutions, self.decoders, reversed(skips),
        ):
            features = upconvolution(features)
            if features.shape[-2:] != skip.shape[-2:]:
                features = F.interpolate(
                    features,
                    size=skip.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            features = decoder(torch.cat((skip, features), dim=1))

        raw_prediction = self.prediction_head(features)
        if self.config.alpha > 1:
            raw_prediction = F.avg_pool2d(
                raw_prediction, kernel_size=self.config.alpha, stride=self.config.alpha,
            )
        depth = torch.sigmoid(raw_prediction[:, :1])
        offsets = self.config.offset_scale * torch.tanh(raw_prediction[:, 1:])
        return {"depth": depth, "offsets": offsets}

    def lift(
        self,
        prediction: Union[Mapping[str, Tensor], Tensor],
        ray_box_entry: Tensor,
        ray_box_exit: Tensor,
        *,
        clamp: bool = True,
    ) -> Tensor:
        return lift_depth_offsets_to_centers(
            prediction, ray_box_entry, ray_box_exit, clamp=clamp,
        )


def _split_prediction(
    prediction: Union[Mapping[str, Tensor], Tensor],
) -> Tuple[Tensor, Tensor]:
    if isinstance(prediction, Mapping):
        missing = {"depth", "offsets"}.difference(prediction)
        if missing:
            raise KeyError(f"Prediction mapping is missing keys: {sorted(missing)}.")
        return prediction["depth"], prediction["offsets"]
    if not torch.is_tensor(prediction):
        raise TypeError("prediction must be a mapping or a torch Tensor.")
    if prediction.ndim != 4 or prediction.shape[1] != 4:
        raise ValueError(
            "Tensor predictions must have shape [B,4,H,W] with depth first, got "
            f"{tuple(prediction.shape)}."
        )
    return prediction[:, :1], prediction[:, 1:]


def lift_depth_offsets_to_centers(
    depth_or_prediction: Union[Mapping[str, Tensor], Tensor],
    offsets_or_ray_box_entry: Tensor,
    ray_box_entry_or_exit: Tensor,
    ray_box_exit: Optional[Tensor] = None,
    *,
    clamp: bool = True,
) -> Tensor:
    """Lift pooled detector predictions into flattened normalised ZYX points.

    Two equivalent call forms are supported::

        lift_depth_offsets_to_centers(prediction, entry, exit)
        lift_depth_offsets_to_centers(depth, offsets, entry, exit)

    ``entry`` and ``exit`` must have shape ``[B,h,w,3]`` and contain the
    normalised ZYX intersections of each detector ray with the reconstruction
    box.  A depth of zero selects ``entry`` and one selects ``exit``.  Flattening
    uses row-major detector order.
    """

    if ray_box_exit is None:
        depth, offsets = _split_prediction(depth_or_prediction)
        ray_box_entry = offsets_or_ray_box_entry
        ray_box_exit = ray_box_entry_or_exit
    else:
        if isinstance(depth_or_prediction, Mapping):
            raise TypeError(
                "The four-argument form expects depth as its first argument."
            )
        depth = depth_or_prediction
        offsets = offsets_or_ray_box_entry
        ray_box_entry = ray_box_entry_or_exit

    if not torch.is_tensor(depth) or not torch.is_tensor(offsets):
        raise TypeError("depth and offsets must be torch Tensors.")
    if depth.ndim != 4 or depth.shape[1] != 1:
        raise ValueError(f"depth must have shape [B,1,h,w], got {tuple(depth.shape)}.")
    if offsets.ndim != 4 or offsets.shape[1] != 3:
        raise ValueError(
            f"offsets must have shape [B,3,h,w], got {tuple(offsets.shape)}."
        )
    if offsets.shape[0] != depth.shape[0] or offsets.shape[-2:] != depth.shape[-2:]:
        raise ValueError(
            "depth and offsets must have matching batch and spatial shapes."
        )

    expected_ray_shape = (
        depth.shape[0],
        depth.shape[2],
        depth.shape[3],
        3,
    )
    if tuple(ray_box_entry.shape) != expected_ray_shape:
        raise ValueError(
            f"ray_box_entry must have shape {expected_ray_shape}, got "
            f"{tuple(ray_box_entry.shape)}."
        )
    if tuple(ray_box_exit.shape) != expected_ray_shape:
        raise ValueError(
            f"ray_box_exit must have shape {expected_ray_shape}, got "
            f"{tuple(ray_box_exit.shape)}."
        )
    if ray_box_entry.device != depth.device or ray_box_exit.device != depth.device:
        raise ValueError("Predictions and ray-box tensors must be on the same device.")
    if ray_box_entry.dtype != depth.dtype or ray_box_exit.dtype != depth.dtype:
        raise ValueError("Predictions and ray-box tensors must have the same dtype.")

    depth_last = depth.permute(0, 2, 3, 1)
    offsets_last = offsets.permute(0, 2, 3, 1)
    centers = ray_box_entry + depth_last * (ray_box_exit - ray_box_entry)
    centers = centers + offsets_last
    if clamp:
        centers = centers.clamp(0.0, 1.0)
    return centers.reshape(depth.shape[0], -1, 3)


def make_gcp_checkpoint(
    model: MonocularGaussianCenterPredictor, **metadata: Any,
) -> Dict[str, Any]:
    """Build the portable checkpoint mapping consumed by the loader."""

    reserved = {"model_config", "model_state_dict"}.intersection(metadata)
    if reserved:
        raise ValueError(f"Checkpoint metadata uses reserved keys: {sorted(reserved)}.")
    return {
        "model_config": model.config.to_dict(),
        "model_state_dict": model.state_dict(),
        **metadata,
    }


def load_gcp_checkpoint(
    path: Union[str, Path], device: Union[str, torch.device] = "cpu",
) -> MonocularGaussianCenterPredictor:
    """Load a predictor checkpoint onto ``device`` and put it in eval mode.

    Checkpoints deliberately contain a plain configuration mapping rather than
    a pickled model instance.  This keeps the format portable across training
    and Stage-2 inference environments.
    """

    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"GCP checkpoint does not exist: {checkpoint_path}")
    map_location = torch.device(device)
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=map_location, weights_only=True,
        )
    except TypeError:  # PyTorch versions predating the weights_only argument.
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("GCP checkpoint must contain a mapping.")
    missing = {"model_config", "model_state_dict"}.difference(checkpoint)
    if missing:
        raise KeyError(f"GCP checkpoint is missing keys: {sorted(missing)}.")

    raw_config = checkpoint["model_config"]
    config = (
        raw_config
        if isinstance(raw_config, GCPModelConfig)
        else GCPModelConfig.from_dict(raw_config)
    )
    state_dict = checkpoint["model_state_dict"]
    if not isinstance(state_dict, Mapping):
        raise TypeError("model_state_dict must be a mapping.")

    model = MonocularGaussianCenterPredictor(config)
    model.load_state_dict(state_dict, strict=True)
    model.to(map_location)
    model.eval()
    model.checkpoint_metadata = {
        key: value
        for key, value in checkpoint.items()
        if key not in {"model_config", "model_state_dict"}
    }
    return model


__all__ = [
    "GCPModelConfig",
    "MonocularGaussianCenterPredictor",
    "lift_depth_offsets_to_centers",
    "load_gcp_checkpoint",
    "make_gcp_checkpoint",
]
