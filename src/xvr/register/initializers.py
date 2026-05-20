from typing import Protocol

import torch
from attrs import define, field
from diffdrr.pose import RigidTransform, convert

from ..io import parse_dicom_pose
from ..model.inference import predict_pose
from ..model.network import load_model
from .context import XrayContext


class PoseInitializer(Protocol):
    """Strategy for computing the initial pose estimate before optimization.

    Concrete implementations expose two pieces of metadata read by
    `Register` during setup, plus a callable that produces a `RigidTransform`
    given the current X-ray context.

    Attributes:
        orientation: Patient orientation for the DRR.
        reverse_x_axis: Horizontally flip the rendered DRRs.
    """

    orientation: str | None
    reverse_x_axis: bool

    def __call__(self, ctx: XrayContext) -> RigidTransform:
        """Compute an initial pose estimate from the X-ray context."""
        ...


@define(slots=False)
class FixedPose:
    """Initial pose specified manually as Euler angles and a translation.

    Useful when an approximate pose is already known, e.g., from a prior
    scan, a clinical estimate, or a hand-tuned guess.

    Args:
        rot: Rotation angles in degrees as (rx, ry, rz), ZXY convention.
        xyz: Translation in mm as (x, y, z).
        eps: Small value added to inputs to avoid degenerate gradients.
    """

    rot: tuple[float, float, float]
    xyz: tuple[float, float, float]
    orientation: str | None = "AP"
    reverse_x_axis: bool = False
    device: str = "cuda"
    eps: float = 1e-8

    def __call__(self, ctx: XrayContext) -> RigidTransform:
        return convert(
            torch.tensor([self.rot], dtype=torch.float32, device=self.device) + self.eps,
            torch.tensor([self.xyz], dtype=torch.float32, device=self.device) + self.eps,
            parameterization="euler_angles",
            convention="ZXY",
            degrees=True,
        )


@define(slots=False)
class ModelPose:
    """Initial pose predicted by a neural network from the X-ray image.

    The checkpoint dictates `orientation` and `reverse_x_axis`; users do not
    set these directly.

    Args:
        ckpt: Path to the model checkpoint.
    """

    ckpt: str
    device: str = "cuda"
    model: torch.nn.Module = field(init=False, repr=False, default=None)
    config: dict = field(init=False, repr=False, factory=dict)
    orientation: str | None = field(init=False, default="AP")
    reverse_x_axis: bool = field(init=False, default=False)

    def __attrs_post_init__(self):
        model, self.config = load_model(self.ckpt)
        self.model = model.to(self.device)
        self.orientation = self.config["orientation"]
        self.reverse_x_axis = self.config["reverse_x_axis"]

    def __call__(self, ctx: XrayContext) -> RigidTransform:
        return predict_pose(self.model, self.config, ctx.img, **ctx.intrinsics)


@define(slots=False)
class DicomPose:
    """Initial pose parsed from pose parameters in the DICOM metadata."""

    orientation: str | None = "AP"
    reverse_x_axis: bool = False
    device: str = "cuda"

    def __call__(self, ctx: XrayContext) -> RigidTransform:
        return parse_dicom_pose(ctx.filename, self.orientation, self.device)
