import attrs
import torch
from diffdrr.pose import RigidTransform, convert
from jaxtyping import Float

from ..io import Intrinsics
from ..model.inference import predict_pose
from ..model.network import load_model
from .base import RegisterBase
from .logging import RegistrationResult


@attrs.define(slots=False)
class RegisterFixed(RegisterBase):
    """Register using a manually specified initial pose.

    Useful for cases where an approximate pose is already known, such as
    when initializing from a prior scan or a clinical estimate.
    """

    def __call__(
        self,
        filename: str,
        rot: tuple[float, float, float],
        xyz: tuple[float, float, float],
        **kwargs,
    ) -> RegistrationResult:
        """Run registration with a manually specified initial pose.

        Args:
            filename: Path to the X-ray image.
            rot: Rotation angles (in degrees) as (rx, ry, rz).
            xyz: Translation (in mm) as (x, y, z).
            **kwargs: See RegisterBase.__call__ for remaining arguments.
        """
        return super().__call__(filename, rot=rot, xyz=xyz, **kwargs)

    def get_initial_pose_estimate(
        self,
        _img,
        _intrinsics,
        rot: tuple[float, float, float],
        xyz: tuple[float, float, float],
    ) -> RigidTransform:
        """Compute the initial pose from user-provided rotation and translation.

        Args:
            _img: Unused. Present for interface compatibility.
            _intrinsics: Unused. Present for interface compatibility.
            rot: Rotation angles (in degrees) as (rx, ry, rz).
            xyz: Translation (in mm) as (x, y, z).

        Returns:
            A 4x4 rigid transformation matrix.
        """
        return convert(
            torch.tensor([rot], dtype=torch.float32, device=self.device),
            torch.tensor([xyz], dtype=torch.float32, device=self.device),
            parameterization="euler_angles",
            convention="ZXY",
            degrees=True,
        )


@attrs.define(slots=False)
class RegisterModel(RegisterBase):
    """Register using a neural network to predict the initial pose.

    The network is run once on the preprocessed X-ray to produce an initial
    pose estimate, which is then refined by gradient-based optimization.

    Args:
        ckpt: Path to the model checkpoint.
    """

    ckpt: str
    model: torch.nn.Module = attrs.field(init=False, repr=False)
    config: dict = attrs.field(init=False, repr=False)

    def __attrs_post_init__(self):
        model, self.config = load_model(self.ckpt)
        self.model = model.to(self.device)
        self.orientation = self.config["orientation"]
        super().__attrs_post_init__()

    def get_initial_pose_estimate(
        self,
        img: Float[torch.Tensor, "1 1 H W"],
        intrinsics: Intrinsics,
    ) -> RigidTransform:
        """Predict the initial pose from the X-ray using a neural network.

        Args:
            img: Preprocessed ground truth X-ray image.
            intrinsics: Camera intrinsics for the X-ray.

        Returns:
            A 4x4 rigid transformation matrix.
        """
        return predict_pose(self.model, self.config, img, **intrinsics)
