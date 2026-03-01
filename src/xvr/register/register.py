import torch
from jaxtyping import Float
from nanodrr.camera import make_rt_inv

from ..io.xray import Intrinsics
from ..model.inference import predict_pose
from ..model.modules.network import load_model
from .base import RegisterBase


class RegisterFixed(RegisterBase):
    """Register using a manually specified initial pose.

    Useful for cases where an approximate pose is already known, such as
    when initializing from a prior scan or a clinical estimate.
    """

    def get_initial_pose_estimate(
        self,
        _img,
        _intrinsics,
        rot: tuple[float, float, float],
        xyz: tuple[float, float, float],
        orientation: str = "AP",
        isocenter: bool = True,
    ) -> Float[torch.Tensor, "1 4 4"]:
        """Compute the initial pose from user-provided rotation and translation.

        Args:
            _gt: Unused. Present for interface compatibility.
            _intrinsics: Unused. Present for interface compatibility.
            rot: Rotation angles (in degrees) as (rx, ry, rz).
            xyz: Translation (in mm) as (x, y, z).
            orientation: Starting orientation of the volume, e.g. "AP" or "lateral".
            isocenter: If True, centers the pose at the subject isocenter.

        Returns:
            A 4x4 rigid transformation matrix.
        """
        return make_rt_inv(
            torch.tensor([rot], dtype=torch.float32, device=self.device),
            torch.tensor([xyz], dtype=torch.float32, device=self.device),
            orientation,
            self.subject.isocenter if isocenter else None,
        )


class RegisterModel(RegisterBase):
    """Register using a neural network to predict the initial pose.

    The network is run once on the preprocessed X-ray to produce an initial
    pose estimate, which is then refined by gradient-based optimization.

    Args:
        ckpt: Path to the model checkpoint.
        **kwargs: Passed to Register.
    """

    def __init__(self, ckpt: str, **kwargs):
        super().__init__(**kwargs)
        self.model, self.config = load_model(ckpt)
        self.model = self.model.to(self.device)

    def get_initial_pose_estimate(
        self,
        img: Float[torch.Tensor, "1 1 H W"],
        intrinsics: Intrinsics,
    ) -> Float[torch.Tensor, "1 4 4"]:
        """Predict the initial pose from the X-ray using a neural network.

        Args:
            gt: Preprocessed ground truth X-ray image.
            intrinsics: Camera intrinsics for the X-ray.
            **kwargs: Unused. Present for interface compatibility.

        Returns:
            A 4x4 rigid transformation matrix.
        """
        return predict_pose(
            self.model,
            img,
            intrinsics,
            self.config["sdd"],
            self.config["delx"],
            self.config["height"],
        )
