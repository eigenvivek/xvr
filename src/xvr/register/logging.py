from dataclasses import dataclass
from pathlib import Path

import torch
from diffdrr.drr import DRR
from diffdrr.pose import RigidTransform
from jaxtyping import Float

from .pose import Pose


@dataclass
class OptimizationLogger:
    """Log the intermediates of a multiscale optimization run."""

    losses: list[float]
    scales: list[float]
    rescale_factors: list[float]
    rots: Float[torch.Tensor, "N 3"]
    xyzs: Float[torch.Tensor, "N 3"]


class RegistrationResult:
    """The result of a registration run.

    Args:
        drr: The DRR renderer with detector geometry.
        pose: The optimized pose model.
        init_pose: The initial pose estimate.
        gt: The preprocessed ground truth X-ray.
        log: The optimization log.
    """

    def __init__(
        self,
        drr: DRR,
        pose: Pose,
        init_pose: RigidTransform,
        gt: Float[torch.Tensor, "1 1 H W"],
        log: OptimizationLogger | None = None,
    ) -> None:
        self.drr = drr
        self.init_pose = init_pose
        self.final_pose = pose()
        self.gt = gt
        self.log = log

    def save(self, path: str | Path) -> None:
        """Save the registration result to a checkpoint file.

        Args:
            path: Output file path.
        """
        if log is not None:
            log = {
                "losses": self.log.losses,
                "scales": self.log.scales,
                "rescale_factors": self.log.rescale_factors,
                "rots": self.log.rots.cpu(),
                "xyzs": self.log.xyzs.cpu(),
            }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "detector": {
                    "sdd": self.drr.detector.sdd,
                    "height": self.drr.detector.height,
                    "width": self.drr.detector.width,
                    "delx": self.drr.detector.delx,
                    "dely": self.drr.detector.dely,
                    "reverse_x_axis": self.drr.detector.reverse_x_axis,
                },
                "init_pose": self.init_pose.matrix.cpu(),
                "final_pose": self.final_pose.matrix.cpu(),
                "gt": self.gt.cpu(),
                "log": log,
            },
            path,
        )
