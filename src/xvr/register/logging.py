from dataclasses import dataclass
from pathlib import Path

import torch
from jaxtyping import Float
from nanodrr.data import Subject
from nanodrr.registration import Registration


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
        reg: The registration module with optimized pose.
        gt: The preprocessed ground truth X-ray.
        log: The optimization log.
    """

    def __init__(
        self,
        reg: Registration,
        gt: Float[torch.Tensor, "1 1 H W"],
        log: OptimizationLogger,
    ) -> None:
        self.reg = reg
        self.gt = gt
        self.log = log

    def save(self, path: str | Path) -> None:
        """Save the registration result to a checkpoint file.

        Args:
            path: Output file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "reg": {
                    "rot": self.reg._rot.detach().cpu(),
                    "xyz": self.reg._xyz.detach().cpu(),
                    "rt_inv": self.reg.rt_inv.cpu(),
                    "k_inv": self.reg.k_inv.cpu(),
                    "sdd": self.reg.sdd.cpu(),
                    "height": self.reg.height,
                    "width": self.reg.width,
                },
                "gt": self.gt.cpu(),
                "log": {
                    "losses": self.log.losses,
                    "scales": self.log.scales,
                    "rescale_factors": self.log.rescale_factors,
                    "rots": self.log.rots.cpu(),
                    "xyzs": self.log.xyzs.cpu(),
                },
            },
            path,
        )

    @classmethod
    def load(
        cls, path: str | Path, subject: Subject, device: str | torch.device = "cpu"
    ) -> "RegistrationResult":
        state = torch.load(path, map_location=device)
        reg = Registration(
            subject=subject,
            rt_inv=state["reg"]["rt_inv"],
            k_inv=state["reg"]["k_inv"],
            sdd=state["reg"]["sdd"],
            height=state["reg"]["height"],
            width=state["reg"]["width"],
        )
        with torch.no_grad():
            reg._rot.copy_(state["reg"]["rot"])
            reg._xyz.copy_(state["reg"]["xyz"])
        log = OptimizationLogger(
            losses=state["log"]["losses"],
            scales=state["log"]["scales"],
            rescale_factors=state["log"]["rescale_factors"],
            rots=state["log"]["rots"],
            xyzs=state["log"]["xyzs"],
        )
        return cls(reg=reg, gt=state["gt"], log=log)
