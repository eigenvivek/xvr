from dataclasses import dataclass
from pathlib import Path

import torch
from jaxtyping import Float
from nanodrr.registration import Registration

from .subject import load_subject


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
                    "subject": {
                        "imagepath": self.reg._imagepath,
                        "labelpath": self.reg._labelpath,
                        "labels": self.reg._labels,
                    },
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
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> "RegistrationResult":
        """Load a registration result from a checkpoint file.

        Args:
            path: Path to the checkpoint file.
            device: Device to load tensors onto.

        Returns:
            The restored registration result.
        """
        state = torch.load(path, map_location=device)
        subject = state["subject"]
        subject = load_subject(subject["imagepath"], subject["labelpath"], subject["labels"]).to(
            device
        )
        reg = Registration(
            subject=subject,
            rt_inv=state["rt_inv"],
            k_inv=state["k_inv"],
            sdd=state["sdd"],
            height=state["height"],
            width=state["width"],
        )
        with torch.no_grad():
            reg._rot.copy_(state["rot"])
            reg._xyz.copy_(state["xyz"])
        log = OptimizationLogger(
            losses=state["losses"],
            scales=state["scales"],
            rescale_factors=state["rescale_factors"],
            rots=state["rots"],
            xyzs=state["xyzs"],
        )
        return cls(reg=reg, gt=state["gt"], log=log)
