import torch
from nanodrr.metrics import (
    GradientNormalizedCrossCorrelation2d,
    MultiscaleNormalizedCrossCorrelation2d,
)

from .losses import GradientMultiscaleNormalizedCrossCorrelation2d

METRICS: dict[str, type[torch.nn.Module]] = {
    "mncc": MultiscaleNormalizedCrossCorrelation2d,
    "gncc": GradientNormalizedCrossCorrelation2d,
    "gmncc": GradientMultiscaleNormalizedCrossCorrelation2d,
}


def load_loss_function(loss: str | torch.nn.Module, **kwargs) -> torch.nn.Module:
    """Initialize a loss function for 2D/3D registration.

    Args:
        loss: Either the name of a loss function in `METRICS` or a callable
            Module whose forward method implements the loss.
        **kwargs: Optional arguments for the constructor of the loss function.
    """
    if isinstance(loss, torch.nn.Module):
        return loss
    try:
        return METRICS[loss](**kwargs)
    except KeyError:
        raise ValueError(f"Unknown metric '{loss}'. Available: {list(METRICS.keys())}")
