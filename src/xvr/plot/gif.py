from pathlib import Path

import nanodrr
import torch
import torch.nn.functional as F
from jaxtyping import Float

from ..register.logging import RegistrationResult


def animate(
    result: RegistrationResult,
    ticks: bool = False,
    savepath: str | Path | None = None,
    **kwargs,
) -> Path | None:
    """Animate the result of a registration call.

    Args:
        result: Output of a registration result
        ticks: Whether to show pixel coordinate ticks
        savepath: Output file path, or `None` for inline display
        **kwargs: Additional arguments forwarded to `nanodrr.plot.animate`

    Returns:
        Path to saved file if `savepath` is provided, otherwise `None`
    """
    # Render the iterates from each iteration
    moving = replay(result)

    # Resize the fixed image to match the moving images
    *_, H, W = moving.shape
    fixed = F.interpolate(result.gt.cpu(), (H, W))

    # Return the GIF
    return nanodrr.plot.animate(moving, out=savepath, fixed_img=fixed, ticks=ticks, **kwargs)


@torch.no_grad()
def replay(result: RegistrationResult) -> Float[torch.Tensor, "B 1 H W"]:
    """Rerender frames from the optimization run at their native resolutions."""
    device = result.reg.rt_inv.device
    current_scale = None
    initial_height = result.reg.height

    # Render the iterates
    preds = []
    for rot, xyz, rescale_factor, scale in zip(
        result.log.rots, result.log.xyzs, result.log.rescale_factors, result.log.scales
    ):
        if scale != current_scale:
            result.reg.rescale_(rescale_factor)
            current_scale = scale
        result.reg._rot.copy_(rot.unsqueeze(0).to(device))
        result.reg._xyz.copy_(xyz.unsqueeze(0).to(device))
        preds.append(result.reg().cpu())

    # Rescale all iterates to the size of the largest image
    *_, H, W = tuple(max([img.shape for img in preds]))
    preds = torch.cat([F.interpolate(img, (H, W), mode="nearest") for img in preds])

    # Reset the detector
    reset_factor = initial_height / result.reg.height
    result.reg.rescale_(reset_factor)

    return preds
