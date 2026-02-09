import torch
from jaxtyping import Float
from nanodrr.camera import make_rt_inv

from nanodrr.data import Subject


def get_random_pose(
    subject: Subject,
    alphamin: float,
    alphamax: float,
    betamin: float,
    betamax: float,
    gammamin: float,
    gammamax: float,
    txmin: float,
    txmax: float,
    tymin: float,
    tymax: float,
    tzmin: float,
    tzmax: float,
    batch_size: int,
    orientation: str,
) -> Float[torch.Tensor, "batch_size 4 4"]:
    """Generate a batch of random poses in SE(3) using specified ranges."""
    dtype = subject.isocenter.dtype
    device = subject.isocenter.device

    alpha = uniform(alphamin, alphamax, batch_size, dtype=dtype, device=device)
    beta = uniform(betamin, betamax, batch_size, dtype=dtype, device=device)
    gamma = uniform(gammamin, gammamax, batch_size, dtype=dtype, device=device)
    tx = uniform(
        txmin, txmax, batch_size, circle_shift=False, dtype=dtype, device=device
    )
    ty = uniform(
        tymin, tymax, batch_size, circle_shift=False, dtype=dtype, device=device
    )
    tz = uniform(
        tzmin, tzmax, batch_size, circle_shift=False, dtype=dtype, device=device
    )
    rot = torch.concat([alpha, beta, gamma], dim=1)
    xyz = torch.concat([tx, ty, tz], dim=1)
    return make_rt_inv(rot, xyz, orientation, subject.isocenter)


def uniform(
    low: float,
    high: float,
    n: int,
    circle_shift: bool = True,
    dtype: torch.dtype = torch.float32,
    device: torch.device = "cuda",
) -> Float[torch.Tensor, "n 1"]:
    x = (high - low) * torch.rand(n, 1, dtype=dtype, device=device) + low
    if circle_shift:
        x = ((x + 180) % 360) - 180
    return x
