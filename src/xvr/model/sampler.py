import torch
from diffdrr.pose import convert


def get_random_pose(
    alphamin,
    alphamax,
    betamin,
    betamax,
    gammamin,
    gammamax,
    txmin,
    txmax,
    tymin,
    tymax,
    tzmin,
    tzmax,
    batch_size,
):
    """Generate a batch of random poses in SE(3) using specified ranges."""
    alpha = uniform(alphamin, alphamax, batch_size, circle_shift=True)
    beta = uniform(betamin, betamax, batch_size, circle_shift=True)
    gamma = uniform(gammamin, gammamax, batch_size, circle_shift=True)
    tx = uniform(txmin, txmax, batch_size)
    ty = uniform(tymin, tymax, batch_size)
    tz = uniform(tzmin, tzmax, batch_size)
    rot = torch.concat([alpha, beta, gamma], dim=1)
    xyz = torch.concat([tx, ty, tz], dim=1)
    return convert(
        rot, xyz, parameterization="euler_angles", convention="ZXY", degrees=True
    )


def uniform(low, high, n, circle_shift=False):
    x = (high - low) * torch.rand(n, 1) + low
    if circle_shift:
        x = ((x + 180) % 360) - 180
    return x
