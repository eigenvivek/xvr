"""Regression tests pinning the registration trajectory itself.

`test_registration.py` asserts the optimizer reaches the right pose. It cannot assert that
it takes the same route: a 17% change in the loss trajectory still lands inside its
tolerance. These tests close that gap by comparing the full trace against stored values.
"""

import pytest

from xvr.register import FixedPose, Register

# Three stages, so the per-stage step-size decay is exercised past the point where
# `math.prod(2**i for i in range(stage + 1))` and a naive `2**stage` still agree.
SCENARIOS = {
    "gmncc_three_stage": dict(metric="gmncc", scales=[4.0, 2.0, 1.0], n_itrs=[15, 15, 15]),
    "mncc": dict(metric="mncc", scales=[2.0, 1.0], n_itrs=[20, 20]),
    "gncc": dict(metric="gncc", scales=[2.0, 1.0], n_itrs=[20, 20]),
}


def trace(result):
    """The parts of a run that are reproducible; `times` is wall-clock and is excluded."""
    log = result.log
    return {
        "n_iterations": len(log.losses),
        "losses": log.losses,
        "scales": log.scales,
        "rescale_factors": log.rescale_factors,
        "rots": log.rots.tolist(),
        "xyzs": log.xyzs.tolist(),
        "final_pose": result.final_pose.matrix[0].tolist(),
    }


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_the_optimization_trajectory_is_unchanged(
    name, phantom_ct, xray_dicom, offset_pose, golden
):
    """The whole trace is pinned, not just the endpoint."""
    options = SCENARIOS[name]
    register = Register(
        initializer=FixedPose(**offset_pose, orientation="AP", device="cpu"),
        imagepath=str(phantom_ct),
        device="cpu",
        patience=[10**9] * len(options["scales"]),
        max_n_plateaus=10**9,
        **options,
    )
    golden(name, trace(register(str(xray_dicom), linearize=False)))
