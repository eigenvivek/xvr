import pytest
import torch
from diffdrr.pose import convert

from xvr.register.pose import Pose

PARAMETERIZATIONS = [
    ("euler_angles", "ZXY"),
    ("se3_log_map", None),
    ("quaternion", None),
    ("rotation_6d", None),
    ("axis_angle", None),
]


def make_init_pose() -> "convert":
    """A generic off-axis pose, well away from the SO(3) singularity at zero."""
    return convert(
        torch.tensor([[0.3, -0.2, 0.1]]),
        torch.tensor([[5.0, 80.0, -3.0]]),
        parameterization="euler_angles",
        convention="ZXY",
    )


@pytest.mark.parametrize("parameterization, convention", PARAMETERIZATIONS)
def test_pose_roundtrips_through_every_parameterization(parameterization, convention):
    """Decoding a pose into parameters and re-encoding must return the same matrix."""
    init = make_init_pose()
    pose = Pose(init, parameterization, convention)
    torch.testing.assert_close(pose().matrix, init.matrix, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("parameterization, convention", PARAMETERIZATIONS)
def test_pose_parameters_are_optimizable_leaves(parameterization, convention):
    """`_rot` and `_xyz` are what the Adam param groups in `_setup_stage` bind to."""
    pose = Pose(make_init_pose(), parameterization, convention)
    for param in (pose._rot, pose._xyz):
        assert param.requires_grad
        assert param.is_leaf


@pytest.mark.parametrize("parameterization, convention", PARAMETERIZATIONS)
def test_gradients_flow_from_the_pose_matrix_to_the_parameters(parameterization, convention):
    """Registration optimizes the pose through the rendered image, so the graph must connect."""
    pose = Pose(make_init_pose(), parameterization, convention)
    pose().matrix.sum().backward()
    assert pose._rot.grad is not None and torch.isfinite(pose._rot.grad).all()
    assert pose._xyz.grad is not None and torch.isfinite(pose._xyz.grad).all()


def test_quaternion_rotation_has_four_components():
    """`OptimizationLogger.rots` is annotated "N 3", which only holds for 3-component rotations."""
    pose = Pose(make_init_pose(), "quaternion", None)
    assert pose._rot.shape == (1, 4)
