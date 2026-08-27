import torch
from diffdrr.pose import RigidTransform

from xvr.utils.transforms import _get_nearest_rigid_transform


def make_rotation(angle: float = 0.4) -> torch.Tensor:
    """A rotation about the z-axis, as a 4x4 homogeneous matrix."""
    c, s = torch.cos(torch.tensor(angle)), torch.sin(torch.tensor(angle))
    matrix = torch.eye(4)
    matrix[:3, :3] = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return matrix[None]


def test_a_rotation_is_returned_unchanged():
    """Projecting something already in SO(3) must be a no-op."""
    matrix = make_rotation()
    projected = _get_nearest_rigid_transform(RigidTransform(matrix)).matrix
    torch.testing.assert_close(projected[0, :3, :3], matrix[0, :3, :3], atol=1e-5, rtol=1e-5)


def test_a_scaled_matrix_projects_onto_the_rotation_group():
    """ANTs transforms carry scale/shear; only the rotation survives the SVD projection."""
    matrix = make_rotation()
    matrix[0, :3, :3] *= 1.7  # uniform scale, definitely not in SO(3)
    rotation = _get_nearest_rigid_transform(RigidTransform(matrix)).matrix[0, :3, :3]

    torch.testing.assert_close(rotation @ rotation.mT, torch.eye(3), atol=1e-5, rtol=1e-5)
    assert torch.linalg.det(rotation).item() > 0


def test_the_result_always_has_unit_determinant():
    """A reflection has det = -1; the projection must return a rotation, not a reflection."""
    matrix = torch.eye(4)[None].clone()
    matrix[0, 0, 0] = -1.0  # flip one axis
    matrix[0, :3, 3] = torch.tensor([3.0, -2.0, 1.0])
    rotation = _get_nearest_rigid_transform(RigidTransform(matrix)).matrix[0, :3, :3]
    assert torch.linalg.det(rotation).item() > 0.99


def test_a_shear_is_removed_but_the_pose_stays_finite():
    torch.manual_seed(0)
    matrix = torch.eye(4)[None].clone()
    matrix[0, :3, :3] += 0.05 * torch.randn(3, 3)
    matrix[0, :3, 3] = torch.tensor([10.0, -5.0, 2.0])
    result = _get_nearest_rigid_transform(RigidTransform(matrix)).matrix
    assert torch.isfinite(result).all()
    torch.testing.assert_close(
        result[0, :3, :3] @ result[0, :3, :3].mT, torch.eye(3), atol=1e-5, rtol=1e-5
    )
