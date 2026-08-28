import torch

from xvr.model.sampler import get_random_pose, uniform

RANGES = dict(
    alphamin=-30.0,
    alphamax=30.0,
    betamin=-20.0,
    betamax=20.0,
    gammamin=-10.0,
    gammamax=10.0,
    txmin=-50.0,
    txmax=50.0,
    tymin=750.0,
    tymax=850.0,
    tzmin=-25.0,
    tzmax=25.0,
)


def test_uniform_stays_within_its_bounds():
    torch.manual_seed(0)
    x = uniform(-3.0, 7.0, 512)
    assert x.shape == (512, 1)
    assert x.min() >= -3.0 and x.max() <= 7.0


def test_circle_shift_wraps_angles_into_the_half_open_turn():
    """Angles are wrapped to [-180, 180) so that e.g. 350 degrees reads as -10."""
    torch.manual_seed(0)
    x = uniform(180.0, 540.0, 512, circle_shift=True)
    assert x.min() >= -180.0 and x.max() < 180.0


def test_random_poses_are_rigid():
    """Every sampled pose must be a genuine element of SE(3), not merely a 4x4 matrix."""
    torch.manual_seed(0)
    matrix = get_random_pose(**RANGES, batch_size=16).matrix
    assert matrix.shape == (16, 4, 4)

    rotation = matrix[:, :3, :3]
    identity = torch.eye(3).expand_as(rotation)
    torch.testing.assert_close(rotation @ rotation.mT, identity, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(torch.linalg.det(rotation), torch.ones(16), atol=1e-5, rtol=1e-5)


def test_sampled_parameters_respect_the_requested_ranges():
    """The ranges bound the *pose parameters*, not the matrix translation column.

    `convert` composes the rotation into the matrix, so `matrix[:, :3, 3]` is not what
    was sampled; decoding back to euler_angles/ZXY is what recovers it.
    """
    torch.manual_seed(0)
    rot, xyz = get_random_pose(**RANGES, batch_size=256).convert("euler_angles", "ZXY")
    degrees = torch.rad2deg(rot)

    lower_xyz = torch.tensor([RANGES["txmin"], RANGES["tymin"], RANGES["tzmin"]])
    upper_xyz = torch.tensor([RANGES["txmax"], RANGES["tymax"], RANGES["tzmax"]])
    assert (xyz >= lower_xyz - 1e-3).all() and (xyz <= upper_xyz + 1e-3).all()

    lower_rot = torch.tensor([RANGES["alphamin"], RANGES["betamin"], RANGES["gammamin"]])
    upper_rot = torch.tensor([RANGES["alphamax"], RANGES["betamax"], RANGES["gammamax"]])
    assert (degrees >= lower_rot - 1e-3).all() and (degrees <= upper_rot + 1e-3).all()


def test_sampling_is_reproducible_under_a_seed():
    """Training runs are only repeatable if the pose stream is."""
    torch.manual_seed(7)
    first = get_random_pose(**RANGES, batch_size=4).matrix
    torch.manual_seed(7)
    torch.testing.assert_close(first, get_random_pose(**RANGES, batch_size=4).matrix)
