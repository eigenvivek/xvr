import torch
import torch.nn.functional as F

from xvr.model.inference import _construct_antipode, _resample_xray
from xvr.model.trainer import _make_voxel_to_grid, make_translation


def make_pose(rot=(0.3, -0.2, 0.1), xyz=(5.0, 80.0, -3.0)):
    from diffdrr.pose import convert

    return convert(
        torch.tensor([list(rot)]),
        torch.tensor([list(xyz)]),
        parameterization="euler_angles",
        convention="ZXY",
    )


# --- make_translation -----------------------------------------------------------------


def test_make_translation_has_no_rotation():
    matrix = make_translation(torch.tensor([[1.0, 2.0, 3.0]])).matrix
    torch.testing.assert_close(matrix[0, :3, :3], torch.eye(3), atol=1e-6, rtol=1e-6)


def test_make_translation_places_the_offset_in_the_last_column():
    matrix = make_translation(torch.tensor([[1.0, 2.0, 3.0]])).matrix
    torch.testing.assert_close(
        matrix[0, :3, 3], torch.tensor([1.0, 2.0, 3.0]), atol=1e-5, rtol=1e-5
    )


# --- _make_voxel_to_grid --------------------------------------------------------------


def test_voxel_to_grid_maps_the_first_and_last_voxel_centers():
    """align_corners=False puts voxel centers at +/-(1 - 1/S), not at +/-1."""
    W = 8
    affine = _make_voxel_to_grid(torch.Size([1, 1, 4, 6, W]), torch.device("cpu"), torch.float32)
    matrix = affine.matrix[0]
    first = matrix[0, 0] * 0 + matrix[0, 3]
    last = matrix[0, 0] * (W - 1) + matrix[0, 3]
    assert first.item() == -(1 - 1 / W)
    assert last.item() == (1 - 1 / W)


def test_voxel_to_grid_agrees_with_grid_sample():
    """The affine exists so a voxel index can be handed straight to `grid_sample`."""
    D, H, W = 3, 4, 5
    volume = torch.arange(D * H * W, dtype=torch.float32).reshape(1, 1, D, H, W)
    affine = _make_voxel_to_grid(volume.shape, torch.device("cpu"), torch.float32).matrix[0]

    index = torch.tensor([2.0, 1.0, 2.0, 1.0])  # (w, h, d, 1) homogeneous voxel index
    normalized = (affine @ index)[:3]
    sampled = F.grid_sample(
        volume, normalized.reshape(1, 1, 1, 1, 3), align_corners=False, mode="nearest"
    )
    assert sampled.item() == volume[0, 0, 2, 1, 2].item()


def test_voxel_to_grid_respects_the_requested_dtype():
    affine = _make_voxel_to_grid(torch.Size([1, 1, 2, 2, 2]), torch.device("cpu"), torch.float64)
    assert affine.matrix.dtype == torch.float64


# --- _construct_antipode --------------------------------------------------------------


def test_the_antipode_is_a_different_pose():
    """`--antipodal` exists because a regressor can land on the opposite side of the arc."""
    pose = make_pose()
    assert not torch.allclose(_construct_antipode(pose).matrix, pose.matrix, atol=1e-4)


def test_applying_the_antipode_twice_returns_the_original():
    """Negating two angles and adding pi twice is the identity, up to the 2*pi wrap."""
    pose = make_pose()
    twice = _construct_antipode(_construct_antipode(pose))
    torch.testing.assert_close(twice.matrix, pose.matrix, atol=1e-4, rtol=1e-4)


def test_the_antipode_leaves_the_translation_untouched():
    """Only the rotation is reflected; the source stays where it was."""
    pose = make_pose()
    _, xyz = pose.convert("euler_angles", "ZXY")
    _, antipodal_xyz = _construct_antipode(pose).convert("euler_angles", "ZXY")
    torch.testing.assert_close(antipodal_xyz, xyz, atol=1e-4, rtol=1e-4)


# --- _resample_xray -------------------------------------------------------------------


def test_resample_requires_square_pixels():
    """Non-square detectors are explicitly unsupported, and must fail loudly."""
    config = {"height": 64, "delx": 1.0, "sdd": 1000.0}
    try:
        _resample_xray(torch.rand(1, 1, 128, 128), 1000.0, 0.3, 0.4, 0.0, 0.0, config)
    except AssertionError as error:
        assert "square" in str(error)
    else:
        raise AssertionError("expected an AssertionError for non-square pixels")


def test_resample_reports_the_original_image_size():
    """The caller crops to `min(height, width)` of the *input*, so these must be pre-resample."""
    config = {"height": 32, "delx": 1.0, "sdd": 1000.0}
    _, height, width = _resample_xray(torch.rand(1, 1, 96, 128), 1000.0, 0.5, 0.5, 0.0, 0.0, config)
    assert (height, width) == (96, 128)
