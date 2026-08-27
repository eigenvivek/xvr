import numpy as np
import pytest
import torch

from xvr.io.xray import (
    _mask_outside_circle,
    _parse_dicom_intrinsics,
    _preprocess_xray,
    _reduce_frames,
    parse_dicom_pose,
    read_xray,
)


def make_frames(n: int = 4, size: int = 8) -> torch.Tensor:
    """A 5D multiframe stack, the shape a cine DICOM produces."""
    torch.manual_seed(0)
    return torch.rand(1, 1, n, size, size)


def make_dataset(**tags):
    """A bare pydicom Dataset carrying only the tags under test."""
    from pydicom.dataset import Dataset

    ds = Dataset()
    ds.DistanceSourceToDetector = 1000.0
    for tag, value in tags.items():
        setattr(ds, tag, value)
    return ds


# --- _reduce_frames -------------------------------------------------------------------


def test_reduce_frames_none_leaves_the_stack_intact():
    frames = make_frames()
    assert _reduce_frames(frames, None) is frames


def test_reduce_frames_max_takes_a_maximum_intensity_projection():
    frames = make_frames()
    torch.testing.assert_close(_reduce_frames(frames, "max"), frames.max(dim=2).values)


def test_reduce_frames_sum_projects_over_time():
    frames = make_frames()
    torch.testing.assert_close(_reduce_frames(frames, "sum"), frames.sum(dim=2))


def test_reduce_frames_int_selects_a_single_frame():
    frames = make_frames()
    torch.testing.assert_close(_reduce_frames(frames, 2), frames[:, :, 2])


def test_reduce_frames_accepts_an_arbitrary_callable():
    frames = make_frames()
    torch.testing.assert_close(_reduce_frames(frames, lambda x: x.mean(dim=2)), frames.mean(dim=2))


def test_reduce_frames_rejects_an_unknown_reduction():
    with pytest.raises(ValueError, match="Unrecognized reducefn"):
        _reduce_frames(make_frames(), "median")


# --- _preprocess_xray -----------------------------------------------------------------


def test_preprocess_crops_half_the_pixels_from_each_edge():
    """`crop` is the total removed per axis, so crop=10 takes 5 from each side."""
    img = torch.rand(1, 1, 40, 40)
    assert _preprocess_xray(img, 10, False, False).shape == (1, 1, 30, 30)


def test_preprocess_rescales_to_the_unit_interval():
    img = torch.rand(1, 1, 16, 16) * 500 - 100
    out = _preprocess_xray(img, 0, False, False)
    assert out.min().item() == pytest.approx(0.0, abs=1e-6)
    assert out.max().item() == pytest.approx(1.0, abs=1e-3)


def test_linearize_inverts_contrast():
    """The log transform maps bright transmission to low attenuation, so ordering flips."""
    img = torch.linspace(0, 1, 16).reshape(1, 1, 4, 4)
    out = _preprocess_xray(img, 0, False, True)
    assert out.flatten()[0] > out.flatten()[-1]
    assert torch.isfinite(out).all()


def test_subtract_background_keeps_the_image_in_the_unit_interval():
    img = torch.cat([torch.full((1, 1, 8, 16), 0.2), torch.rand(1, 1, 8, 16)], dim=2)
    out = _preprocess_xray(img, 0, True, False)
    assert out.min().item() >= 0.0 and out.max().item() <= 1.0


# --- _mask_outside_circle -------------------------------------------------------------


def test_mask_outside_circle_blanks_the_corners():
    """A circular detector inscribed in the frame leaves the corners outside."""
    img = torch.ones(1, 1, 32, 32)
    img[..., 0, 0] = 5.0  # a corner value that must be overwritten
    out = _mask_outside_circle(img, radius=16.0)
    assert out[..., 0, 0].item() == img.min().item()
    assert out[..., 16, 16].item() == 1.0


def test_mask_outside_circle_is_a_noop_when_the_radius_covers_the_frame():
    img = torch.rand(1, 1, 16, 16)
    torch.testing.assert_close(_mask_outside_circle(img, radius=100.0), img)


def test_mask_outside_circle_is_centered():
    """The mask is symmetric under a 180-degree rotation of the frame."""
    out = _mask_outside_circle(torch.ones(1, 1, 32, 32), radius=10.0)
    torch.testing.assert_close(out, out.flip(-1).flip(-2))


# --- _parse_dicom_intrinsics ----------------------------------------------------------


def test_intrinsics_prefer_pixel_spacing():
    intrinsics, _ = _parse_dicom_intrinsics(make_dataset(PixelSpacing=[0.3, 0.4]))
    assert (intrinsics.dely, intrinsics.delx) == (0.3, 0.4)


def test_intrinsics_fall_back_to_imager_pixel_spacing():
    intrinsics, _ = _parse_dicom_intrinsics(make_dataset(ImagerPixelSpacing=[0.5, 0.6]))
    assert (intrinsics.dely, intrinsics.delx) == (0.5, 0.6)


def test_intrinsics_require_some_pixel_spacing():
    with pytest.raises(AttributeError, match="Cannot find pixel spacing"):
        _parse_dicom_intrinsics(make_dataset())


def test_detector_origin_defaults_to_zero():
    intrinsics, _ = _parse_dicom_intrinsics(make_dataset(PixelSpacing=[1.0, 1.0]))
    assert (intrinsics.x0, intrinsics.y0) == (0.0, 0.0)


def test_posterior_foot_images_are_flagged_for_reorientation():
    """Both halves of the rule must hold: a PF orientation *and* a negative primary angle."""
    _, pf_to_af = _parse_dicom_intrinsics(
        make_dataset(
            PixelSpacing=[1.0, 1.0], PatientOrientation=["P", "F"], PositionerPrimaryAngle=-30.0
        )
    )
    assert pf_to_af is True


@pytest.mark.parametrize(
    "tags",
    [
        {"PatientOrientation": ["P", "F"], "PositionerPrimaryAngle": 30.0},  # positive angle
        {"PatientOrientation": ["A", "F"], "PositionerPrimaryAngle": -30.0},  # not PF
        {},  # tags absent entirely
    ],
)
def test_other_orientations_are_left_alone(tags):
    _, pf_to_af = _parse_dicom_intrinsics(make_dataset(PixelSpacing=[1.0, 1.0], **tags))
    assert pf_to_af is False


# --- read_xray ------------------------------------------------------------------------


def test_read_xray_returns_a_batched_single_channel_image(tmp_path, write_dicom):
    path = write_dicom(tmp_path / "x.dcm", np.random.default_rng(0).random((24, 32)))
    img, intrinsics, pf_to_af = read_xray(path)
    assert img.shape == (1, 1, 24, 32)
    assert intrinsics.sdd == 1000.0
    assert pf_to_af is False


def test_read_xray_reduces_a_multiframe_series(tmp_path, write_dicom):
    path = write_dicom(tmp_path / "cine.dcm", np.random.default_rng(0).random((5, 16, 16)))
    img, _, _ = read_xray(path, reducefn="max")
    assert img.shape == (1, 1, 16, 16)


def test_read_xray_flips_posterior_foot_images(tmp_path, write_dicom):
    """A PF/RAO acquisition is mirrored so downstream geometry can assume AF."""
    array = np.random.default_rng(0).random((16, 16))
    tags = dict(PatientOrientation=["P", "F"], PositionerPrimaryAngle=-30.0)
    flipped, _, pf_to_af = read_xray(
        write_dicom(tmp_path / "pf.dcm", array, **tags), linearize=False
    )
    plain, _, _ = read_xray(write_dicom(tmp_path / "af.dcm", array), linearize=False)
    assert pf_to_af is True
    torch.testing.assert_close(flipped, plain.flip(-1), atol=1e-5, rtol=1e-5)


def test_read_xray_applies_the_circular_detector_mask(tmp_path, write_dicom):
    path = write_dicom(tmp_path / "x.dcm", np.random.default_rng(0).random((32, 32)))
    masked, _, _ = read_xray(path, radius=8.0)
    unmasked, _, _ = read_xray(path)
    assert masked[..., 0, 0] == masked.min()
    assert not torch.allclose(masked, unmasked)


# --- parse_dicom_pose -----------------------------------------------------------------


def test_parse_dicom_pose_reads_the_positioner_angles(tmp_path, write_dicom):
    path = write_dicom(
        tmp_path / "x.dcm",
        np.random.default_rng(0).random((8, 8)),
        PositionerPrimaryAngle=15.0,
        PositionerSecondaryAngle=-10.0,
        DistanceSourceToPatient=800.0,
    )
    pose = parse_dicom_pose(path, orientation="AP", device="cpu")
    assert pose.matrix.shape == (1, 4, 4)
    assert torch.isfinite(pose.matrix).all()


def test_pa_orientation_flips_the_source_to_patient_sign(tmp_path, write_dicom):
    """`multiplier = -1 if orientation == "PA"` puts the source on the opposite side."""
    path = write_dicom(
        tmp_path / "x.dcm",
        np.random.default_rng(0).random((8, 8)),
        PositionerPrimaryAngle=0.0,
        PositionerSecondaryAngle=0.0,
        DistanceSourceToPatient=800.0,
    )
    ap = parse_dicom_pose(path, orientation="AP", device="cpu")
    pa = parse_dicom_pose(path, orientation="PA", device="cpu")
    torch.testing.assert_close(ap.matrix[0, :3, 3], -pa.matrix[0, :3, 3])
