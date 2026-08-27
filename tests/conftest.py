"""Shared fixtures.

Most helpers in this suite are plain module-level factories, following the style of
`nanodrr`'s tests. The few things that live here are either process-wide settings or
artifacts expensive enough to build once per session.
"""

import numpy as np
import pytest
import torch

# Pose the synthetic X-ray is rendered from. Every registration test knows this is the
# answer, so recovery can be asserted against ground truth instead of a stored trajectory.
TRUE_ROT = (3.0, -4.0, 2.0)
TRUE_XYZ = (6.0, 850.0, -5.0)

DETECTOR = 48
SDD = 1000.0
PIXEL_SPACING = 2.0


def pytest_addoption(parser):
    parser.addoption(
        "--update-golden",
        action="store_true",
        help="Rewrite the stored numerics in tests/data/golden/ instead of comparing against them.",
    )


@pytest.fixture(autouse=True)
def _determinism():
    """Pin the sources of run-to-run variation.

    Thread count is the important one: CPU float reductions are ordered by the number of
    threads, so an unpinned run reproduces neither across machines nor across CI runners.
    """
    torch.manual_seed(0)
    torch.set_num_threads(1)


@pytest.fixture(scope="session")
def write_dicom():
    """Factory writing a minimal but valid X-ray DICOM, for `read_xray` to parse.

    Extra DICOM tags are passed through as keyword arguments, so a test can build the
    variant it needs (multiframe, missing pixel spacing, PF orientation, ...).
    """
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    def _write(path, array, sdd=SDD, spacing=PIXEL_SPACING, **tags):
        array = np.asarray(array)
        scaled = (array - array.min()) / (array.max() - array.min() + 1e-8)
        pixels = (scaled * 65535).astype(np.uint16)

        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.12.1"  # X-Ray Angiographic
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian

        ds = Dataset()
        ds.file_meta = meta
        ds.preamble = b"\0" * 128
        ds.SOPClassUID = meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.Modality = "XA"
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        if pixels.ndim == 3:
            ds.NumberOfFrames = pixels.shape[0]
        ds.Rows, ds.Columns = pixels.shape[-2], pixels.shape[-1]
        ds.DistanceSourceToDetector = sdd
        ds.ImagerPixelSpacing = [spacing, spacing]
        ds.PixelData = pixels.tobytes()
        for tag, value in tags.items():
            setattr(ds, tag, value)

        ds.save_as(path, enforce_file_format=True)
        return path

    return _write


@pytest.fixture(scope="session")
def phantom_ct(tmp_path_factory):
    """A 48^3 CT phantom: three dense blocks in air, at 2 mm isotropic spacing.

    The blocks are deliberately asymmetric along all three axes. A symmetric phantom
    gives the registration landscape multiple equivalent optima, which makes pose
    recovery unstable for reasons that look like numerical drift but are not.
    """
    import nibabel as nib

    size = 48
    volume = np.full((size, size, size), -1000.0, dtype=np.float32)  # air
    volume[10:30, 12:20, 14:34] = 1200.0  # cortical-bone-like slab
    volume[22:26, 28:40, 20:24] = 700.0  # a bar on a different axis
    volume[32:40, 16:24, 30:38] = 300.0  # soft-tissue-like blob

    affine = np.diag([2.0, 2.0, 2.0, 1.0]).astype(np.float32)
    path = tmp_path_factory.mktemp("ct") / "phantom.nii.gz"
    nib.save(nib.Nifti1Image(volume, affine), path)
    return path


@pytest.fixture(scope="session")
def true_pose():
    """The pose `xray_dicom` is rendered from, in the DRR's world frame."""
    from diffdrr.pose import convert

    return convert(
        torch.tensor([TRUE_ROT]),
        torch.tensor([TRUE_XYZ]),
        parameterization="euler_angles",
        convention="ZXY",
        degrees=True,
    )


@pytest.fixture(scope="session")
def xray_dicom(tmp_path_factory, phantom_ct, true_pose, write_dicom):
    """A DICOM X-ray of `phantom_ct`, rendered at `true_pose`.

    Because the image is a DRR rather than a real detector reading, tests that register
    against it must pass `linearize=False`: the log transform in `_preprocess_xray`
    inverts contrast, which would put the metric's optimum nowhere near `true_pose`.
    """
    from diffdrr.data import read
    from diffdrr.drr import DRR

    drr = DRR(
        subject=read(str(phantom_ct), orientation="AP"),
        height=DETECTOR,
        width=DETECTOR,
        sdd=SDD,
        delx=PIXEL_SPACING,
        dely=PIXEL_SPACING,
        x0=0.0,
        y0=0.0,
        reverse_x_axis=False,
        renderer="trilinear",
        voxel_shift=0.0,
    )
    image = drr(true_pose).detach().squeeze().numpy()
    return write_dicom(tmp_path_factory.mktemp("xray") / "phantom.dcm", image)


@pytest.fixture(scope="session")
def offset_pose():
    """A starting pose a few degrees and centimetres off `true_pose`.

    This stands in for what a real initializer hands the optimizer: close enough to be in
    the basin of attraction, far enough that recovering `true_pose` is a real result.
    """
    return {
        "rot": [TRUE_ROT[0] + 4.0, TRUE_ROT[1] + 4.0, TRUE_ROT[2] - 3.0],
        "xyz": [TRUE_XYZ[0] + 12.0, TRUE_XYZ[1] + 15.0, TRUE_XYZ[2] - 12.0],
    }


@pytest.fixture
def golden(request):
    """Compare a run's numerics against a committed reference, or rewrite it.

    The recovery tests prove the optimizer finds the right answer, but they tolerate large
    changes to *how* it gets there. This is the layer that pins the trajectory itself, so a
    change in the metric, the optimizer, or the preprocessing cannot pass unnoticed.

    Regenerate with `pytest --update-golden`, then justify the diff in the pull request.
    """
    import json
    from pathlib import Path

    directory = Path(__file__).parent / "data" / "golden"

    def _compare(name, actual, rtol=1e-4, atol=1e-6):
        path = directory / f"{name}.json"
        if request.config.getoption("--update-golden"):
            directory.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(actual, indent=1) + "\n")
            pytest.skip(f"rewrote {path.relative_to(Path(__file__).parent.parent)}")

        assert path.exists(), f"missing golden file {path}; regenerate with --update-golden"
        expected = json.loads(path.read_text())

        assert actual.keys() == expected.keys()
        for key, value in actual.items():
            if isinstance(value, (int, str)) and not isinstance(value, bool):
                assert value == expected[key], f"{name}.{key}: {value!r} != {expected[key]!r}"
            else:
                np.testing.assert_allclose(
                    value, expected[key], rtol=rtol, atol=atol, err_msg=f"{name}.{key} drifted"
                )

    return _compare
