from pathlib import Path
from typing import Callable

import numpy as np
import torch
from jaxtyping import Float
from nanodrr.camera import make_rt_inv
from nanodrr.data import Subject
from pydicom import dcmread
from torchvision.transforms.functional import center_crop

from .intrinsics import Intrinsics


def read_xray(
    filename: Path,
    crop: int = 0,
    subtract_background: bool = False,
    linearize: bool = True,
    reducefn: str | int | Callable = "max",
) -> tuple[Float[torch.Tensor, "1 1 H W"], Intrinsics, bool]:
    """Read and preprocess an X-ray image from a DICOM file.

    Loads the pixel array, parses imaging system intrinsics, optionally reorients
    the image, reduces any temporal dimension, and applies configurable preprocessing.

    Args:
        filename: Path to the DICOM file.
        crop: Total number of pixels to remove from each spatial dimension.
            For example, ``crop=10`` removes 5 pixels from each edge.
        subtract_background: If True, subtract the mode image intensity and
            clamp the result to [0, 1].
        linearize: If True, convert the image from exponential to linear form
            by applying a log transform.
        reducefn: Controls how a multiframe (5D) DICOM is collapsed to a single
            2D image. Accepted values:

            - ``"max"``: max intensity projection over the temporal dimension.
            - ``"sum"``: sum projection over the temporal dimension.
            - ``int``: index of the frame to extract.
            - ``Callable``: arbitrary function applied to the 5D tensor.
            - ``None``: no reduction; tensor remains 5D.

    Returns:
        img: Preprocessed image tensor of shape ``(1, 1, H, W)``.
        intrinsics: Imaging system intrinsic parameters.
        pf_to_af: True if the image was reoriented from posterior-foot
            to anterior-foot orientation.
    """
    # Get the image
    ds = dcmread(filename)
    img = torch.from_numpy(ds.pixel_array.astype(np.int32)).to(torch.float32)[None, None]

    # Get the C-arm intrinsics and flip the columns if image is PF
    intrinsics, pf_to_af = _parse_dicom_intrinsics(ds)
    if pf_to_af:
        img = img.flip(-1)

    # Reduce a temporal dimension
    if img.ndim == 5:
        img = _reduce_frames(img, reducefn)

    # Preprocess the X-ray image
    img = _preprocess_xray(img, crop, subtract_background, linearize)

    return img, intrinsics, pf_to_af


def _parse_dicom_intrinsics(ds) -> tuple[Intrinsics, bool]:
    """Parse imaging intrinsics and patient orientation from a pydicom Dataset."""
    # Get intrinsic parameters of the imaging system
    sdd = ds.DistanceSourceToDetector
    try:
        dely, delx = ds.PixelSpacing
    except AttributeError:
        try:
            dely, delx = ds.ImagerPixelSpacing
        except AttributeError:
            raise AttributeError("Cannot find pixel spacing in DICOM file")
    try:
        y0, x0 = ds.DetectorActiveOrigin
    except AttributeError:
        y0, x0 = 0.0, 0.0

    # Reorient RAO images from posterior-foot (PF) to anterior-foot (AF)
    # https://dicom.innolitics.com/ciods/x-ray-angiographic-image/general-image/00200020
    pf_to_af = False
    try:
        if ds.PatientOrientation == ["P", "F"] and ds.PositionerPrimaryAngle < 0:
            pf_to_af = True
    except AttributeError:
        pass

    return Intrinsics(sdd, delx, dely, x0, y0), pf_to_af


def _reduce_frames(img: torch.Tensor, reducefn: str | int | Callable | None) -> torch.Tensor:
    """Collapse the temporal dimension of a 5D image tensor; see ``read_xray`` for ``reducefn`` values."""
    if reducefn is None:
        return img
    elif reducefn == "max":
        return img.max(dim=2).values
    elif reducefn == "sum":
        return img.sum(dim=2)
    elif isinstance(reducefn, int):
        return img[:, :, reducefn]
    elif callable(reducefn):
        return reducefn(img)
    else:
        raise ValueError(f"Unrecognized reducefn: {reducefn}")


def _preprocess_xray(
    img: torch.Tensor,
    crop: int,
    subtract_background: bool,
    linearize: bool,
) -> torch.Tensor:
    """Apply crop, normalization, background subtraction, and optional log-linearization to a 4D X-ray tensor."""
    # Remove edge artifacts caused by the collimator
    if crop != 0:
        *_, height, width = img.shape
        img = center_crop(img, (height - crop, width - crop))

    # Rescale to [0, 1]
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)

    # Subtract background color (the mode image intensity)
    if subtract_background:
        background = img.flatten().mode().values.item()
        img -= background
        img = torch.clamp(img, -1, 0) + 1  # Restrict to [0, 1]

    # Convert X-ray from exponential to linear form
    if linearize:
        img += 1
        img = img.max().log() - img.log()

    return img


def _parse_dicom_pose(
    filename: str,
    orientation: str | None = "AP",
    subject: Subject | None = None,
):
    """Convert DICOM pose params to a C-arm SE(3) pose."""
    multiplier = -1 if orientation == "PA" else 1

    ds = dcmread(filename)
    alpha = float(ds.PositionerPrimaryAngle)
    beta = float(ds.PositionerSecondaryAngle)
    sid = multiplier * float(ds.DistanceSourceToPatient)

    isocenter = subject.isocenter if subject is not None else None

    return make_rt_inv(
        torch.tensor([[alpha, beta, 0.0]]),
        torch.tensor([[0.0, sid, 0.0]]),
        orientation,
        isocenter,
    )
