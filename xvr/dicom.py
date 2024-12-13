from pathlib import Path
from typing import Callable

import torch
from diffdrr.pose import convert
from pydicom import dcmread
from torchvision.transforms.functional import center_crop


def read_xray(
    filename: Path,
    crop: int = 0,
    subtract_background: bool = False,
    linearize: bool = True,
    reducefn: str | int | Callable = "max",
):
    """
    Read and preprocess an X-ray image from a DICOM file. Returns the pixel array and imaging system intrinsics.

    filename : Path
        Path to the DICOM file.
    crop : int, optional
        Number of pixels to crop from each edge of the image.
    subtract_background : bool, optional
        Subtract the mode image intensity from the image.
    linearize : bool, optional
        Convert the X-ray image from exponential to linear form.
    reducefn :
        If DICOM is multiframe, how to extract a single 2D image for registration.
    """

    # Get the image and imaging system intrinsics
    img, sdd, delx, dely, x0, y0, pf_to_af = _parse_dicom(filename)

    # Preprocess the X-ray image
    img = _preprocess_xray(img, crop, subtract_background, linearize, reducefn)

    return img, sdd, delx, dely, x0, y0, pf_to_af


def _parse_dicom(filename):
    """Get pixel array and intrinsic parameters from DICOM"""

    # Get the image
    ds = dcmread(filename)
    img = ds.pixel_array
    img = torch.from_numpy(img).to(torch.float32)[None, None]

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
            img = torch.flip(img, dims=[-1])
            pf_to_af = True
    except AttributeError:
        pass

    return img, float(sdd), float(delx), float(dely), float(x0), float(y0), pf_to_af


def _parse_dicom_pose(filename, orientation):
    multiplier = -1 if orientation == "PA" else 1
    ds = dcmread(filename)
    alpha = float(ds.PositionerPrimaryAngle) / 180 * torch.pi
    beta = float(ds.PositionerSecondaryAngle) / 180 * torch.pi
    sid = multiplier * float(ds.DistanceSourceToPatient)
    pose = convert(
        torch.tensor([[alpha, beta, 0.0]]),
        torch.tensor([[0.0, sid, 0.0]]),
        parameterization="euler_angles",
        convention="ZXY",
    )
    return pose


def _preprocess_xray(img, crop, subtract_background, linearize, reducefn):
    """Configurable X-ray preprocessing"""

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

    # If the image has a temporal dimension, take a max intensity projection
    if img.ndim == 5:
        if reducefn == "max":
            img = img.max(dim=2).values
        elif reducefn == "sum":
            img = img.sum(dim=2)
        elif isinstance(reducefn, int):
            img = img[:, :, reducefn]
        elif isinstance(reducefn, Callable):
            img = reducefn(img)
        elif reducefn is None:
            pass
        else:
            raise ValueError(f"Unrecognized reducefn: {reducefn}")

    return img
