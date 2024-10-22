from pathlib import Path

import torch
from pydicom import dcmread
from torchvision.transforms.functional import center_crop


def read_xray(
    filename: Path,
    crop: int = 0,
    subtract_background: bool = False,
    linearize: bool = True,
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
    """

    # Get the image and imaging system intrinsics
    img, sdd, delx, dely, x0, y0 = _parse_dicom(filename)

    # Preprocess the X-ray image
    img = _preprocess_xray(img, crop, subtract_background, linearize)

    return img, sdd, delx, dely, x0, y0


def _parse_dicom(filename):
    """Get pixel array and intrinsic parameters from DICOM"""

    # Get the image
    ds = dcmread(filename)
    img = ds.pixel_array
    if img.ndim == 3:
        img = img[0]  # Get the first frame
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

    return img, float(sdd), float(delx), float(dely), float(x0), float(y0)


def _preprocess_xray(img, crop, subtract_background, linearize):
    """Configurable X-ray preprocessing"""

    # Remove edge artifacts caused by the collimator
    if crop != 0:
        *_, height, width = img.shape
        img = center_crop(img, (height - crop, width - crop))

    # Rescale to [0, 1]
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)

    # Subtract background color (the mode image intensity)
    if subtract_background:
        background = img.mode().values.mode().values.item()
        img -= background
        img = torch.clamp(img, -1, 0) + 1  # Restrict to [0, 1]

    # Convert X-ray from exponential to linear form
    if linearize:
        img += 1
        img = img.max().log() - img.log()

    return img
