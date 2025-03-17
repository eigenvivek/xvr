from pathlib import Path

import numpy as np
import pydicom
import torch
from diffdrrdata.deepfluoro import DeepFluoroDataset
from diffdrrdata.ljubljana import LjubljanaDataset
from diffdrrdata.utils import load_file
from torchio import ScalarImage
from tqdm import tqdm


def write_dicom(
    img,
    filepath,
    source_to_detector_distance,
    rows,
    cols,
    row_spacing,
    col_spacing,
    row_origin,
    col_origin,
):
    metadata = pydicom.Dataset()
    metadata.MediaStorageSOPClassUID = pydicom.uid.XRayRadiofluoroscopicImageStorage
    metadata.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    metadata.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    ds = pydicom.Dataset()
    ds.file_meta = metadata

    ds.Rows = rows  # in pixels
    ds.Columns = cols  # in pixels
    ds.PixelSpacing = [row_spacing, col_spacing]  # in mm
    ds.DistanceSourceToDetector = source_to_detector_distance  # in mm
    ds.DetectorActiveOrigin = [row_origin, col_origin]  # in mm

    ds.BitsStored = 16
    ds.BitsAllocated = ds.BitsStored
    ds.PixelRepresentation = 0  # unsigned
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = img.tobytes()

    ds.save_as(filepath, write_like_original=False)
    
    
def parse_intrinsic_parameters(proj_params):
    intrinsic = proj_params["intrinsic"][:]

    cols = proj_params["num-cols"][()]
    rows = proj_params["num-rows"][()]
    col_spacing = proj_params["pixel-col-spacing"][()]
    row_spacing = proj_params["pixel-row-spacing"][()]

    fx = -intrinsic[0, 0] * col_spacing
    fy = -intrinsic[1, 1] * row_spacing
    assert fx == fy
    source_to_detector_distance = fx

    col_origin = -(cols / 2 - intrinsic[0, -1]) * col_spacing
    row_origin = -(rows / 2 - intrinsic[1, -1]) * row_spacing

    return (
        source_to_detector_distance,
        cols,
        rows,
        col_spacing,
        row_spacing,
        col_origin,
        row_origin,
    )


def convert_to_dcm(proj):
    img = proj["pixels"][:]
    img /= img.max() / (2**16 - 1)
    img = img.astype(np.uint16)

    intrinsic = proj["intrinsic"][:]
    col_spacing = proj["col-spacing"][()]
    row_spacing = proj["row-spacing"][()]
    rows, cols = img.shape

    fx = -intrinsic[0, 0] * col_spacing
    fy = -intrinsic[1, 1] * row_spacing
    assert fx == fy
    source_to_detector_distance = fx

    col_origin = -(cols / 2 - intrinsic[0, -1]) * col_spacing
    row_origin = -(rows / 2 - intrinsic[1, -1]) * row_spacing

    metadata = pydicom.Dataset()
    metadata.MediaStorageSOPClassUID = pydicom.uid.XRayAngiographicImageStorage
    metadata.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    metadata.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    ds = pydicom.Dataset()
    ds.file_meta = metadata

    ds.Rows = rows  # in pixels
    ds.Columns = cols  # in pixels
    ds.PixelSpacing = [row_spacing, col_spacing]  # in mm / pixel
    ds.DistanceSourceToDetector = source_to_detector_distance  # in mm
    ds.DetectorActiveOrigin = [row_origin, col_origin]  # in mm

    ds.BitsStored = 16
    ds.BitsAllocated = ds.BitsStored
    ds.PixelRepresentation = 0  # unsigned
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = img.tobytes()

    return ds


### DeepFluoro ###
print("Converting the DeepFluoro dataset to DICOM...")
DeepFluoroDataset(1)  # Download the dataset if not already
f = load_file("ipcai_2020_full_res_data.h5")
(
    source_to_detector_distance,
    cols,
    rows,
    col_spacing,
    row_spacing,
    col_origin,
    row_origin,
) = parse_intrinsic_parameters(f["proj-params"])
intrinsics = dict(
    source_to_detector_distance=source_to_detector_distance,
    cols=cols,
    rows=rows,
    col_spacing=col_spacing,
    row_spacing=row_spacing,
    col_origin=col_origin,
    row_origin=row_origin,
)

for idx, subject_id in enumerate(
    ["17-1882", "18-1109", "18-0725", "18-2799", "18-2800", "17-1905"]
):
    savepath = Path(f"data/deepfluoro/subject{idx + 1:02d}/xrays")
    savepath.mkdir(parents=True, exist_ok=True)

    projs = f[subject_id]["projections"]
    for proj in tqdm(projs, ncols=75):

        # Parse the image
        img = projs[proj]["image/pixels"][:].astype(np.uint16)
        if projs[proj]["rot-180-for-up"][()]:
            img = np.rot90(img, k=2)

        # Save cols-ray as a DICOM file
        filepath = savepath / f"{proj}.dcm"
        write_dicom(img, filepath, **intrinsics)


### Ljubljana ###
print("\nConverting the Ljubljana dataset to DICOM...")
LjubljanaDataset(1)  # Download the dataset if not already
f = load_file("ljubljana.h5")
for subject in tqdm(f, ncols=75):
    savepath = Path(f"data/ljubljana/{subject}/xrays")
    savepath.mkdir(parents=True, exist_ok=True)

    ds = convert_to_dcm(f[subject]["proj-ap"])
    ds.save_as(savepath / "frontal.dcm", write_like_original=False)

    ds = convert_to_dcm(f[subject]["proj-lat"])
    ds.save_as(savepath / "lateral.dcm", write_like_original=False)

    ds = convert_to_dcm(f[subject]["proj-ap-max"])
    ds.save_as(savepath / "frontal_max.dcm", write_like_original=False)

    ds = convert_to_dcm(f[subject]["proj-lat-max"])
    ds.save_as(savepath / "lateral_max.dcm", write_like_original=False)


for idx in range(1, 11):
    savepath = Path(f"data/ljubljana/subject{idx:02d}/xrays")
    savepath.mkdir(parents=True, exist_ok=True)
    ljubljana = LjubljanaDataset(idx)
    for jdx in tqdm(range(len(ljubljana)), ncols=75):
        img, pose, sdd, height, width, delx, dely, x0, y0 = ljubljana[jdx]
        *_, height, width = img.shape
        if jdx == 0:
            name = "frontal"
        elif jdx == 1:
            name = "lateral"
        else:
            raise ValueError("Unrecognized number of X-rays")
        torch.save(
            {
                "pose": pose.matrix,
                "intrinsics": dict(
                    sdd=sdd,
                    delx=delx,
                    dely=dely,
                    x0=x0,
                    y0=y0,
                    height=height,
                    width=width,
                ),
            },
            savepath / f"{name}.pt",
        )

    # Save volume
    data = ljubljana.subject.volume.data.flip(1).clone()
    affine = ljubljana.subject.volume.affine.copy()
    affine[0, 0] *= -1.0
    volume = ScalarImage(tensor=data, affine=affine)
    volume.save(f"data/ljubljana/subject{idx:02d}/volume.nii.gz")
