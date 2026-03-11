from itertools import repeat, zip_longest
from pathlib import Path
from typing import Optional

import torch
from nanodrr.data import Subject
from torch.utils.data import WeightedRandomSampler
from torchio import (
    LabelMap,
    Queue,
    ScalarImage,
    SubjectsDataset,
    SubjectsLoader,
    UniformSampler,
)
from torchio import Subject as TorchioSubject
from tqdm import tqdm


def initialize_subjects(
    volpath: str,
    maskpath: Optional[str],
    patch_size: Optional[tuple],
    num_samples: int,
    num_workers: int,
    pin_memory: bool,
    weights: Optional[tuple[float, ...]] = None,
    replacement: bool = True,
):
    """Initializes an iterator of subjects for training.

    Args:
        volpath: Path to a single CT volume file or a directory containing
            multiple volumes as .nii.gz files.
        maskpath: Path to a single label map or a directory of label maps
            corresponding to the CTs. If None, masks are omitted.
        patch_size: Spatial dimensions (h, w, d) for random cropping. If None,
            entire volumes are returned.
        num_samples: Total number of training iterations, used to size the sampler.
        num_workers: Number of worker processes for the dataloader.
        pin_memory: Whether to pin memory in the dataloader for faster GPU transfers.
        weights: Sampling probability for each volume. If None, volumes are
            sampled uniformly.
        replacement: Whether to sample volumes with replacement.
    """
    # If only a single subject is passed, load it and return an infinite iterator
    if Path(volpath).is_file():
        subject = Subject.from_filepath(volpath, maskpath).cuda()
        return repeat(subject)

    # Else, construct a list of all volumes and masks
    # We assume volumes and masks have the same name, but are in different folders
    volumes = sorted(Path(volpath).glob("[!.]*.nii.gz"))
    masks = sorted(Path(maskpath).glob("[!.]*.nii.gz")) if maskpath is not None else []
    itr = zip_longest(volumes, masks)
    pbar = tqdm(itr, desc="Lazily loading CTs...", total=len(volumes), ncols=200)

    # Lazily load a list of all subjects in the dataset
    subjects = []
    for volpath, maskpath in pbar:
        volume = ScalarImage(volpath)
        mask = LabelMap(maskpath) if maskpath is not None else None
        subject = TorchioSubject(volume=volume, mask=mask)
        subjects.append(subject)

    # Construct a dataloader with efficient IO
    subjects = SubjectsDataset(subjects)
    if weights is None:
        weights = [1 / len(volumes) for _ in range(len(volumes))]
    subject_sampler = WeightedRandomSampler(
        weights, replacement=replacement, num_samples=num_samples
    )

    # Return entire volumes
    if patch_size is None:
        subjects = SubjectsLoader(
            subjects,
            sampler=subject_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=4,
        )
        return SubjectIterator(subjects)

    # Return random crops
    patch_sampler = UniformSampler(patch_size)
    patches_queue = Queue(
        subjects,
        max_length=160,
        samples_per_volume=32,
        sampler=patch_sampler,
        subject_sampler=subject_sampler,
        shuffle_subjects=False,
        shuffle_patches=True,
        num_workers=num_workers,
    )

    subjects = SubjectsLoader(
        patches_queue,
        batch_size=1,
        num_workers=0,
        pin_memory=pin_memory,
    )

    return SubjectIterator(subjects)


class SubjectIterator:
    """Wraps a SubjectsLoader to yield Subject objects instead of dicts."""

    def __init__(self, loader: SubjectsLoader):
        self.loader = loader

    def _to_subject(self, data: dict) -> Subject:
        imagedata = Subject._to_bcdhw(data["volume"]["data"][0]).float()
        affine = torch.from_numpy(data["volume"]["affine"][0]).to(dtype=torch.float32)

        mask_data = data.get("mask")
        labeldata = (
            Subject._to_bcdhw(mask_data["data"][0]).float()
            if mask_data is not None
            else torch.zeros_like(imagedata)
        )

        voxel_to_world = affine
        world_to_voxel = torch.inverse(affine)
        voxel_to_grid = Subject._make_voxel_to_grid(imagedata.shape)

        center = (torch.tensor(imagedata.shape[2:], dtype=torch.float32) - 1) / 2
        isocenter = voxel_to_world[:3, :3] @ center + voxel_to_world[:3, 3]

        return Subject(
            imagedata,
            labeldata,
            voxel_to_world,
            world_to_voxel,
            voxel_to_grid,
            isocenter,
        ).cuda()

    def __iter__(self):
        for data in self.loader:
            yield self._to_subject(data)
