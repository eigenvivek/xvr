import random
from itertools import zip_longest
from pathlib import Path
from typing import Optional

import torch
from nanodrr.data.io import Subject as NanoSubject
from nanodrr.data.preprocess import hu_to_mu
from torch.utils.data import WeightedRandomSampler
from torchio import (
    Compose,
    LabelMap,
    Queue,
    ScalarImage,
    Subject,
    SubjectsDataset,
    SubjectsLoader,
    Transform,
    UniformSampler,
)
from tqdm import tqdm


def initialize_subjects(
    volpath: str,  # A single CT or a directory with multiple volumes
    maskpath: Optional[str],  # Optional labelmaps corresponding to the CTs
    patch_size: Optional[tuple],  # Tuple for random crop sizes (h, w, d)
    num_samples: int,  # Total number of training iterations
    num_workers: int,  # Number of workers for the dataloader
    pin_memory: bool,  # Pin memory for the dataloader
    weights: Optional[tuple[float, ...]] = None,  # Sampling probability for each volume
    replacement: bool = True,  # Sample with replacement
):
    # If only a single subject is passed, load it and return
    single_subject = False
    if Path(volpath).is_file():
        single_subject = True
        subject = NanoSubject.from_filepath(volpath, maskpath)
        return subject, single_subject

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
        subject = Subject(volume=volume, mask=mask)
        subjects.append(subject)

    # Construct a dataloader with efficient IO
    subjects = SubjectsDataset(subjects, transform=Compose([RandomHUToMu()]))
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
        return SubjectIterator(subjects), single_subject

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

    return SubjectIterator(subjects), single_subject


class RandomHUToMu(Transform):
    def __init__(
        self,
        mu_water: float = 0.0192,
        mu_bone_range: tuple[float, float] = (0.0, 0.2),
        hu_bone: float = 1000.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mu_water = mu_water
        self.mu_bone_range = mu_bone_range
        self.hu_bone = hu_bone

    def apply_transform(self, subject: Subject) -> Subject:
        mu_bone = random.uniform(*self.mu_bone_range)
        for image in subject.get_images(intensity_only=True):
            image.set_data(hu_to_mu(image.data, self.mu_water, mu_bone, self.hu_bone))
        return subject


class SubjectIterator:
    """Wraps a SubjectsLoader to yield Subject objects instead of dicts."""

    def __init__(self, loader: SubjectsLoader):
        self.loader = loader

    def _to_subject(self, data: dict) -> NanoSubject:
        imagedata = NanoSubject._to_bcdhw(data["volume"]["data"][0]).float()
        affine = torch.from_numpy(data["volume"]["affine"][0], dtype=torch.float32)

        mask_data = data.get("mask")
        labeldata = (
            NanoSubject._to_bcdhw(mask_data["data"][0]).float()
            if mask_data is not None
            else torch.zeros_like(imagedata)
        )

        voxel_to_world = affine
        world_to_voxel = torch.inverse(affine)
        voxel_to_grid = NanoSubject._make_voxel_to_grid(imagedata.shape)

        center = (torch.tensor(imagedata.shape[2:], dtype=torch.float32) - 1) / 2
        isocenter = voxel_to_world[:3, :3] @ center + voxel_to_world[:3, 3]

        return NanoSubject(
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
