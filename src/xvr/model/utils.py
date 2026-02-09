from itertools import zip_longest
from pathlib import Path
from typing import Optional

import torch
from nanodrr.drr import DRR
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

from nanodrr.data import Subject as Subject

from ..utils import XrayTransforms, get_4x4
from .network import PoseRegressor
from .scheduler import IdentitySchedule, WarmupCosineSchedule


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
        subject = Subject.from_filepath(volpath, maskpath)
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
        )
        return subjects, single_subject

    # Return random crops
    patch_sampler = UniformSampler(patch_size)
    patches_queue = Queue(
        subjects,
        max_length=64,
        samples_per_volume=4,
        sampler=patch_sampler,
        subject_sampler=subject_sampler,
        shuffle_subjects=False,
        num_workers=num_workers,
    )

    subjects = SubjectsLoader(
        patches_queue,
        batch_size=1,
        num_workers=0,
        pin_memory=pin_memory,
    )

    return subjects, single_subject


def initialize_modules(
    model_name,
    pretrained,
    parameterization,
    convention,
    norm_layer,
    unit_conversion_factor,
    sdd,
    height,
    delx,
    orientation,
    reverse_x_axis,
    renderer,
    lr,
    n_total_itrs,
    n_warmup_itrs,
    n_grad_accum_itrs,
    subject,
    disable_scheduler,
    ckptpath,
    reuse_optimizer,
):
    # Initialize the pose regression model
    model = PoseRegressor(
        model_name=model_name,
        pretrained=pretrained,
        parameterization=parameterization,
        convention=convention,
        norm_layer=norm_layer,
        height=height,
        unit_conversion_factor=unit_conversion_factor,
    ).cuda()

    # If a checkpoint is passed, reload the model state
    ckpt, start_itr, model_number = _load_checkpoint(ckptpath, reuse_optimizer)
    if ckpt is not None:
        print("Loading previous model weights...")
        model.load_state_dict(ckpt["model_state_dict"])

    # Initialize the optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if disable_scheduler:
        scheduler = IdentitySchedule(optimizer)
    else:
        warmup_itrs = n_warmup_itrs / n_grad_accum_itrs
        total_itrs = n_total_itrs / n_grad_accum_itrs
        scheduler = WarmupCosineSchedule(optimizer, warmup_itrs, total_itrs)

    # Optionally, reload the optimizer and scheduler
    if ckpt is not None and reuse_optimizer:
        print("Reinitializing optimizer...")
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    model.train()

    # Initialize the DRR module
    drr = DRR.from_carm_intrinsics(
        sdd=sdd,
        delx=delx,
        dely=delx,
        height=height,
        width=height,
        dtype=torch.float32,
        device="cuda",
    )
    transforms = XrayTransforms(height)

    return model, drr, transforms, optimizer, scheduler, start_itr, model_number


def _load_checkpoint(ckptpath, reuse_optimizer):
    if ckptpath is not None:
        ckpt = torch.load(ckptpath, weights_only=False)
        if reuse_optimizer:
            return ckpt, ckpt["itr"], ckpt["model_number"]
        else:
            return ckpt, 0, 0
    return None, 0, 0


def initialize_coordinate_frame(warp, img, invert):
    if warp is None:
        return None
    return get_4x4(warp, img, invert).cuda()
