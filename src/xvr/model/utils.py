from glob import glob
from itertools import zip_longest
from pathlib import Path
from typing import Optional

import torch
from diffdrr.data import load_example_ct, read
from diffdrr.drr import DRR
from peft import LoraConfig, get_peft_model
from torch.utils.data import RandomSampler
from torchio import (
    LabelMap,
    Queue,
    ScalarImage,
    Subject,
    SubjectsDataset,
    SubjectsLoader,
    UniformSampler,
)
from tqdm import tqdm

from ..utils import XrayTransforms, get_4x4
from .network import PoseRegressor
from .scheduler import IdentitySchedule, WarmupCosineSchedule


def initialize_subjects(
    volpath: str,  # Single volume of glob pattern for list of volumes
    maskpath: Optional[str],  # Corresponding mask or glob patterns
    orientation: Optional[str],  # "AP", "PA", or None
    patch_size: Optional[tuple],  # Tuple for random crop sizes (h, w, d)
    num_samples: int,  # Total number of training iterations
    num_workers: int,
    pin_memory: bool,
    replacement: bool = True,
):
    # If only a single subject is passed, load it and return
    single_subject = False
    if Path(volpath).is_file():
        single_subject = True
        subject = read(volpath, maskpath, orientation=orientation)
        return subject, single_subject

    # Else, construct a list of all volumes and masks
    # We assume volumes and masks have the same name, but are in different folders
    volumes = sorted(glob(volpath))
    masks = sorted(glob(maskpath)) if maskpath is not None else []
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
    subjects = SubjectsDataset(subjects)
    subject_sampler = RandomSampler(subjects, replacement, num_samples)

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
    lora_target_modules,
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

    # Optionally, create the LoRA model
    if lora_target_modules is not None:
        print("Creating LoRA version of the model...")
        config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=lora_target_modules,
            lora_dropout=0.1,
            bias="none",
            modules_to_save=["xyz_regression", "rot_regression"],
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()

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

    # If more than one subject is provided, initialize the DRR module with a dummy CT
    drr = DRR(
        subject if subject is not None else load_example_ct(orientation=orientation),
        sdd=sdd,
        height=height,
        delx=delx,
        reverse_x_axis=reverse_x_axis,
        renderer=renderer,
    )
    drr.density = None  # Unload the precomputed density map to free up memory
    if not hasattr(drr, "mask"):
        drr.mask = None
    transforms = XrayTransforms(height)

    # If a single subject was provided, expose their volume and isocenter in the DRR module
    if subject is not None:
        drr.register_buffer("volume", subject.volume.data.squeeze())
        drr.register_buffer("center", torch.tensor(subject.volume.get_center())[None])
    drr = drr.cuda().to(torch.float32)

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
