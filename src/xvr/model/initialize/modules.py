import torch
from nanodrr.drr import DRR

from ...utils import XrayTransforms
from ..modules import IdentitySchedule, PoseRegressor, WarmupCosineSchedule


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
    lr,
    n_total_itrs,
    n_warmup_itrs,
    n_grad_accum_itrs,
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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, fused=True)
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
        x0=0.0,
        y0=0.0,
        dtype=torch.float32,
        device="cuda",
    )
    transforms = XrayTransforms(height).cuda()

    return model, drr, transforms, optimizer, scheduler, start_itr, model_number


def _load_checkpoint(ckptpath, reuse_optimizer):
    if ckptpath is not None:
        ckpt = torch.load(ckptpath, weights_only=False)
        if reuse_optimizer:
            return ckpt, ckpt["itr"], ckpt["model_number"]
        else:
            return ckpt, 0, 0
    return None, 0, 0
