from datetime import datetime
from itertools import zip_longest
from pathlib import Path
from random import choice

import matplotlib.pyplot as plt
import torch
import wandb
from diffdrr.data import load_example_ct, read
from diffdrr.drr import DRR
from diffdrr.pose import RigidTransform
from diffdrr.visualization import plot_drr, plot_mask
from timm.utils.agc import adaptive_clip_grad as adaptive_clip_grad_
from torchio import LabelMap, ScalarImage, Subject
from tqdm import tqdm

from ..renderer import render
from ..utils import XrayTransforms
from .augmentations import XrayAugmentations
from .loss import PoseRegressionLoss
from .network import PoseRegressor
from .sampler import get_random_pose
from .scheduler import WarmupCosineSchedule


class Trainer:
    def __init__(
        self,
        volpath,
        maskpath,
        outpath,
        alphamin,
        alphamax,
        betamin,
        betamax,
        gammamin,
        gammamax,
        txmin,
        txmax,
        tymin,
        tymax,
        tzmin,
        tzmax,
        sdd,
        height,
        delx,
        orientation,
        reverse_x_axis,
        renderer="trilinear",
        parameterization="se3_log_map",
        convention=None,
        model_name="resnet18",
        pretrained=False,
        norm_layer="groupnorm",
        p_augmentation=0.5,
        lr=5e-3,
        weight_geo=1e-2,
        weight_dice=1.0,
        batch_size=96,
        n_total_itrs=100_000,
        n_warmup_itrs=1_000,
        n_grad_accum_itrs=4,
        n_save_every_itrs=2_500,
        ckptpath=None,
        reuse_optimizer=False,
        preload_volumes=False,
    ):
        # Record all hyperparameters to be checkpointed
        self.config = locals()
        del self.config["self"]

        # Initialize a lazy list of all 3D volumes
        self.preload_volumes = preload_volumes
        self.subjects, self.single_subject = _initialize_subjects(
            volpath, maskpath, orientation, self.preload_volumes
        )

        # Initialize all deep learning modules
        (
            self.model,
            self.drr,
            self.transforms,
            self.optimizer,
            self.scheduler,
            self.start_itr,
            self.model_number,
        ) = _initialize_modules(
            model_name,
            pretrained,
            parameterization,
            convention,
            norm_layer,
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
            self.subjects if self.single_subject else None,
            ckptpath,
            reuse_optimizer,
        )

        # Initialize the loss function
        self.lossfn = PoseRegressionLoss(sdd, weight_geo, weight_dice)

        # Set up augmentations
        self.contrast_distribution = torch.distributions.Uniform(1.0, 10.0)
        self.augmentations = XrayAugmentations(p_augmentation)

        # Define the pose distribution
        self.pose_distribution = dict(
            alphamin=alphamin,
            alphamax=alphamax,
            betamin=betamin,
            betamax=betamax,
            gammamin=gammamin,
            gammamax=gammamax,
            txmin=txmin,
            txmax=txmax,
            tymin=tymin,
            tymax=tymax,
            tzmin=tzmin,
            tzmax=tzmax,
            batch_size=batch_size,
        )

        self.n_total_itrs = n_total_itrs
        self.n_grad_accum_itrs = n_grad_accum_itrs
        self.n_save_every_itrs = n_save_every_itrs

        self.outpath = outpath

    def train(self, run=None):
        pbar = tqdm(
            range(self.start_itr, self.n_total_itrs),
            desc="Training model...",
            ncols=200,
        )
        for itr in pbar:
            # Checkpoint the model
            if itr % self.n_save_every_itrs == 0:
                self._checkpoint(itr)

            # Run an iteration of the training loop
            try:
                log, imgs, masks = self.step(itr)
            except RuntimeError as e:
                print(e)
                continue

            # Log metrics (and optionally save to wandb)
            pbar.set_postfix(log)
            if run is not None:
                self._log_wandb(itr, log, imgs, masks)

    def step(self, itr):
        if self.single_subject:
            log, imgs, masks = self._step_single_subject(itr)
        else:
            log, imgs, masks = self._step_random_subject(itr)
        return log, imgs, masks

    def _step_single_subject(self, itr):
        log, imgs, masks = self._run_iteration(itr)
        return log, imgs, masks

    def _step_random_subject(self, itr):
        subject = choice(self.subjects)
        if not self.preload_volumes:
            subject.load()
            log, imgs, masks = self._run_iteration(itr, subject)
            subject.unload()
        else:
            log, imgs, masks = self._run_iteration(itr, subject)
        return log, imgs, masks

    def _run_iteration(self, itr, subject=None):
        # Sample a batch of DRRs
        img, mask, pose, keep, contrast = self._render_samples(subject)

        # Keep only those samples with >80% intersection with the volume
        img = img[keep]
        mask = mask[keep]
        pose = RigidTransform(pose[keep])

        # Regress the poses and render the predicted DRRs
        pred_pose = self.model(img)
        pred_img, pred_mask, _ = render(
            self.drr, pred_pose, subject, contrast, centerize=False
        )
        imgs = torch.concat([img[:4], pred_img[:4]])
        masks = torch.concat([mask[:4], pred_mask[:4]])

        # Compute the loss
        loss, mncc, dgeo, rgeo, tgeo, dice = self.lossfn(
            img, mask, pose, pred_img, pred_mask, pred_pose
        )
        loss = loss / self.n_grad_accum_itrs

        # Optimize the model
        loss.mean().backward()
        if ((itr + 1) % self.n_grad_accum_itrs == 0) or (
            (itr + 1) == self.n_total_itrs
        ):
            adaptive_clip_grad_(self.model.parameters())
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        # Return losses and imgs
        log = {
            "mncc": mncc.mean().item(),
            "dgeo": dgeo.mean().item(),
            "rgeo": rgeo.mean().item(),
            "tgeo": tgeo.mean().item(),
            "dice": dice.mean().item(),
            "loss": loss.mean().item(),
            "lr": self.scheduler.get_last_lr()[0],
            "kept": keep.float().mean().item(),
        }
        return log, imgs, masks

    def _render_samples(self, subject, img_threshold=0.8, mask_threshold=0.05):
        # Sample a batch of random poses
        pose = get_random_pose(**self.pose_distribution).cuda()

        # Render random DRRs and apply augmentations/transforms
        contrast = self.contrast_distribution.sample().item()
        with torch.no_grad():
            img, mask, pose = render(self.drr, pose, subject, contrast, centerize=True)

            if mask.shape[1] == 1:
                # Keep if >80% of the image is non-zero pixels
                keep = mask.to(img).flatten(1).mean(1) > img_threshold
            else:
                # Keep if >5% of the image contains pixels corresponding to masked structures
                keep = mask[:, 1:].sum(dim=1, keepdim=True)
                keep = (keep > 0).to(img).flatten(1).mean(1) > mask_threshold

            img = self.augmentations(img)
            img = self.transforms(img)

        return img, mask, pose, keep, contrast

    def _log_wandb(self, itr, log, imgs, masks):
        if itr % 250 == 0:
            fig, axs = plt.subplots(ncols=4, nrows=2)
            plot_drr(imgs, axs=axs.flatten(), ticks=False)
            if masks.shape[1] > 1:
                plot_mask(masks[:, 1:], axs=axs.flatten())
            plt.tight_layout()
            log["imgs"] = fig
            plt.close()
        wandb.log(log)

    def _checkpoint(self, itr):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "itr": itr,
                "model_number": self.model_number,
                "date": datetime.now(),
                "config": self.config,
            },
            (savepath := f"{self.outpath}/{self.model_number:04d}.pth"),
        )
        tqdm.write(f"Saving checkpoint: {savepath}")
        self.model_number += 1


def _initialize_subjects(volpath, maskpath, orientation, preload_volumes):
    # If only a single subject is passed, load it and return
    volpath = Path(volpath)
    if volpath.is_file():
        subject = read(volpath, maskpath, orientation=orientation, center_volume=False)
        single_subject = True
        return subject, single_subject
    else:
        single_subject = False

    # Else, construct a list of all volumes and masks
    # We assume volumes and masks have the same name, but are in different folders
    volumes = sorted(volpath.glob("*.nii.gz"))
    masks = sorted(Path(maskpath).glob("*.nii.gz")) if maskpath is not None else []
    itr = zip_longest(volumes, masks)
    pbar = tqdm(itr, desc="Reading CTs...", total=len(volumes), ncols=200)

    # Lazily load a list of all subjects in the dataset
    subjects = []
    for volpath, maskpath in pbar:
        subject = Subject(
            volume=ScalarImage(volpath),
            mask=LabelMap(maskpath) if maskpath is not None else None,
        )
        subjects.append(subject)

    # Optionally, load all volumes in memory
    if preload_volumes:
        for subject in tqdm(subjects, desc="Preloading CTs into memory...", ncols=200):
            subject.load()

    return subjects, single_subject


def _load_checkpoint(ckptpath, reuse_optimizer):
    if ckptpath is not None:
        ckpt = torch.load(ckptpath, weights_only=False)
        if reuse_optimizer:
            return ckpt, ckpt["itr"], ckpt["model_number"]
        else:
            return ckpt, 0, 0
    return None, 0, 0


def _initialize_modules(
    model_name,
    pretrained,
    parameterization,
    convention,
    norm_layer,
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
    ).cuda()

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

    # Initialize the optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    warmup_itrs = n_warmup_itrs / n_grad_accum_itrs
    total_itrs = n_total_itrs / n_grad_accum_itrs
    scheduler = WarmupCosineSchedule(optimizer, warmup_itrs, total_itrs)

    # If a checkpoint is passed, reload the states for the model, optimizer, and scheduler
    ckpt, start_itr, model_number = _load_checkpoint(ckptpath, reuse_optimizer)
    if ckpt is not None:
        print("Loading previous model weights...")
        model.load_state_dict(ckpt["model_state_dict"])
        if reuse_optimizer:
            print("Reinitializing optimizer...")
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    model.train()

    return model, drr, transforms, optimizer, scheduler, start_itr, model_number
