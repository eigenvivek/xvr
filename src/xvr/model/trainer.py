from datetime import datetime
from math import prod
from pathlib import Path
from random import choice

import matplotlib.pyplot as plt
import torch
import wandb
from diffdrr.data import load_example_ct
from diffdrr.drr import DRR
from diffdrr.pose import RigidTransform
from diffdrr.visualization import plot_drr
from psutil import virtual_memory
from timm.utils.agc import adaptive_clip_grad as adaptive_clip_grad_
from torchio import ScalarImage
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
        inpath,
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
        renderer,
        orientation,
        reverse_x_axis,
        parameterization,
        convention,
        model_name,
        pretrained,
        norm_layer,
        p_augmentation,
        lr,
        weight_geo,
        batch_size,
        n_total_itrs,
        n_warmup_itrs,
        n_grad_accum_itrs,
        n_save_every_itrs,
    ):
        
        # Record all hyperparameters to be checkpointed
        self.config = locals()
        del self.config["self"]

        # Initialize a lazy list of all 3D volumes
        self.subjects, self.loaded = initialize_subjects(inpath)

        # Initialize all deep learning modulues
        self.model, self.drr, self.transforms, self.optimizer, self.scheduler = initialize_modules(
            model_name,
            pretrained,
            parameterization,
            convention,
            norm_layer,
            sdd,
            height,
            delx,
            reverse_x_axis,
            renderer,
            lr,
            n_total_itrs,
            n_warmup_itrs,
            n_grad_accum_itrs,
        )

        # Initialize the loss function
        self.lossfn = PoseRegressionLoss(sdd, weight_geo)

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
        )
        self.batch_size = batch_size

        self.n_total_itrs = n_total_itrs
        self.n_grad_accum_itrs = n_grad_accum_itrs
        self.n_save_every_itrs = n_save_every_itrs

        self.outpath = outpath
        self.model_number = 0

    def train(self, run=None):
        for itr in (pbar := tqdm(range(self.n_total_itrs), desc="Training model...", ncols=200)):
            
            # Checkpoint the model
            if itr % self.n_save_every_itrs == 0:
                self._checkpoint(itr)
                
            # Run an iteration of the training loop
            try:
                log, imgs = self.step(itr)
            except RuntimeError as e:
                print(e)

            # Log metrics (and optionally save to wandb)
            pbar.set_postfix(log)
            if run is not None:
                self._log_wandb(itr, log, imgs)

    def step(self, itr):
        subject = choice(self.subjects)
        if not self.loaded:
            subject.load()
            log, imgs = self._run_iteration(itr, subject)
            subject.unload()
        else:
            log, imgs = self._run_iteration(itr, subject)
        return log, imgs

    def _render_samples(self, subject, threshold=0.8):
        
        # Sample a batch of random poses
        pose = get_random_pose(**self.pose_distribution, batch_size=self.batch_size).cuda()

        # Render random DRRs and apply augmentations/transforms
        contrast = self.contrast_distribution.sample().item()
        with torch.no_grad():
            img, pose = render(self.drr, pose, subject, contrast, centerize=True)
            keep = (img == 0).to(img).flatten(1).mean(1) < threshold
            img = self.augmentations(img)
            img = self.transforms(img)

        return img, pose, keep, contrast

    def _run_iteration(self, itr, subject):

        # Sample a batch of DRRs
        img, pose, keep, contrast = self._render_samples(subject)

        # Keep only those samples with >80% intersection with the volume
        img = img[keep]
        pose = RigidTransform(pose[keep])

        # Regress the poses and render the predicted DRRs
        pred_pose = self.model(img)
        pred_img, _ = render(self.drr, pred_pose, subject, contrast, centerize=False)
        imgs = torch.concat([img[:4], pred_img[:4]])

        # Compute the loss
        loss, mncc, dgeo, rgeo, tgeo = self.lossfn(img, pose, pred_img, pred_pose)
        loss = loss / self.n_grad_accum_itrs

        # Optimize the model
        self.optimizer.zero_grad()
        loss.mean().backward()
        adaptive_clip_grad_(self.model.parameters())
        if ((itr + 1) % self.n_grad_accum_itrs == 0) or ((itr + 1) == self.n_total_itrs):
            self.optimizer.step()
            self.scheduler.step()

        # Return losses and imgs
        log = {
            "mncc": mncc.mean().item(),
            "dgeo": dgeo.mean().item(),
            "rgeo": rgeo.mean().item(),
            "tgeo": tgeo.mean().item(),
            "loss": loss.mean().item(),
            "lr": self.scheduler.get_last_lr()[0],
            "kept": keep.float().mean().item(),
        }
        return log, imgs

    def _log_wandb(self, itr, log, imgs):
        if itr % 1000 == 0:
            fig, axs = plt.subplots(ncols=4, nrows=2)
            plot_drr(imgs, axs=axs.flatten(), ticks=False)
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
                "date": datetime.now(),
                "config": self.config,
            },
            (savepath := f"{self.outpath}/{self.model_number:04d}.pth"),
        )
        tqdm.write(f"Saving checkpoint: {savepath}")
        self.model_number += 1


def initialize_subjects(inpath):
    # Get all volumes
    subjects = []
    inpath = Path(inpath)
    niftis = [inpath] if inpath.is_file() else sorted(inpath.glob("*.nii.gz"))

    # Lazily load all volumes
    for filepath in tqdm(niftis, desc="Reading CTs...", ncols=200):
        subjects.append(ScalarImage(filepath))

    # If the volumes can fit in memory, read all data now
    # Else, volumes will be individually loaded/unloaded during training (slower but enables bigger training sets)
    if (loaded := _subjects_fit_in_memory(subjects)):
        for subject in tqdm(subjects, desc="Preloading CTs into memory...", ncols=200):
            subject.load()

    return subjects, loaded


def _subjects_fit_in_memory(subjects):
    available = virtual_memory().available / (1024**2)  # Available memory in MiB
    required = sum([_size(subject) for subject in subjects])  # Total memory for all subjects in MiB
    return required < available

def _size(subject: ScalarImage, element_size=4):
    """Size of a volume in MiB (assumes float32)."""
    return element_size * prod(subject.spatial_shape) / (1024**2)


def initialize_modules(
    model_name,
    pretrained,
    parameterization,
    convention,
    norm_layer,
    sdd,
    height,
    delx,
    reverse_x_axis,
    renderer,
    lr,
    n_total_itrs,
    n_warmup_itrs,
    n_grad_accum_itrs,
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

    # Initialize a DRR renderer with a placeholder subject
    drr = DRR(
        load_example_ct(),
        sdd=sdd,
        height=height,
        delx=delx,
        reverse_x_axis=reverse_x_axis,
        renderer=renderer,
    ).cuda()
    transforms = XrayTransforms(height)

    # Initialize the optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    warmup_itrs = n_warmup_itrs / n_grad_accum_itrs
    total_itrs = n_total_itrs / n_grad_accum_itrs
    scheduler = WarmupCosineSchedule(optimizer, warmup_itrs, total_itrs)

    return model, drr, transforms, optimizer, scheduler
