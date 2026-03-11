from datetime import datetime

import matplotlib.pyplot as plt
import torch
import wandb
from jaxtyping import Float
from nanodrr.data import Subject
from nanodrr.plot import plot_drr
from timm.utils.agc import adaptive_clip_grad
from tqdm import tqdm

from .data import XrayAugmentations, get_random_pose
from .initialize import (
    initialize_coordinate_frame,
    initialize_modules,
    initialize_subjects,
)
from .modules import PoseRegressionLoss

torch.set_float32_matmul_precision("high")


class Trainer:
    def __init__(
        self,
        volpath: str,
        outpath: str,
        alphamin: float,
        alphamax: float,
        betamin: float,
        betamax: float,
        gammamin: float,
        gammamax: float,
        txmin: float,
        txmax: float,
        tymin: float,
        tymax: float,
        tzmin: float,
        tzmax: float,
        sdd: float,
        height: int,
        delx: float,
        maskpath: str | None = None,
        orientation: str = "AP",
        reverse_x_axis: bool = False,
        parameterization: str = "quaternion_adjugate",
        convention: str = "ZXY",
        model_name: str = "resnet18",
        pretrained: bool = False,
        norm_layer: str = "groupnorm",
        unit_conversion_factor: float = 1000.0,
        p_augmentation: float = 0.333,
        lr: float = 2e-4,
        weight_ncc: float = 1e0,
        weight_geo: float = 1e-2,
        weight_dice: float = 1e0,
        batch_size: int = 116,
        n_total_itrs: int = 1_000_000,
        n_warmup_itrs: int = 1_000,
        n_grad_accum_itrs: int = 4,
        n_save_every_itrs: int = 1_000,
        disable_scheduler: bool = False,
        ckptpath: str | None = None,
        reuse_optimizer: bool = False,
        warp: str | None = None,
        invert: bool = False,
        patch_size: tuple[int, int, int] | None = None,
        num_workers: int = 4,
        pin_memory: bool = False,
        weights: list[float] | None = None,
        use_compile: bool = False,
        use_bf16: bool = False,
        img_threshold: float = 0.10,
        mask_threshold: float = 0.05,
    ):
        """Train a pose regression model.

        Args:
            volpath: CT or directory of CTs for pretraining.
            outpath: Directory in which to save model weights.
            alphamin: Minimum primary angle (in degrees).
            alphamax: Maximum primary angle (in degrees).
            betamin: Minimum secondary angle (in degrees).
            betamax: Maximum secondary angle (in degrees).
            gammamin: Minimum tertiary angle (in degrees).
            gammamax: Maximum tertiary angle (in degrees).
            txmin: Minimum x-offset (in millimeters).
            txmax: Maximum x-offset (in millimeters).
            tymin: Minimum y-offset (in millimeters).
            tymax: Maximum y-offset (in millimeters).
            tzmin: Minimum z-offset (in millimeters).
            tzmax: Maximum z-offset (in millimeters).
            sdd: Source-to-detector distance (in millimeters).
            height: DRR height (in pixels).
            delx: DRR pixel size (in millimeters / pixel).
            maskpath: Optional labelmaps corresponding to the CTs.
            orientation: Orientation of CT volumes.
            reverse_x_axis: Obey radiologic convention (e.g., heart on right).
            parameterization: Parameterization of SO(3) for regression.
            convention: If parameterization='euler_angles', specify order.
            model_name: Name of model to instantiate from the timm library.
            pretrained: Load pretrained ImageNet-1k weights.
            norm_layer: Normalization layer.
            unit_conversion_factor: Scale factor for translation prediction (e.g., from m to mm).
            p_augmentation: Base probability of image augmentations during training.
            lr: Maximum learning rate.
            weight_ncc: Weight on mNCC loss term.
            weight_geo: Weight on geodesic loss term.
            weight_dice: Weight on Dice loss term.
            batch_size: Number of DRRs per batch.
            n_total_itrs: Number of iterations for training the model.
            n_warmup_itrs: Number of iterations for warming up the learning rate.
            n_grad_accum_itrs: Number of iterations for gradient accumulation.
            n_save_every_itrs: Number of iterations before saving a new model checkpoint.
            disable_scheduler: Turn off cosine learning rate scheduler.
            ckptpath: Checkpoint of a pretrained pose regressor.
            reuse_optimizer: Initialize the previous optimizer's state.
            warp: SimpleITK transform to warp input CT to checkpoint's reference frame.
            invert: Whether to invert the warp or not.
            patch_size: Optional random crop size; if None, return entire volume.
            num_workers: Number of subprocesses to use in the dataloader.
            pin_memory: Copy volumes into CUDA pinned memory before returning.
            weights: Probability for sampling each volume in volpath.
            use_compile: Compile forward pass with max-autotune-no-cudagraphs.
            use_bf16: Run all ops in bf16.
            img_threshold: Minimum fraction of foreground pixels to keep a DRR.
            mask_threshold: Minimum fraction of mask pixels to keep a DRR.
        """
        # Record all hyperparameters to be checkpointed
        self.config = locals()
        del self.config["self"]

        # Initialize a lazy list of all 3D volumes
        self.subjects = initialize_subjects(
            volpath,
            maskpath,
            patch_size,
            n_total_itrs,
            num_workers,
            pin_memory,
            weights,
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
        ) = initialize_modules(
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
        )

        # Initialize the loss function
        self.lossfn = PoseRegressionLoss(sdd, weight_ncc, weight_geo, weight_dice)

        # Set up augmentations
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
            orientation=orientation,
        )

        # Initialize a conversion between the template and canonical frames of reference
        self.reframe = initialize_coordinate_frame(warp, volpath, invert)

        # Save training config
        self.n_total_itrs = n_total_itrs
        self.n_grad_accum_itrs = n_grad_accum_itrs
        self.n_save_every_itrs = n_save_every_itrs
        self.outpath = outpath

        self.img_threshold = img_threshold
        self.mask_threshold = mask_threshold

        # Set up training optimizations (compile/bf16)
        self.use_compile = use_compile
        self.use_bf16 = use_bf16
        self.dtype = torch.bfloat16 if self.use_bf16 else torch.float32
        if self.use_compile:
            self.compute_loss = torch.compile(
                self.compute_loss, fullgraph=True, mode="max-autotune-no-cudagraphs"
            )

    def train(self, run=None):
        pbar = tqdm(
            range(self.start_itr, self.n_total_itrs),
            initial=self.start_itr,
            total=self.n_total_itrs,
            desc="Training model...",
            ncols=200,
        )

        for itr, subject in zip(pbar, self.subjects):
            # Checkpoint the model
            if itr % self.n_save_every_itrs == 0:
                self._checkpoint(itr)

            # Run an iteration of the training loop
            log, imgs, masks = self.step(itr, subject)

            # Log metrics (and optionally save to wandb)
            pbar.set_postfix({k: f"{v:.3f}" for k, v in log.items()})
            if run is not None:
                self._log_wandb(itr, log, imgs, masks)

        # Save the final model
        self._checkpoint(itr)

    def step(self, itr: int, subject: Subject):
        # Compute the loss for a single step
        loss, metrics, keep, imgs, masks = self.compute_loss(subject.to(self.dtype))
        loss.backward()

        # Optimize the model
        if ((itr + 1) % self.n_grad_accum_itrs == 0) or ((itr + 1) == self.n_total_itrs):
            adaptive_clip_grad(self.model.parameters())
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        # Return losses and imgs
        log = {
            "mncc": metrics.mncc.mean().item(),
            "dgeo": metrics.dgeo.mean().item(),
            "rgeo": metrics.rgeo.mean().item(),
            "tgeo": metrics.tgeo.mean().item(),
            "dice": metrics.dice.mean().item(),
            "loss": loss.item(),
            "lr": self.scheduler.get_last_lr()[0],
            "kept": keep.float().mean().item(),
        }
        return log, imgs, masks

    def compute_loss(self, subject: Subject):
        with torch.autocast(device_type="cuda", dtype=self.dtype, enabled=self.use_bf16):
            # Sample a batch of random poses relative to the subject's coordinate frame
            pose = get_random_pose(subject=subject, **self.pose_distribution)

            # Vary the bone attenuation coefficient
            mu_bone = torch.empty(1, device=subject._image_hu.device).uniform_(0.0, 0.3)
            subject.set_mu(mu_bone=mu_bone)

            # Render a batch of images and flag samples with sufficient anatomy in the view
            with torch.no_grad():
                img, mask, keep = self.render_samples(subject, pose)
                x = self.augmentations(img.float())
                x = self.transforms(x)

            # Regress the poses of the DRRs (and optionally convert between reference frames)
            pred_pose = self.model(x)
            if self.reframe is not None:
                pred_pose = self.reframe @ pred_pose

            # Render DRRs from the predicted poses
            pred_img, pred_mask, _ = self.render_samples(subject, pred_pose)

            # Recenter both poses at the world origin
            shift = torch.zeros(1, 4, 4, device=pose.device, dtype=pose.dtype)
            shift[:, :3, 3] = subject.isocenter
            pose, pred_pose = pose - shift, pred_pose - shift

            # Compute the loss
            img, pred_img = self.transforms(img), self.transforms(pred_img)
            loss, metrics = self.lossfn(img, mask, pose, pred_img, pred_mask, pred_pose)
            n_kept = keep.sum().clamp(min=1)
            loss = (loss * keep).sum() / (n_kept * self.n_grad_accum_itrs)

            # Save images
            imgs = torch.concat([x[:4], pred_img[:4]])
            masks = torch.concat([mask[:4], pred_mask[:4]])

        return loss, metrics, keep, imgs, masks

    def render_samples(
        self, subject: Subject, pose: Float[torch.Tensor, "B 4 4"]
    ) -> tuple[
        Float[torch.Tensor, "B 1 H W"],
        Float[torch.Tensor, "B C H W"],
        Float[torch.Tensor, "B C"],
    ]:
        # Render a batch of DRRs
        img = self.drr(subject, pose)

        # Create a foreground mask and collapse potentially multichannel images to a single DRR
        mask = img > 0
        img = img.sum(dim=1, keepdim=True)

        # Flag empty images/masks
        if mask.shape[1] == 1:
            keep = mask.float().flatten(1).mean(1) > self.img_threshold
        else:
            keep = mask[:, 1:].sum(dim=1, keepdim=True)
            keep = (keep > 0).float().flatten(1).mean(1) > self.mask_threshold

        return img, mask, keep

    def _log_wandb(self, itr, log, imgs, masks):
        ncols = len(imgs) // 2
        if itr % 250 == 0 and ncols > 0:
            fig, axs = plt.subplots(ncols=ncols, nrows=2)
            plot_drr(imgs, masks, axs=axs.flatten(), ticks=False)
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
