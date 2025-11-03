from datetime import datetime

import matplotlib.pyplot as plt
import torch
import wandb
from diffdrr.pose import RigidTransform
from diffdrr.visualization import plot_drr, plot_mask
from timm.utils.agc import adaptive_clip_grad as adaptive_clip_grad_
from tqdm import tqdm

from ..renderer import render
from .augmentations import XrayAugmentations
from .loss import PoseRegressionLoss
from .sampler import get_random_pose
from .utils import initialize_coordinate_frame, initialize_modules, initialize_subjects


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
        orientation="AP",
        reverse_x_axis=False,
        renderer="trilinear",
        parameterization="se3_log_map",
        convention=None,
        model_name="resnet18",
        pretrained=False,
        norm_layer="groupnorm",
        unit_conversion_factor=1000.0,
        p_augmentation=0.5,
        lr=5e-3,
        weight_ncc=1e0,
        weight_geo=1e-2,
        weight_dice=1e0,
        batch_size=96,
        n_total_itrs=100_000,
        n_warmup_itrs=1_000,
        n_grad_accum_itrs=4,
        n_save_every_itrs=2_500,
        disable_scheduler=False,
        ckptpath=None,
        reuse_optimizer=False,
        lora_target_modules=None,
        warp=None,
        invert=False,
        patch_size=None,
        num_workers=4,
        pin_memory=True,
    ):
        # Record all hyperparameters to be checkpointed
        self.config = locals()
        del self.config["self"]

        # Initialize a lazy list of all 3D volumes
        self.subjects, self.single_subject = initialize_subjects(
            volpath,
            maskpath,
            orientation,
            patch_size,
            n_total_itrs,
            num_workers,
            pin_memory,
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
            orientation,
            reverse_x_axis,
            renderer,
            lr,
            n_total_itrs,
            n_warmup_itrs,
            n_grad_accum_itrs,
            self.subjects if self.single_subject else None,
            disable_scheduler,
            ckptpath,
            reuse_optimizer,
            lora_target_modules,
        )

        # Initialize the loss function
        self.lossfn = PoseRegressionLoss(sdd, weight_ncc, weight_geo, weight_dice)

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

        # Initialize a conversion between the template and canonical frames of reference
        self.reframe = initialize_coordinate_frame(warp, volpath, invert)

        # Save training config
        self.n_total_itrs = n_total_itrs
        self.n_grad_accum_itrs = n_grad_accum_itrs
        self.n_save_every_itrs = n_save_every_itrs
        self.outpath = outpath

    def train(self, run=None):
        pbar = tqdm(
            range(self.start_itr, self.n_total_itrs),
            initial=self.start_itr,
            total=self.n_total_itrs,
            desc="Training model...",
            ncols=200,
        )

        if self.single_subject:
            self.subjects = (None for _ in range(self.n_total_itrs))

        for itr, subject in zip(pbar, self.subjects):
            # Checkpoint the model
            if itr % self.n_save_every_itrs == 0:
                self._checkpoint(itr)

            # Run an iteration of the training loop
            try:
                log, imgs, masks = self.step(itr, subject)
            except Exception as e:
                print(e)
                continue

            # Log metrics (and optionally save to wandb)
            pbar.set_postfix(log)
            if run is not None:
                self._log_wandb(itr, log, imgs, masks)

        # Save the final model
        self._checkpoint(itr)

    def step(self, itr, subject):
        # Sample a batch of DRRs
        img, mask, pose, keep, contrast = self._render_samples(subject)

        # Only keep samples that capture the volume
        img = img[keep]
        mask = mask[keep]
        pose = RigidTransform(pose[keep])

        # Regress the poses (and optionally convert between reference frames)
        pred_pose = self.model(img)
        if self.reframe is not None:
            pred_pose = pred_pose.compose(self.reframe)

        # Render DRRs from the predicted poses
        pred_img, pred_mask, _ = render(
            self.drr, pred_pose, subject, contrast, centerize=False
        )
        pred_img = self.transforms(pred_img)

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
        imgs = torch.concat([img[:4], pred_img[:4]])
        masks = torch.concat([mask[:4], pred_mask[:4]])
        return log, imgs, masks

    def _render_samples(self, subject, img_threshold=0.65, mask_threshold=0.05):
        # Sample a batch of random poses
        pose = get_random_pose(**self.pose_distribution).cuda()

        # Render random DRRs and apply augmentations/transforms
        contrast = self.contrast_distribution.sample().item()
        with torch.no_grad():
            img, mask, pose = render(self.drr, pose, subject, contrast, centerize=True)

            if mask.shape[1] == 1:
                # Keep if >65% of the image is non-zero pixels
                keep = mask.to(img).flatten(1).mean(1) > img_threshold
            else:
                # Keep if >5% of the image contains pixels corresponding to masked structures
                keep = mask[:, 1:].sum(dim=1, keepdim=True)
                keep = (keep > 0).to(img).flatten(1).mean(1) > mask_threshold

            img = self.augmentations(img)
            img = self.transforms(img)

        return img, mask, pose, keep, contrast

    def _log_wandb(self, itr, log, imgs, masks):
        ncols = len(imgs) // 2
        if itr % 250 == 0 and ncols > 0:
            fig, axs = plt.subplots(ncols=ncols, nrows=2)
            plot_drr(imgs, axs=axs.flatten(), ticks=False)
            if masks.shape[1] > 1:
                plot_mask(masks[:, 1:], alpha=0.25, axs=axs.flatten())
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
