from datetime import datetime

import matplotlib.pyplot as plt
import torch
import wandb
from diffdrr.visualization import plot_drr, plot_mask
from jaxtyping import Float
from timm.utils.agc import adaptive_clip_grad as adaptive_clip_grad_
from torchio import LabelMap, ScalarImage
from tqdm import tqdm

from nanodrr.data import Subject

from ..config.trainer import args
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
        orientation=args.orientation,
        reverse_x_axis=args.reverse_x_axis,
        renderer=args.renderer,
        parameterization=args.parameterization,
        convention=args.convention,
        model_name=args.model_name,
        pretrained=args.pretrained,
        norm_layer=args.norm_layer,
        unit_conversion_factor=args.unit_conversion_factor,
        p_augmentation=args.p_augmentation,
        lr=args.lr,
        weight_ncc=args.weight_ncc,
        weight_geo=args.weight_geo,
        weight_dice=args.weight_dice,
        weight_mvc=args.weight_mvc,
        batch_size=args.batch_size,
        n_total_itrs=args.n_total_itrs,
        n_warmup_itrs=args.n_warmup_itrs,
        n_grad_accum_itrs=args.n_grad_accum_itrs,
        n_save_every_itrs=args.n_save_every_itrs,
        disable_scheduler=args.disable_scheduler,
        ckptpath=None,
        reuse_optimizer=args.reuse_optimizer,
        warp=None,
        invert=args.invert,
        patch_size=None,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        weights=None,
    ):
        # Record all hyperparameters to be checkpointed
        self.config = locals()
        del self.config["self"]

        # Initialize a lazy list of all 3D volumes
        self.subjects, self.single_subject = initialize_subjects(
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
            orientation,
            reverse_x_axis,
            renderer,
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
            orientation=orientation,
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

    def step(self, itr: int, subject: dict):
        # Load the subject
        subject = self.load(
            subject, mu_water=0.019
        )  # TODO: augment water attenuation here

        # Sample a batch of random poses relative to the subject's coordinate frame
        pose = get_random_pose(subject=subject, **self.pose_distribution)

        # Render a batch of images and only keep samples with sufficient anatomy?
        with torch.no_grad():
            img, mask, keep = self.render_samples(subject, pose)

        img = img[keep]
        mask = mask[keep]
        pose = pose[keep]

        # Regress the poses of the DRRs (and optionally convert between reference frames)
        x = self.transforms(self.augmentations(img))
        pred_pose = self.model(x)
        if self.reframe is not None:
            pred_pose = self.reframe @ pred_pose

        # Render DRRs from the predicted poses
        pred_img, pred_mask, _ = self.render_samples(subject, pred_pose.matrix)

        # Compute the loss
        img, pred_img = self.transforms(img), self.transforms(pred_img)
        loss, mncc, dgeo, rgeo, tgeo, dice, mvc = self.lossfn(
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
            "mvc": mvc.mean().item(),
            "loss": loss.mean().item(),
            "lr": self.scheduler.get_last_lr()[0],
            "kept": keep.float().mean().item(),
        }
        imgs = torch.concat([x[:4], pred_img[:4]])
        masks = torch.concat([mask[:4], pred_mask[:4]])
        return log, imgs, masks

    def load(self, subject: dict, mu_water: float) -> Subject:
        image = ScalarImage(
            tensor=subject["volume"]["data"][0], affine=subject["volume"]["affine"][0]
        )
        try:
            label = LabelMap(
                tensor=subject["mask"]["data"][0], affine=subject["mask"]["affine"][0]
            )
        except (KeyError, TypeError):
            label = None
        return Subject.from_images(
            image, label, convert_to_mu=True, mu_water=mu_water
        ).cuda()

    def render_samples(
        self,
        subject: Subject,
        pose: Float[torch.Tensor, "B 4 4"],
        img_threshold: float = 0.10,
        mask_threshold: float = 0.05,
    ):
        # Render a batch of DRRs
        img = self.drr(subject, pose)

        # Create a foreground mask and collapse potentially multichannel images to a single DRR
        mask = img > 0
        img = img.sum(dim=1, keepdim=True)

        # Discard empty imgs/masks
        if mask.shape[1] == 1:
            # Keep if >10% of the image is non-zero pixels
            keep = mask.to(img).flatten(1).mean(1) > img_threshold
        else:
            # Keep if >5% of the image contains pixels corresponding to masked structures
            keep = mask[:, 1:].sum(dim=1, keepdim=True)
            keep = (keep > 0).to(img).flatten(1).mean(1) > mask_threshold

        return img, mask, keep

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
