from datetime import datetime

import matplotlib.pyplot as plt
import torch
import wandb
from diffdrr.data import transform_hu_to_density
from diffdrr.pose import RigidTransform, convert
from diffdrr.visualization import plot_drr, plot_mask
from timm.utils.agc import adaptive_clip_grad as adaptive_clip_grad_
from tqdm import tqdm

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
        parameterization="quaternion_adjugate",
        convention=None,
        model_name="resnet18",
        pretrained=False,
        norm_layer="groupnorm",
        unit_conversion_factor=1000.0,
        p_augmentation=0.333,
        lr=2e-4,
        weight_ncc=1e0,
        weight_geo=1e-2,
        weight_dice=1e0,
        weight_mvc=0,
        batch_size=96,
        n_total_itrs=100_000,
        n_warmup_itrs=1_000,
        n_grad_accum_itrs=4,
        n_save_every_itrs=1_000,
        disable_scheduler=False,
        ckptpath=None,
        reuse_optimizer=False,
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
        )

        # Initialize the loss function
        self.lossfn = PoseRegressionLoss(
            sdd, weight_ncc, weight_geo, weight_dice, weight_mvc
        )

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
        # Sample a batch of random poses
        pose = get_random_pose(**self.pose_distribution).cuda()

        # Load the subject and translate the pose to its isocenter
        vol, seg, affinv, offset = self.load(
            subject, pose.matrix.dtype, pose.matrix.device
        )
        pose = pose.compose(offset)

        # Render a batch of DRRs and keep samples that capture the volume
        contrast = self.contrast_distribution.sample().item()
        tmp = transform_hu_to_density(vol, contrast)

        with torch.no_grad():
            img, mask, keep = self.render_samples(tmp, seg, affinv, pose)

        img = img[keep]
        mask = mask[keep]
        pose = pose[keep]

        # Regress the poses of the DRRs (and optionally convert between reference frames)
        x = self.transforms(self.augmentations(img))
        pred_pose = self.model(x)
        if self.reframe is not None:
            pred_pose = pred_pose.compose(self.reframe)

        # Render DRRs from the predicted poses
        pred_img, pred_mask, _ = self.render_samples(tmp, seg, affinv, pred_pose)

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

    def load(self, subject, dtype, device):
        # Load 3D imaging data into memory and optionally move the pose to the volume's isocenter
        if subject is None:
            volume, mask, affinv = (
                self.drr.volume,
                self.drr.mask,
                self.drr.affine_inverse,
            )
            offset = make_translation(self.drr.center)
            return volume, mask, affinv, offset

        # Process torchio patch
        volume = subject["volume"]["data"].squeeze().to(device, dtype)
        try:
            mask = subject["mask"]["data"].data.squeeze().to(device, dtype)
        except TypeError:
            mask = None

        # Get the volume's isocenter and construct a translation to it
        affine = torch.from_numpy(subject["volume"]["affine"]).to(dtype=dtype)
        affine = RigidTransform(affine)
        center = (torch.tensor(volume.shape)[None, None] - 1) / 2
        center = affine(center)[0].to(device, dtype)
        offset = make_translation(center)

        # Make the inverse affine
        affine = torch.from_numpy(subject["volume"]["affine"]).to(device, dtype)
        affinv = RigidTransform(affine.inverse())

        return volume, mask, affinv, offset

    def render_samples(
        self, tmp, seg, affinv, pose, img_threshold=0.10, mask_threshold=0.05
    ):
        # Get the source and target locations for every ray in voxel coordinates
        source, target = self.drr.detector(pose, None)
        img = (target - source).norm(dim=-1).unsqueeze(1)
        source, target = affinv(source), affinv(target)

        # Render a batch of DRRs
        img = self.drr.renderer(tmp, source, target, img, mask=seg)
        img = self.drr.reshape_transform(img, batch_size=len(pose))

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


def make_translation(xyz):
    rot = torch.zeros_like(xyz)
    return convert(rot, xyz, parameterization="euler_angles", convention="ZXY")
