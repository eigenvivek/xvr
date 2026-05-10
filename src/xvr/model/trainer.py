from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
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
        volpath: str,
        maskpath: str,
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
        orientation: str = "AP",
        reverse_x_axis: bool = False,
        parameterization: str = "quaternion_adjugate",
        convention: str = "ZXY",
        model_name: str = "resnet18",
        pretrained: bool = True,
        norm_layer: str = "groupnorm",
        unit_conversion_factor: float = 1000.0,
        p_augmentation: float = 0.5,
        lr: float = 5e-3,
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
        img_threshold: float = 0.10,
        mask_threshold: float = 0.05,
    ):
        """Train a pose regression model.

        Args:
            volpath: CT or directory of CTs for pretraining.
            maskpath: Optional labelmaps corresponding to the CTs.
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
            orientation: Orientation of CT volumes.
            reverse_x_axis: Horizontally flip the rendered DRRs.
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
            img_threshold: Minimum fraction of foreground pixels to keep a DRR.
            mask_threshold: Minimum fraction of mask pixels to keep a DRR.
        """

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
        self.img_threshold = img_threshold
        self.mask_threshold = mask_threshold
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
        vol, seg, world2grid, offset = self.load(subject, pose.matrix.dtype, pose.matrix.device)
        pose = pose.compose(offset)

        # Render a batch of DRRs and keep samples that capture the volume
        contrast = self.contrast_distribution.sample().item()
        tmp = transform_hu_to_density(vol, contrast)

        with torch.no_grad():
            img, mask, keep = self.render_samples(tmp, seg, world2grid, pose)

        # Regress the poses of the DRRs (and optionally convert between reference frames)
        x = self.transforms(self.augmentations(img))
        pred_pose = self.model(x)
        if self.reframe is not None:
            pred_pose = pred_pose.compose(self.reframe)

        # Render DRRs from the predicted poses
        pred_img, pred_mask, _ = self.render_samples(tmp, seg, world2grid, pred_pose)

        # Compute the loss
        img, pred_img = self.transforms(img), self.transforms(pred_img)
        loss, mncc, dgeo, rgeo, tgeo, dice = self.lossfn(
            img, mask, pose, pred_img, pred_mask, pred_pose
        )
        n_kept = keep.sum().clamp(min=1)
        loss = (loss * keep).sum() / (n_kept * self.n_grad_accum_itrs)

        # Optimize the model
        loss.mean().backward()
        if ((itr + 1) % self.n_grad_accum_itrs == 0) or ((itr + 1) == self.n_total_itrs):
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
            voxel2grid = _make_voxel_to_grid(volume.permute(2, 1, 0).shape, device, dtype)
            world2grid = affinv.compose(voxel2grid)
            offset = make_translation(self.drr.center)
            return volume, mask, world2grid, offset

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

        # Make the world2grid matrix
        affine = torch.from_numpy(subject["volume"]["affine"]).to(device, dtype)
        voxel2grid = _make_voxel_to_grid(volume.permute(2, 1, 0).shape, device, dtype)
        world2grid = RigidTransform(affine.inverse()).compose(voxel2grid)

        return volume, mask, world2grid, offset

    def render_samples(self, tmp, seg, world2grid, pose, n_samples=500):
        # Make the cam2grid transform
        cam2grid = self.drr.detector.reorient.compose(pose).compose(world2grid)

        # Initialize the source and target points
        src, tgt = self.drr.detector.source.clone(), self.drr.detector.target.clone()
        tgt = self.drr.detector.calibration(tgt)
        step_size = (tgt - src).norm(dim=-1) / float(n_samples - 1)

        # Create the sampling points
        src_, tgt_ = cam2grid(src), cam2grid(tgt)
        t = torch.linspace(0, 1, n_samples, device="cuda", dtype=src_.dtype)
        pts = torch.lerp(
            src_[:, None, :, None],
            tgt_[:, None, :, None],
            t[None, :, None, None, None],
        )
        B, *_ = pts.shape

        # Sample the volume
        img = F.grid_sample(
            tmp.permute(2, 1, 0)[None, None].expand(B, -1, -1, -1, -1),
            pts,
            mode="bilinear",
            align_corners=False,
        )[:, 0, ..., 0]  # [B, n_samples, N]
        img = img * step_size[:, None, :]

        # Sample the grid
        if seg is not None:
            idx = F.grid_sample(
                seg.permute(2, 1, 0)[None, None].expand(B, -1, -1, -1, -1),
                pts,
                mode="nearest",
                align_corners=False,
            )[:, 0, ..., 0].long()  # [B, n_samples, N]
        else:
            idx = torch.zeros_like(img).long()

        # Render out the image
        C = int(seg.max() + 1)
        out = torch.zeros(B, C, idx.shape[-1], device=img.device, dtype=img.dtype)
        out.scatter_add_(1, idx, img)
        img = out.reshape(B, C, self.drr.detector.height, self.drr.detector.width)

        # Create a foreground mask and collapse potentially multichannel images to a single DRR
        mask = img > 0
        img = img.sum(dim=1, keepdim=True)

        # Discard empty imgs/masks
        if mask.shape[1] == 1:
            keep = mask.to(img).flatten(1).mean(1) > self.img_threshold
        else:
            keep = mask[:, 1:].sum(dim=1, keepdim=True)
            keep = (keep > 0).to(img).flatten(1).mean(1) > self.mask_threshold

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


def make_translation(xyz) -> RigidTransform:
    rot = torch.zeros_like(xyz)
    return convert(rot, xyz, parameterization="euler_angles", convention="ZXY")


def _make_voxel_to_grid(
    shape: torch.Size, device: torch.device, dtype: torch.dtype
) -> RigidTransform:
    r"""Build the affine matrix mapping voxel indices to `grid_sample` normalized coordinates.

    PyTorch's `grid_sample` with `align_corners=False` defines normalized coordinates
    so that the full voxel extent `[-0.5, S - 0.5]` maps to `[-1, 1]`.
    A voxel center at index $$i$$ therefore maps to:

    $$x_{\text{norm}} = \frac{2}{S} i + \left(\frac{1}{S} - 1\right)$$

    This method encodes that per-axis scaling and offset into a homogeneous
    affine matrix applied to voxel-index coordinates.

    Args:
        shape: Volume shape `(1, 1, D, H, W)`.

    Returns:
        Affine matrix mapping voxel indices to `[-1, 1]` normalized grid coordinates.
    """
    *_, D, H, W = shape
    size = torch.tensor([W, H, D], dtype=torch.float32)
    mat = torch.eye(4, dtype=dtype)
    mat[:3, :3] = torch.diag(2.0 / size)
    mat[:3, 3] = 1.0 / size - 1.0
    return RigidTransform(mat.to(device))
