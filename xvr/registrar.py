from pathlib import Path

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from diffdrr.metrics import (
    MultiscaleNormalizedCrossCorrelation2d,
    GradientNormalizedCrossCorrelation2d,
)
from diffdrr.registration import Registration
from diffdrr.visualization import plot_drr

from .dicom import _parse_dicom, _preprocess_xray
from .renderer import initialize_drr
from .model import load_model, predict_pose, _correct_pose
from .utils import XrayTransforms


class Registrar:
    def __init__(
        self,
        volume,
        mask,
        ckptpath,
        crop=0,
        subtract_background=False,
        linearize=True,
        warp=None,
        invert=False,
        model_only=False,
        labels=None,
        scales="8",
        reverse_x_axis=True,
        renderer="trilinear",
        parameterization="euler_angles",
        convention="ZXY",
        lr_rot=1e-2,
        lr_xyz=1e0,
        patience=10,
        threshold=1e-4,
        max_n_itrs=500,
        max_n_plateaus=3,
        verbose=2,
        read_kwargs={},
        drr_kwargs={},
    ):
        # Initialize the model and its config
        self.ckptpath = ckptpath
        self.model, self.config, self.date = load_model(self.ckptpath, meta=True)
        self.model_only = model_only

        # Initialize a DRR object with placeholder intrinsic parameters
        # These are reset after a real DICOM file is parsed
        self.volume = volume
        self.mask = mask
        self.labels = labels

        self.drr = initialize_drr(
            self.volume,
            self.mask,
            self.labels,
            self.config["orientation"],
            height=1436,
            width=1436,
            sdd=1020.0,
            delx=0.194,
            dely=0.194,
            x0=0.0,
            y0=0.0,
            reverse_x_axis=reverse_x_axis,
            renderer=renderer,
            read_kwargs=read_kwargs,
            drr_kwargs=drr_kwargs,
        )

        # Initialize the image similarity metric
        imagesim1 = MultiscaleNormalizedCrossCorrelation2d([None, 9], [0.5, 0.5])
        imagesim2 = GradientNormalizedCrossCorrelation2d(patch_size=11, sigma=10).cuda()
        self.imagesim = lambda x, y: 0.5 * imagesim1(x, y) + 0.5 * imagesim2(x, y)

        ### Other arguments

        # X-ray preprocessing
        self.crop = crop
        self.subtract_background = subtract_background
        self.linearize = linearize

        # Initial pose correction
        self.warp = warp
        self.invert = invert

        # Registration SE(3) parameterization
        self.parameterization = parameterization
        self.convention = convention

        # Multiscale registration arguments
        self.scales = scales
        self.lr_rot = lr_rot
        self.lr_xyz = lr_xyz
        self.patience = patience
        self.threshold = threshold
        self.max_n_itrs = max_n_itrs
        self.max_n_plateaus = max_n_plateaus
        self.verbose = verbose

    def run(self, i2d):
        # Predict the initial pose with a pretrained network
        gt, sdd, delx, dely, x0, y0, init_pose, height = initialize_pose(
            i2d,
            self.model,
            self.config,
            self.crop,
            self.subtract_background,
            self.linearize,
            self.warp,
            self.volume,
            self.invert,
        )

        # Update the DRR's intrinsic parameters
        *_, height, width = gt.shape
        self.drr.set_intrinsics_(sdd, height, width, delx, dely, x0, y0)
        if self.model_only:
            return gt, self.drr, init_pose, None, {}

        # Initialize the diffdrr.registration.Registration module
        rot, xyz = init_pose.convert(self.parameterization, self.convention)
        reg = Registration(self.drr, rot, xyz, self.parameterization, self.convention)

        # Parse the scales for multiscale registration
        scales = _parse_scales(self.scales, self.crop, height)

        # Perform multiscale registration
        params = [
            torch.concat(reg.pose.convert("euler_angles", "ZXY"), dim=-1)
            .squeeze()
            .tolist()
        ]
        nccs = []
        alphas = [[self.lr_rot, self.lr_xyz]]

        step_size_scalar = 1.0
        for stage, scale in enumerate(scales, start=1):
            # Rescale DRR detector and ground truth image
            reg.drr.rescale_detector_(scale)
            transform = XrayTransforms(reg.drr.detector.height, reg.drr.detector.width)
            img = transform(gt).cuda()

            # Initialize the optimizer and scheduler
            step_size_scalar *= 2 ** (stage - 1)
            optimizer = torch.optim.Adam(
                [
                    {"params": [reg.rotation], "lr": self.lr_rot / step_size_scalar},
                    {"params": [reg.translation], "lr": self.lr_xyz / step_size_scalar},
                ],
                maximize=True,
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=0.1,
                patience=self.patience,
                threshold=self.threshold,
                mode="max",
            )

            # Iteratively optimize at this scale until improvements in image similarity plateau
            n_plateaus = 0
            current_lr = torch.inf

            pbar = range(self.max_n_itrs)
            if self.verbose > 0:
                pbar = tqdm(pbar, ncols=100, desc=f"Stage {stage}")

            for itr in pbar:
                optimizer.zero_grad()
                pred_img = reg()
                pred_img = transform(pred_img)
                loss = self.imagesim(img, pred_img)
                loss.backward()
                optimizer.step()
                scheduler.step(loss)

                # Record current loss
                if self.verbose > 1:
                    pbar.set_postfix_str(f"ncc = {loss.item():5.3f}")
                nccs.append(loss.item())
                params.append(
                    torch.concat(reg.pose.convert("euler_angles", "ZXY"), dim=-1)
                    .squeeze()
                    .tolist()
                )

                # Determine update to the learning rate
                lr = scheduler.get_last_lr()
                alphas.append(lr)
                if lr[0] < current_lr:
                    current_lr = lr[0]
                    n_plateaus += 1
                    if self.verbose > 1:
                        tqdm.write("→ Plateaued... decreasing step size")
                if n_plateaus == self.max_n_plateaus:
                    break

                if self.verbose > 2:
                    if itr % 5 == 0:
                        plot_drr(torch.concat([img, pred_img, img - pred_img]))
                        plt.show()

        # Record the final NCC value
        with torch.no_grad():
            pred_img = reg()
            pred_img = transform(pred_img)
            loss = self.imagesim(img, pred_img)
        nccs.append(loss.item())
        trajectory = _make_csv(
            params,
            nccs,
            alphas,
            columns=["r1", "r2", "r3", "tx", "ty", "tz", "ncc", "lr_rot", "lr_xyz"],
        )

        return gt, self.drr, init_pose, reg.pose, dict(trajectory=trajectory)

    def __call__(self, i2d, outpath):
        savepath = Path(outpath) / f"{i2d.stem}.pt"
        savepath.parent.mkdir(parents=True, exist_ok=True)

        drr, init_pose, final_pose, kwargs = self.run(i2d)
        init_pose = init_pose.matrix.detach().cpu()
        if final_pose is not None:
            final_pose = final_pose.matrix.detach().cpu()
        if self.warp is not None:
            warp = Path(self.warp).resolve()

        torch.save(
            {
                "arguments": {
                    "i2d": Path(i2d).resolve(),
                    "volume": Path(self.volume).resolve(),
                    "mask": Path(self.mask).resolve(),
                    "ckptpath": Path(self.ckptpath).resolve(),
                    "outpath": Path(outpath).resolve(),
                    "crop": self.crop,
                    "subtract_background": self.subtract_background,
                    "linearize": self.linearize,
                    "warp": warp,
                    "invert": self.invert,
                    "model_only": self.model_only,
                    "labels": self.labels,
                    "scales": self.scales,
                    "reverse_x_axis": self.reverse_x_axis,
                    "parameterization": self.parameterization,
                    "convention": self.convention,
                    "lr_rot": self.lr_rot,
                    "lr_xyz": self.lr_xyz,
                    "patience": self.patience,
                    "max_n_itrs": self.max_n_itrs,
                    "max_n_plateaus": self.max_n_plateaus,
                },
                "intrinsics": {
                    "sdd": drr.detector.sdd,
                    "delx": drr.detector.delx,
                    "dely": drr.detector.dely,
                    "x0": drr.detector.x0,
                    "y0": drr.detector.y0,
                },
                "init_pose": init_pose,
                "final_pose": final_pose,
                "config": self.config,
                "date": self.date,
                **kwargs,
            },
            savepath,
        )


def initialize_pose(
    i2d,
    model,
    config,
    crop,
    subtract_background,
    linearize,
    warp,
    volume,
    invert,
    meta=True,
):
    """Get initial pose estimate and image intrinsics."""

    # Preprocess X-ray image and get imaging system intrinsics
    gt, sdd, delx, dely, x0, y0 = _parse_dicom(i2d)
    gt = _preprocess_xray(gt, crop, subtract_background, linearize)

    # Predict the pose of the X-ray image
    init_pose, height = predict_pose(model, config, gt, sdd, delx, dely, x0, y0, meta)

    # Optionally, correct the pose by warping the CT volume to the template
    init_pose = _correct_pose(init_pose, warp, volume, invert)

    return gt, sdd, delx, dely, x0, y0, init_pose, height


def _parse_scales(scales: str, crop: int, height: int):
    pyramid = [1.0] + [float(x) * (height / (height + crop)) for x in scales.split(",")]
    scales = []
    for idx in range(len(pyramid) - 1):
        scales.append(pyramid[idx] / pyramid[idx + 1])
    return scales


def _make_csv(*metrics, columns):
    import numpy as np
    import pandas as pd

    ls = []
    for metric in metrics:
        metric = np.array(metric)
        if metric.ndim == 1:
            metric = metric[..., np.newaxis]
        ls.append(metric)
    ls = np.concatenate(ls, axis=1)
    df = pd.DataFrame(ls, columns=columns)
    return df
