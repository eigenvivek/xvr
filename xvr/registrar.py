from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from diffdrr.metrics import (
    GradientNormalizedCrossCorrelation2d,
    MultiscaleNormalizedCrossCorrelation2d,
)
from diffdrr.pose import convert
from diffdrr.registration import Registration
from diffdrr.visualization import plot_drr
from torchvision.utils import save_image
from tqdm import tqdm

from xvr.dicom import _parse_dicom_pose, read_xray
from xvr.model import _correct_pose, load_model, predict_pose
from xvr.renderer import initialize_drr
from xvr.utils import XrayTransforms


class _RegistrarBase:
    def __init__(
        self,
        volume,
        mask,
        orientation,
        labels,
        crop,
        subtract_background,
        linearize,
        reducefn,
        scales,
        reverse_x_axis,
        renderer,
        parameterization,
        convention,
        lr_rot,
        lr_xyz,
        patience,
        threshold,
        max_n_itrs,
        max_n_plateaus,
        init_only,
        saveimg,
        verbose,
        read_kwargs,
        drr_kwargs,
        save_kwargs,
    ):
        # Initialize a DRR object with placeholder intrinsic parameters
        # These are reset after a real DICOM file is parsed
        self.volume = volume
        self.mask = mask
        self.orientation = orientation
        self.labels = labels
        self.reverse_x_axis = reverse_x_axis
        self.renderer = renderer
        self.read_kwargs = read_kwargs
        self.drr_kwargs = drr_kwargs
        self.drr = initialize_drr(
            self.volume,
            self.mask,
            self.labels,
            self.orientation,
            height=1436,
            width=1436,
            sdd=1020.0,
            delx=0.194,
            dely=0.194,
            x0=0.0,
            y0=0.0,
            reverse_x_axis=self.reverse_x_axis,
            renderer=self.renderer,
            read_kwargs=self.read_kwargs,
            drr_kwargs=self.drr_kwargs,
        )

        # Initialize the image similarity metric
        sim1 = MultiscaleNormalizedCrossCorrelation2d([None, 9], [0.5, 0.5])
        sim2 = GradientNormalizedCrossCorrelation2d(patch_size=11, sigma=10).cuda()
        self.imagesim = lambda x, y, beta: beta * sim1(x, y) + (1 - beta) * sim2(x, y)

        ### Other arguments

        # X-ray preprocessing
        self.crop = crop
        self.subtract_background = subtract_background
        self.linearize = linearize
        self.reducefn = reducefn

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

        # Misc parameters
        self.init_only = init_only
        self.saveimg = saveimg
        self.verbose = verbose
        self.save_kwargs = save_kwargs

    def initialize_pose(self, i2d):
        """Get initial pose estimate and image intrinsics."""
        raise NotImplementedError

    def run(self, i2d, beta=0.5):
        # Predict the initial pose with a pretrained network
        gt, sdd, delx, dely, x0, y0, init_pose = self.initialize_pose(i2d)
        *_, height, width = gt.shape
        intrinsics = {
            "sdd": sdd,
            "height": height,
            "width": width,
            "delx": delx,
            "dely": dely,
            "x0": x0,
            "y0": y0,
        }

        # Update the DRR's intrinsic parameters
        self.drr.set_intrinsics_(**intrinsics)
        if self.init_only:
            return gt, intrinsics, deepcopy(self.drr), init_pose, None, {}

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
                loss = self.imagesim(img, pred_img, beta=beta)
                loss.backward()
                optimizer.step()
                scheduler.step(loss)

                # Record current loss
                if self.verbose > 0:
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
            loss = self.imagesim(img, pred_img, beta=beta)
        nccs.append(loss.item())
        trajectory = _make_csv(
            params,
            nccs,
            alphas,
            columns=["r1", "r2", "r3", "tx", "ty", "tz", "ncc", "lr_rot", "lr_xyz"],
        )

        return (
            gt,
            intrinsics,
            deepcopy(self.drr),
            init_pose,
            reg.pose,
            dict(trajectory=trajectory),
        )

    def __call__(self, i2d, outpath):
        # Make the savepath
        savepath = Path(outpath) / f"{i2d.stem}"
        savepath.mkdir(parents=True, exist_ok=True)

        # Run the registration
        gt, intrinsics, drr, init_pose, final_pose, kwargs = self.run(i2d)

        # Generate DRRs from the intial and final pose estimates
        if self.saveimg:
            init_img = drr(init_pose).detach().cpu()
            if final_pose is not None:
                final_img = drr(final_pose).detach().cpu()
        else:
            init_img = None
            final_img = None

        init_pose = init_pose.matrix.detach().cpu()
        if final_pose is not None:
            final_pose = final_pose.matrix.detach().cpu()

        # Save the results
        self.save(
            savepath,
            gt,
            init_img,
            final_img,
            i2d,
            intrinsics,
            init_pose,
            final_pose,
            kwargs,
        )

    def save(
        self,
        savepath,
        gt,
        init_img,
        final_img,
        i2d,
        intrinsics,
        init_pose,
        final_pose,
        kwargs,
    ):
        # Organize all the passed parameters to xvr.register
        mask = Path(self.mask).resolve() if self.mask is not None else None
        parameters = {
            "drr": {
                "volume": Path(self.volume).resolve(),
                "mask": mask,
                "labels": self.labels,
                "orientation": self.orientation,
                **intrinsics,
                "reverse_x_axis": self.reverse_x_axis,
                "renderer": self.renderer,
                "read_kwargs": self.read_kwargs,
                "drr_kwargs": self.drr_kwargs,
            },
            "xray": {
                "filename": Path(i2d).resolve(),
                "crop": self.crop,
                "subtract_background": self.subtract_background,
                "linearize": self.linearize,
                "reducefn": self.reducefn,
            },
            "optimization": {
                "init_only": self.init_only,
                "scales": self.scales,
                "parameterization": self.parameterization,
                "convention": self.convention,
                "lr_rot": self.lr_rot,
                "lr_xyz": self.lr_xyz,
                "patience": self.patience,
                "max_n_itrs": self.max_n_itrs,
                "max_n_plateaus": self.max_n_plateaus,
            },
            "init_pose": init_pose,
            "final_pose": final_pose,
            **self.save_kwargs,
            **kwargs,
        }

        # Save parameters and all generated images to a temporary directory
        # Then save a compressed folder to the savepath
        torch.save(parameters, f"{savepath}/parameters.pt")
        if self.saveimg:
            save_image(gt, f"{savepath}/gt.png", normalize=True)
            save_image(init_img, f"{savepath}/init_img.png", normalize=True)
            if final_img is not None:
                save_image(final_img, f"{savepath}/final_img.png", normalize=True)


class RegistrarModel(_RegistrarBase):
    def __init__(
        self,
        volume,
        mask,
        ckptpath,
        labels=None,
        crop=0,
        subtract_background=False,
        linearize=True,
        reducefn="max",
        warp=None,
        invert=False,
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
        init_only=False,
        saveimg=False,
        verbose=2,
        read_kwargs={},
        drr_kwargs={},
    ):
        # Initialize the model and its config
        self.ckptpath = ckptpath
        self.model, self.config, self.date = load_model(self.ckptpath, meta=True)

        # Initial pose correction
        self.warp = warp
        self.invert = invert

        super().__init__(
            volume,
            mask,
            self.config["orientation"],
            labels,
            crop,
            subtract_background,
            linearize,
            reducefn,
            scales,
            reverse_x_axis,
            renderer,
            parameterization,
            convention,
            lr_rot,
            lr_xyz,
            patience,
            threshold,
            max_n_itrs,
            max_n_plateaus,
            init_only,
            saveimg,
            verbose,
            read_kwargs,
            drr_kwargs,
            save_kwargs={
                "type": "model",
                "ckptpath": self.ckptpath,
                "date": self.date,
                "warp": self.warp,
                "invert": self.invert,
            },
        )

    def initialize_pose(self, i2d):
        # Preprocess X-ray image and get imaging system intrinsics
        gt, sdd, delx, dely, x0, y0 = read_xray(
            i2d, self.crop, self.subtract_background, self.linearize, self.reducefn
        )

        # Predict the pose of the X-ray image
        init_pose = predict_pose(self.model, self.config, gt, sdd, delx, dely, x0, y0)

        # Optionally, correct the pose by warping the CT volume to the template
        init_pose = _correct_pose(init_pose, self.warp, self.volume, self.invert)

        return gt, sdd, delx, dely, x0, y0, init_pose


class RegistrarDicom(_RegistrarBase):
    def __init__(
        self,
        volume,
        mask,
        orientation,
        labels=None,
        crop=0,
        subtract_background=False,
        linearize=True,
        reducefn="max",
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
        init_only=False,
        saveimg=False,
        verbose=2,
        read_kwargs={},
        drr_kwargs={},
    ):
        super().__init__(
            volume,
            mask,
            orientation,
            labels,
            crop,
            subtract_background,
            linearize,
            reducefn,
            scales,
            reverse_x_axis,
            renderer,
            parameterization,
            convention,
            lr_rot,
            lr_xyz,
            patience,
            threshold,
            max_n_itrs,
            max_n_plateaus,
            init_only,
            saveimg,
            verbose,
            read_kwargs,
            drr_kwargs,
            save_kwargs={"type": "dicom"},
        )

    def initialize_pose(self, i2d):
        # Preprocess X-ray image and get imaging system intrinsics
        gt, sdd, delx, dely, x0, y0 = read_xray(
            i2d, self.crop, self.subtract_background, self.linearize, self.reducefn
        )

        # Parse the pose from dicom parameters
        init_pose = _parse_dicom_pose(i2d, self.orientation).cuda()

        return gt, sdd, delx, dely, x0, y0, init_pose


class RegistrarFixed(_RegistrarBase):
    def __init__(
        self,
        volume,
        mask,
        orientation,
        rot,
        xyz,
        labels=None,
        reducefn="max",
        crop=0,
        subtract_background=False,
        linearize=True,
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
        init_only=False,
        saveimg=False,
        verbose=2,
        read_kwargs={},
        drr_kwargs={},
    ):
        super().__init__(
            volume,
            mask,
            orientation,
            labels,
            crop,
            subtract_background,
            linearize,
            reducefn,
            scales,
            reverse_x_axis,
            renderer,
            parameterization,
            convention,
            lr_rot,
            lr_xyz,
            patience,
            threshold,
            max_n_itrs,
            max_n_plateaus,
            init_only,
            saveimg,
            verbose,
            read_kwargs,
            drr_kwargs,
            save_kwargs={"type": "fixed"},
        )

        rot = torch.tensor([rot], dtype=torch.float32)
        xyz = torch.tensor([xyz], dtype=torch.float32)
        self.init_pose = convert(
            rot, xyz, parameterization=self.parameterization, convention=self.convention
        ).cuda()

    def initialize_pose(self, i2d):
        # Preprocess X-ray image and get imaging system intrinsics
        gt, sdd, delx, dely, x0, y0 = read_xray(
            i2d, self.crop, self.subtract_background, self.linearize, self.reducefn
        )
        return gt, sdd, delx, dely, x0, y0, self.init_pose


def _parse_scales(scales: str, crop: int, height: int):
    pyramid = [1.0] + [float(x) * ((height - crop) / height) for x in scales.split(",")]
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
