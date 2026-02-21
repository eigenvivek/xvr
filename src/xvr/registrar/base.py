import time
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from diffdrr.metrics import (
    GradientNormalizedCrossCorrelation2d,
    MultiscaleNormalizedCrossCorrelation2d,
)
from diffdrr.registration import Registration
from diffdrr.visualization import plot_drr
from torchvision.utils import save_image
from tqdm import tqdm

from ..renderer import initialize_drr
from ..utils import XrayTransforms


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
        equalize,
        reducefn,
        scales,
        n_itrs,
        reverse_x_axis,
        renderer,
        parameterization,
        convention,
        voxel_shift,
        lr_rot,
        lr_xyz,
        patience,
        threshold,
        max_n_plateaus,
        init_only,
        saveimg,
        verbose,
        read_kwargs,
        drr_kwargs,
        save_kwargs,
    ):
        # DRR arguments
        self.volume = volume
        self.mask = mask
        self.orientation = orientation
        self.labels = labels
        self.reverse_x_axis = reverse_x_axis
        self.renderer = renderer
        self.read_kwargs = read_kwargs
        self.drr_kwargs = drr_kwargs

        self.drr_kwargs["voxel_shift"] = voxel_shift

        # X-ray preprocessing
        self.crop = crop
        self.subtract_background = subtract_background
        self.linearize = linearize
        self.equalize = equalize
        self.reducefn = reducefn

        # Registration SE(3) parameterization
        self.parameterization = parameterization
        self.convention = convention

        # Multiscale registration arguments
        self.scales = scales.split(",")
        self.n_itrs = [int(n_itr) for n_itr in n_itrs.split(",")]
        assert len(self.scales) == len(self.n_itrs)

        self.lr_rot = lr_rot
        self.lr_xyz = lr_xyz
        self.patience = patience
        self.threshold = threshold
        self.max_n_plateaus = max_n_plateaus

        # Misc parameters
        self.init_only = init_only
        self.saveimg = saveimg
        self.verbose = verbose
        self.save_kwargs = save_kwargs

        # Initialize a DRR object with placeholder intrinsic parameters
        # These are reset after a real DICOM file is parsed
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

    def initialize_pose(self, i2d: str | Path):
        """Get initial pose estimate and image intrinsics."""
        raise NotImplementedError

    def initialize_imagesim(
        self, mncc_patch_size: int, gncc_patch_size: int, sigma: float, beta: float
    ):
        """Initialize gradient multiscale normalized cross correlation."""
        sim1 = MultiscaleNormalizedCrossCorrelation2d([None, mncc_patch_size], [0.5, 0.5])
        sim2 = GradientNormalizedCrossCorrelation2d(gncc_patch_size, sigma).cuda()
        return lambda x, y: beta * sim1(x, y) + (1 - beta) * sim2(x, y)

    def run(
        self,
        i2d: str | Path,
        mncc_patch_size: int,
        gncc_patch_size: int,
        sigma: float,
        beta: float,
    ):
        # Initialize the image similarity metric
        imagesim = self.initialize_imagesim(mncc_patch_size, gncc_patch_size, sigma, beta)

        # Predict the initial pose with a pretrained network
        gt, sdd, delx, dely, x0, y0, pf_to_af, init_pose = self.initialize_pose(i2d)
        *_, height, width = gt.shape
        intrinsics = dict(
            sdd=sdd,
            height=height,
            width=width,
            delx=delx,
            dely=dely,
            x0=-x0,
            y0=y0,
        )

        # Parse the scales for multiscale registration
        scales = _parse_scales(self.scales, self.crop, height)

        # Update the DRR's intrinsic parameters
        self.drr.set_intrinsics_(**intrinsics)
        if self.init_only:
            self.drr.rescale_detector_(scales[0])
            return (
                gt,
                intrinsics,
                deepcopy(self.drr),
                init_pose,
                None,
                dict(pf_to_af=pf_to_af),
            )

        # Initialize the diffdrr.registration.Registration module
        rot, xyz = init_pose.convert(self.parameterization, self.convention)
        reg = Registration(self.drr, rot, xyz, self.parameterization, self.convention)

        # Run test-time optimization and save the results
        params, nccs, times, alphas = self.run_test_time_optimization(gt, reg, scales, imagesim)
        columns = [
            "r1",
            "r2",
            "r3",
            "tx",
            "ty",
            "tz",
            "ncc",
            "times",
            "lr_rot",
            "lr_xyz",
        ]
        trajectory = _make_csv(params, nccs, times, alphas, columns=columns)

        return (
            gt,
            intrinsics,
            deepcopy(self.drr),
            init_pose,
            reg.pose,
            dict(pf_to_af=pf_to_af, runtime=sum(times), trajectory=trajectory),
        )

    def run_test_time_optimization(self, gt, reg, scales, imagesim):
        # Perform multiscale registration
        params = [torch.concat(reg.pose.convert("euler_angles", "ZXY"), dim=-1).squeeze().tolist()]
        nccs = []
        times = [0.0]
        alphas = [[self.lr_rot, self.lr_xyz]]

        step_size_scalar = 1.0
        for stage, (scale, n_itr) in enumerate(zip(scales, self.n_itrs), start=1):
            # Rescale DRR detector and ground truth image
            reg.drr.rescale_detector_(scale)
            transform = XrayTransforms(
                reg.drr.detector.height,
                reg.drr.detector.width,
                equalize=self.equalize,
            )
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

            pbar = range(n_itr)
            if self.verbose > 0:
                pbar = tqdm(pbar, ncols=100, desc=f"Stage {stage}")

            for itr in pbar:
                torch.cuda.synchronize()
                t0 = time.time()
                optimizer.zero_grad()
                pred_img = reg()
                pred_img = transform(pred_img)
                loss = imagesim(img, pred_img)
                loss.backward()
                optimizer.step()
                scheduler.step(loss.detach())
                torch.cuda.synchronize()
                t1 = time.time()

                # Record current loss
                if self.verbose > 0:
                    pbar.set_postfix_str(f"ncc = {loss.item():5.3f}")
                nccs.append(loss.item())
                times.append(t1 - t0)
                params.append(
                    torch.concat(reg.pose.convert("euler_angles", "ZXY"), dim=-1).squeeze().tolist()
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
            loss = imagesim(img, pred_img)
        nccs.append(loss.item())

        return params, nccs, times, alphas

    def __call__(
        self,
        i2d: str,
        outpath: str,
        mncc_patch_size: int = 9,
        gncc_patch_size: int = 11,
        sigma: float = 0.0,
        beta: float = 0.5,
    ):
        # Make the savepath
        i2d = Path(i2d)
        savepath = Path(outpath) / f"{i2d.stem}"
        savepath.mkdir(parents=True, exist_ok=True)

        # Run the registration
        gt, intrinsics, drr, init_pose, final_pose, kwargs = self.run(
            i2d, mncc_patch_size, gncc_patch_size, sigma, beta
        )

        # Generate DRRs from the initial and final pose estimates
        if self.saveimg:
            init_img = drr(init_pose).detach().cpu()
            if final_pose is not None:
                final_img = drr(final_pose).detach().cpu()
            else:
                final_img = None
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
                "equalize": self.equalize,
                "init_only": self.init_only,
                "scales": self.scales,
                "n_itrs": self.n_itrs,
                "parameterization": self.parameterization,
                "convention": self.convention,
                "lr_rot": self.lr_rot,
                "lr_xyz": self.lr_xyz,
                "patience": self.patience,
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


def _parse_scales(scales: str, crop: int, height: int):
    pyramid = [1.0] + [float(x) * (height / (height + crop)) for x in scales]
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
