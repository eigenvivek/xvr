import torch
from diffdrr.pose import RigidTransform

from ..io import read_xray
from .base import _RegistrarBase


class RegistrarRestart(_RegistrarBase):
    def __init__(
        self,
        volume,
        mask,
        orientation,
        ckpt,
        labels=None,
        reducefn="max",
        crop=0,
        subtract_background=False,
        linearize=True,
        equalize=False,
        scales="8",
        n_itrs="100",
        reverse_x_axis=True,
        renderer="trilinear",
        parameterization="euler_angles",
        convention="ZXY",
        voxel_shift=0.0,
        lr_rot=1e-2,
        lr_xyz=1e0,
        patience=10,
        threshold=1e-4,
        max_n_plateaus=3,
        init_only=False,
        saveimg=False,
        verbose=1,
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
            save_kwargs={"type": "fixed"},
        )

        ckpt = torch.load(ckpt, weights_only=False)
        self.init_pose = RigidTransform(ckpt["final_pose"]).cuda()

    def initialize_pose(self, i2d):
        # Preprocess X-ray image and get imaging system intrinsics
        gt, sdd, delx, dely, x0, y0, pf_to_af = read_xray(
            i2d, self.crop, self.subtract_background, self.linearize, self.reducefn
        )
        return gt, sdd, delx, dely, x0, y0, pf_to_af, self.init_pose
