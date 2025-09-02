import torch
from diffdrr.pose import convert

from ..dicom import read_xray
from .base import _RegistrarBase


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
        gt, sdd, delx, dely, x0, y0, pf_to_af = read_xray(
            i2d, self.crop, self.subtract_background, self.linearize, self.reducefn
        )
        return gt, sdd, delx, dely, x0, y0, pf_to_af, self.init_pose
