from ..dicom import read_xray
from ..model.inference import _correct_pose, predict_pose
from ..model.network import load_model
from .base import _RegistrarBase


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
        verbose=1,
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
        gt, sdd, delx, dely, x0, y0, pf_to_af = read_xray(
            i2d, self.crop, self.subtract_background, self.linearize, self.reducefn
        )

        # Predict the pose of the X-ray image
        init_pose = predict_pose(self.model, self.config, gt, sdd, delx, dely, x0, y0)

        # Optionally, correct the pose by warping the CT volume to the template
        init_pose = _correct_pose(init_pose, self.warp, self.volume, self.invert)

        return gt, sdd, delx, dely, x0, y0, pf_to_af, init_pose
