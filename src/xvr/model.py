import math

import torch
from diffdrr.registration import PoseRegressor
from diffdrr.utils import resample
from torch.optim.lr_scheduler import LambdaLR
from torchvision.transforms.functional import center_crop

from .utils import XrayTransforms, get_4x4


def load_model(ckptpath, meta=False):
    """Load a pretrained pose regression model"""
    ckpt = torch.load(ckptpath, weights_only=False)
    config = ckpt["config"]

    model_state_dict = ckpt["model_state_dict"]
    model = PoseRegressor(
        model_name=config["model_name"],
        parameterization=config["parameterization"],
        convention=config["convention"],
        norm_layer=config["norm_layer"],
        height=config["height"],
    ).cuda()
    model.load_state_dict(model_state_dict)
    model.eval()

    if meta:
        return model, config, ckpt["date"]
    else:
        return model, config


def predict_pose(model, config, img, sdd, delx, dely, x0, y0, meta=False):
    # Resample the X-ray image to match the model's assumed intrinsics
    img, height, width = _resample_xray(img, sdd, delx, dely, x0, y0, config)
    height = min(height, width)
    img = center_crop(img, (height, height))

    # Resize the image and normalize pixel intensities
    transforms = XrayTransforms(config["height"])
    img = transforms(img).cuda()

    # Predict pose
    with torch.no_grad():
        init_pose = model(img)

    if meta:
        return init_pose, height
    else:
        return init_pose


def _resample_xray(img, sdd, delx, dely, x0, y0, config):
    """Resample the image to match the model's assumed intrinsics"""
    assert delx == dely, "Non-square pixels are not yet supported"

    model_height = config["height"]
    model_delx = config["delx"]

    _, _, height, width = img.shape
    subsample = min(height, width) / model_height
    new_delx = model_delx / subsample

    img = resample(img, sdd, delx, x0, y0, config["sdd"], new_delx, 0, 0)

    return img, height, width


def _correct_pose(pose, warp, volume, invert):
    if warp is None:
        return pose

    # Get the closest SE(3) transformation relating the CT to some reference frame
    T = get_4x4(warp, volume, invert).cuda()
    return pose.compose(T)


class WarmupCosineSchedule(LambdaLR):
    """
    Linear warmup and then cosine decay.
    Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
    Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
    If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.

    Copied from https://github.com/TalSchuster/pytorch-transformers/blob/64fff2a53977ac1caac32c960d2b01f16b7eb913/pytorch_transformers/optimization.py#L64-L81
    """

    def __init__(self, optimizer, warmup_steps, t_total, cycles=0.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(
            max(1, self.t_total - self.warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(self.cycles) * 2.0 * progress))
        )
