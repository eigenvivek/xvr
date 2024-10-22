import torch
from diffdrr.registration import PoseRegressor
from diffdrr.utils import resample
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
