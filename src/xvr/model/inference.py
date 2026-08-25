import torch
from diffdrr.pose import RigidTransform, convert
from diffdrr.utils import resample
from jaxtyping import Float
from torchvision.transforms.functional import center_crop

from ..utils import XrayTransforms, read_rigid_transform
from .network import PoseRegressor


def predict_pose(
    model: PoseRegressor,
    config: dict,
    img: Float[torch.Tensor, "1 1 H W"],
    sdd: float,
    delx: float,
    dely: float,
    x0: float,
    y0: float,
) -> RigidTransform:
    """Regress the camera pose of an X-ray with a trained pose regressor.

    The X-ray is resampled to the intrinsics the model was trained on, center
    cropped, and normalized before being passed through the network. This is how
    a checkpoint is used as an initializer for iterative registration.

    Args:
        model: Trained `PoseRegressor`.
        config: The checkpoint's saved training config, supplying the model's
            assumed `sdd`, `height`, and `delx`.
        img: X-ray image tensor of shape ``(1, 1, H, W)``.
        sdd: Source-to-detector distance of the X-ray, in millimeters.
        delx: Pixel spacing of the X-ray along the x-axis, in millimeters.
        dely: Pixel spacing of the X-ray along the y-axis, in millimeters.
            Must equal `delx`; non-square pixels are not supported.
        x0: Detector origin offset along the x-axis, in millimeters.
        y0: Detector origin offset along the y-axis, in millimeters.

    Returns:
        The predicted camera pose.
    """
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
    T = read_rigid_transform(warp, volume, invert).cuda()
    return pose.compose(T)


def _construct_antipode(pose: RigidTransform) -> RigidTransform:
    rot, xyz = pose.convert("euler_angles", "ZXY")
    rot[..., 0:2] *= -1
    rot[..., 0] += torch.pi
    return convert(rot, xyz, parameterization="euler_angles", convention="ZXY")
