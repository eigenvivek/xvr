import torch
from jaxtyping import Float
from nanodrr.camera import invert_rt_inv, make_k_inv, make_rt_inv, resample
from nanodrr.data import Subject
from torchvision.transforms.functional import center_crop

from ..io import Intrinsics
from ..utils import XrayTransforms, get_4x4


def predict_pose(
    model: torch.nn.Module,
    img: Float[torch.Tensor, "1 C H W"],
    img_intrinsics: Intrinsics,
    model_sdd: float,
    model_delx: float,
    model_height: int,
) -> Float[torch.Tensor, "1 4 4"]:
    """Predict the pose of a X-rays ."""
    img = _resample_xray(img, img_intrinsics, model_sdd, model_delx, model_height)
    init_pose = model(img)
    return init_pose


def _resample_xray(
    img: Float[torch.Tensor, "1 C Hi Wi"],
    img_intrinsics: Intrinsics,
    model_sdd: float,
    model_delx: float,
    model_height: int,
) -> Float[torch.Tensor, "1 C Hm Wm"]:
    *_, height, width = img.shape
    side_length = min(height, width)
    new_delx = model_delx * (model_height / side_length)

    k_inv_old = make_k_inv(
        **img_intrinsics,
        height=height,
        width=width,
        dtype=img.dtype,
        device=img.device,
    )
    k_inv_new = make_k_inv(
        *(model_sdd, new_delx, new_delx, 0.0, 0.0),
        height,
        width,
        dtype=img.dtype,
        device=img.device,
    )
    img = resample(img, k_inv_old, k_inv_new)

    img = center_crop(img, (side_length, side_length))
    transforms = XrayTransforms(model_height).to(img)
    return transforms(img)


def _correct_pose(pose, warp, volume, invert):
    if warp is None:
        return pose

    # Get the closest SE(3) transformation relating the CT to some reference frame
    T = get_4x4(warp, volume, invert).cuda()
    return pose.compose(T)


def _construct_antipode(
    pose: Float[torch.Tensor, "B 4 4"],
    orientation: str | None = "AP",
    subject: Subject | None = None,
) -> Float[torch.Tensor, "B 4 4"]:
    isocenter = subject.isocenter if subject is not None else None
    rot, xyz = invert_rt_inv(pose, orientation, isocenter)
    rot[..., 0:2] *= -1
    rot[..., 0] += torch.pi
    return make_rt_inv(rot, xyz, orientation, isocenter)
