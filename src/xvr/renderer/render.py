from typing import Optional

import torch
from diffdrr.data import transform_hu_to_density
from diffdrr.drr import DRR
from diffdrr.pose import RigidTransform, convert


def render(
    drr: DRR,
    pose: RigidTransform,
    subject: Optional[dict] = None,
    contrast: float = 1.0,
    centerize: bool = True,
):
    """
    Render a batch of DRRs from a volume (and optional mask). If a mask is a provided, the rendered
    DRRs will be multi-channel (i.e., one channel for each structure in the volume's labelmap).
    """

    # Load 3D imaging data into memory and optionally move the pose to the volume's isocenter
    if isinstance(subject, dict):
        volume, mask, affinv, offset = load(
            subject, dtype=pose.matrix.dtype, device=pose.matrix.device
        )
    else:
        volume, mask, affinv = drr.volume, drr.mask, drr.affine_inverse
        offset = make_translation(drr.center)
    if centerize:
        pose = pose.compose(offset)

    # Get the source and target locations for every ray in voxel coordinates
    source, target = drr.detector(pose, None)
    img = (target - source).norm(dim=-1).unsqueeze(1)
    source, target = affinv(source), affinv(target)

    # Render a batch of DRRs
    tmp = transform_hu_to_density(volume, contrast)
    img = drr.renderer(tmp, source, target, img, mask=mask)
    img = drr.reshape_transform(img, batch_size=len(pose))

    # Create a foreground mask and collapse potentially multichannel images to a single DRR
    mask = img > 0
    img = img.sum(dim=1, keepdim=True)

    return img, mask, pose


def load(subject, dtype, device):
    # Load the volume and optional mask into memory
    data = subject["volume"]["data"].squeeze().to(dtype=dtype, device=device)
    if subject["mask"] is not None:
        mask = subject["mask"]["data"].data.squeeze().to(dtype=dtype, device=device)

    # Get the volume's isocenter and construct a translation to it
    affine = torch.from_numpy(subject["volume"]["affine"]).to(dtype=dtype)
    affine = RigidTransform(affine)

    center = (torch.tensor(data.shape)[None, None] - 1) / 2
    center = affine(center)[0].to(dtype=dtype, device=device)

    offset = make_translation(center)

    # Make the inverse affine
    affine = torch.from_numpy(subject["volume"]["affine"]).to(
        dtype=dtype, device=device
    )
    affinv = RigidTransform(affine.inverse())

    return data, mask, affinv, offset


def make_translation(xyz):
    rot = torch.zeros_like(xyz)
    return convert(rot, xyz, parameterization="euler_angles", convention="ZXY")
