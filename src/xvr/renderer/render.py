import torch
from diffdrr.data import transform_hu_to_density
from diffdrr.drr import DRR
from diffdrr.pose import RigidTransform, convert
from torchio import ScalarImage, LabelMap


def render(
    drr: DRR,
    pose: RigidTransform,
    contrast: float,
    volume: ScalarImage,
    mask: LabelMap = None,
    centerize: bool = True,
):
    """
    Render a batch of DRRs from a volume (and optional mask). If a mask is a provided, the rendered
    DRRs wqill be multi-channel (i.e., one channel for each structure in the volume's labelmap).
    """

    # Load 3D imaging data into memory and optionally move the pose to the volume's isocenter
    volume, mask, affinv, offset = load(
        volume, mask, dtype=pose.matrix.dtype, device=pose.matrix.device
    )
    if centerize:
        pose = offset.compose(pose)

    # Get the source and target locations for every ray in voxel coordinates
    source, target = drr.detector(pose, None)
    img = (target - source).norm(dim=-1).unsqueeze(1)
    source, target = affinv(source), affinv(target)

    # Render a batch of DRRs
    tmp = transform_hu_to_density(volume, contrast)
    img = drr.renderer(tmp, source, target, img, mask=mask)
    img = drr.reshape_transform(img, batch_size=len(pose))

    return img, pose


def load(volume, mask, dtype, device):
    # Load the volume and optional mask into memory
    data = volume.data.squeeze().to(dtype=dtype, device=device)
    if mask is not None:
        mask = mask.data.squeeze().to(dtype=dtype, device=device)

    # Save the inverse affine for moving from world to voxel coordinates
    affine = torch.from_numpy(volume.affine).to(dtype=dtype, device=device)
    affine = RigidTransform(affine)
    affinv = affine.inverse()

    # Make a transform from the origin in world coordinates to the volume's isocenter
    center = torch.tensor(volume.get_center())[None].to(dtype=dtype, device=device)
    offset = make_translation(center)

    return data, mask, affinv, offset


def make_translation(xyz):
    rot = torch.zeros_like(xyz)
    return convert(rot, xyz, parameterization="euler_angles", convention="ZXY")
