import torch
from diffdrr.data import transform_hu_to_density
from diffdrr.drr import DRR
from diffdrr.pose import RigidTransform, convert
from torchio import ScalarImage


def render(
    drr: DRR,
    pose: RigidTransform,
    subject: ScalarImage,
    contrast: float,
    centerize: bool = True,
):
    """
    Render a batch of DRRs from a subject. Data synthesis engine for training
    a PoseRegression model.
    """

    volume, affinv, offset = load(
        subject, dtype=pose.matrix.dtype, device=pose.matrix.device
    )

    if centerize:
        # pose = pose.compose(offset)
        pose = offset.compose(pose)

    source, target = drr.detector(pose, None)
    img = (target - source).norm(dim=-1).unsqueeze(1)
    source, target = affinv(source), affinv(target)

    tmp = transform_hu_to_density(volume, contrast)
    img = drr.renderer(tmp, source, target, img)
    img = drr.reshape_transform(img, batch_size=len(pose))

    return img, pose


def load(subject, dtype, device):
    volume = subject.data.squeeze().to(dtype=dtype, device=device)
    affine = torch.from_numpy(subject.affine).to(dtype=dtype, device=device)

    affine = RigidTransform(affine)
    affinv = affine.inverse()

    center = torch.tensor(subject.get_center())[None].to(dtype=dtype, device=device)
    offset = make_translation(center)

    return volume, affinv, offset


def make_translation(xyz):
    rot = torch.zeros_like(xyz)
    return convert(rot, xyz, parameterization="euler_angles", convention="ZXY")
