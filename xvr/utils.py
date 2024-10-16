from random import randint, random

import kornia.augmentation as K
import numpy as np
import torch
import torchio
from ants import read_transform
from diffdrr.data import transform_hu_to_density
from diffdrr.pose import RigidTransform, convert, make_matrix
from torchvision.transforms import Compose, Normalize, Resize
from torchvision.transforms.functional import center_crop, pad


def get_random_pose(config):
    """Generate a batch of random poses in SE(3) using specified ranges."""
    alpha = uniform(config["alphamin"], config["alphamax"], config["batch_size"])
    beta = uniform(config["betamin"], config["betamax"], config["batch_size"])
    gamma = uniform(config["gammamin"], config["gammamax"], config["batch_size"])
    tx = uniform(config["txmin"], config["txmax"], config["batch_size"])
    ty = uniform(config["tymin"], config["tymax"], config["batch_size"])
    tz = uniform(config["tzmin"], config["tzmax"], config["batch_size"])
    rot = torch.concat([alpha, beta, gamma], dim=1) / 180 * torch.pi
    xyz = torch.concat([tx, ty, tz], dim=1)
    return convert(rot, xyz, parameterization="euler_angles", convention="ZXY")


def uniform(low, high, n):
    return (high - low) * torch.rand(n, 1) + low


def render(drr, pose, volume, contrast):
    """Render a batch of DRRs given a pose and volume."""
    source, target = drr.detector(pose, None)
    img = drr.render(
        transform_hu_to_density(volume, contrast),
        source,
        target,
    )
    img = drr.reshape_transform(img, batch_size=len(pose))
    return img, source, target


class Standardize(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (x - x.min()) / (x.max() - x.min() + 1e-6)


def XrayTransforms(height, width=None, mean=0.15, std=0.1):
    width = height if width is None else width
    return Compose(
        [
            Standardize(),
            Resize((height, width)),
            Normalize([mean], [std]),
        ]
    )


def XrayAugmentations(p=0.5):
    class RandomCenterCrop(torch.nn.Module):
        def __init__(self, p, maxcrop):
            super().__init__()
            self.p = p
            self.maxcrop = maxcrop

        def forward(self, x):
            if random() > p:
                return x
            *_, h, w = x.shape
            crop = randint(0, self.maxcrop)
            x = center_crop(x, [h - 2 * crop, w - 2 * crop])
            x = pad(x, [crop, crop, crop, crop])
            return x

    return K.AugmentationSequential(
        Standardize(),
        K.RandomAutoContrast(p=p),
        K.RandomBoxBlur(p=p),
        K.RandomEqualize(p=p),
        K.RandomGaussianNoise(std=0.01, p=p),
        K.RandomInvert(p=p),
        K.RandomLinearIllumination(p=p),
        K.RandomSharpness(p=p),
        K.RandomErasing(p=p),
        RandomCenterCrop(p=p, maxcrop=10),
        same_on_batch=False,
        transformation_matrix_mode="skip",
    )


def get_4x4(mat, img, invert=False):
    """Get the rigid or affine matrix for warping img_warped -> img."""
    img = torchio.ScalarImage(img)

    transform = read_transform(mat)
    if invert:
        transform = transform.invert()
    R = transform.parameters[:9].reshape(3, 3)
    t = transform.parameters[9:]
    c = transform.fixed_parameters
    global_t = -R @ c + t + c

    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = global_t

    D = np.eye(4)
    dirs = np.array(img.direction).reshape(3, 3)
    D[:3, :3] = dirs

    Tinv = np.eye(4)
    Tinv[:3, 3] = -np.array(img.get_center())

    T = Tinv @ D @ M @ np.linalg.inv(D)
    T = torch.from_numpy(T).to(torch.float32)
    T = RigidTransform(T)

    return project_onto_SO3(T)


def project_onto_SO3(T: RigidTransform):
    """Convert the upper 3x3 to a matrix in SO(3) (i.e., unitary with det=+1)."""
    M = T.matrix[0]
    A = M[:3, :3]
    At = M[:3, 3]
    U, S, V = A.svd()
    t = A.inverse() @ At
    S = torch.ones_like(S)
    S[..., -1] = (U @ V.mT).det()
    R = torch.einsum("ij, j, jk -> ik", U, S, V.mT)
    t = R @ t
    return RigidTransform(make_matrix(R[None], t[None]))
