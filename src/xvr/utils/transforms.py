import warnings
from pathlib import Path

import numpy as np
import torch
from diffdrr.pose import RigidTransform, make_matrix
from jaxtyping import Float
from SimpleITK import ReadTransform
from torchio import ScalarImage
from torchio.data.io import get_sitk_metadata_from_ras_affine


def read_rigid_transform(mat: str | Path, img: str | Path, invert: bool = False) -> RigidTransform:
    """Get the rigid or affine matrix for warping img_warped -> img."""

    M = _read_transform(mat, invert)

    img = ScalarImage(img)
    D = _read_direction(img)
    T = _read_isocenter(img)

    T = T @ D @ M @ np.linalg.inv(D)
    T = torch.from_numpy(T).to(torch.float32)
    T = RigidTransform(T)

    return _get_nearest_rigid_transform(T)


def _read_transform(mat: str | Path, invert: bool) -> Float[np.ndarray, "4 4"]:
    xform = ReadTransform(mat)
    if invert:
        xform = xform.GetInverse()

    Rt = np.array(xform.GetParameters())
    R = Rt[:9].reshape(3, 3)
    t = Rt[9:]
    c = np.array(xform.GetFixedParameters())
    t = -R @ c + t + c

    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t

    return M


def _read_direction(img: ScalarImage) -> Float[np.ndarray, "4 4"]:
    D = np.eye(4)
    if img.orientation == ("L", "P", "S"):
        D[:3, :3] = np.array(img.direction).reshape(3, 3)
    elif img.orientation == ("R", "A", "S"):
        *_, direction = get_sitk_metadata_from_ras_affine(img.affine)
        D[:3, :3] = np.array(direction).reshape(3, 3)
    else:
        warnings.warn(
            f"Unrecognized orientation {img.orientation}; assuming LPS+ directions. If the corrected pose is completely wrong, check here first."
        )
        D[:3, :3] = np.array(img.direction).reshape(3, 3)
    return D


def _read_isocenter(img: ScalarImage) -> Float[np.ndarray, "4 4"]:
    T = np.eye(4)
    T[:3, 3] = -np.array(img.get_center())
    return T


def _get_nearest_rigid_transform(T: RigidTransform) -> RigidTransform:
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
