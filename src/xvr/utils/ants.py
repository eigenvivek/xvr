import warnings

import ants
import numpy as np
import torch
import torchio
from diffdrr.pose import RigidTransform, make_matrix
from torchio.data.io import get_sitk_metadata_from_ras_affine


def get_4x4(mat, img, invert=False):
    """Get the rigid or affine matrix for warping img_warped -> img."""
    img = torchio.ScalarImage(img)

    transform = ants.read_transform(mat)
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
    if img.orientation == ("L", "P", "S"):
        D[:3, :3] = np.array(img.direction).reshape(3, 3)
    elif img.orientation == ("R", "A", "S"):
        D[:3, :3] = direction(img)
    else:
        warnings.warn(
            f"Unrecognized orientation {img.orientation}; assuming LPS+ directions. If the corrected pose is completely wrong, check here first."
        )
        D[:3, :3] = np.array(img.direction).reshape(3, 3)

    Tinv = np.eye(4)
    Tinv[:3, 3] = -np.array(img.get_center())

    T = Tinv @ D @ M @ np.linalg.inv(D)
    T = torch.from_numpy(T).to(torch.float32)
    T = RigidTransform(T)

    return project_onto_SO3(T)


def ants_rigid_register(fix_filename, mov_filename, savepath):
    """Rigidly register the new volume to the template with ANTs."""
    img_fix = ants.image_read(fix_filename)
    img_mov = ants.image_read(mov_filename)
    result = ants.registration(
        img_fix,
        img_mov,
        type_of_transform="Rigid",
        aff_random_sampling_rate=0.666,
        aff_iterations=(200, 200, 50),
        aff_shrink_factors=(6, 4, 2),
        aff_smoothing_sigmas=(3, 2, 1),
    )
    transform = ants.read_transform(result["fwdtransforms"][0])
    ants.write_transform(transform, savepath)


def direction(img: torchio.ScalarImage):
    """Volume directions in RAS space (comport with ANTS convention)."""
    *_, direction = get_sitk_metadata_from_ras_affine(img.affine)
    return np.array(direction).reshape(3, 3)


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
