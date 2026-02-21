from pathlib import Path

import torch
from torchio import ScalarImage

from ...utils import get_4x4


def initialize_coordinate_frame(warp, img, invert):
    if warp is not None:
        return get_4x4(warp, img, invert).cuda().matrix
    if Path(img).is_file():
        isocenter = ScalarImage(img).get_center()
        transform = torch.eye(4, dtype=torch.float32)[None]
        transform[:, :3, 3] = isocenter
        return transform.cuda()
    return None
