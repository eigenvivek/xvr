from ...utils import get_4x4


def initialize_coordinate_frame(warp, img, invert):
    if warp is None:
        return None
    return get_4x4(warp, img, invert).cuda().matrix
