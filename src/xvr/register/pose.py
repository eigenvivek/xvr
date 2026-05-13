import torch
from diffdrr.pose import RigidTransform, convert


class Pose(torch.nn.Module):
    """Optimizable pose defined on the global chart."""

    def __init__(self, init: RigidTransform):
        super().__init__()
        rot, xyz = init.convert("se3_log_map")
        self._rot = torch.nn.Parameter(rot)
        self._xyz = torch.nn.Parameter(xyz)

    def forward(self) -> RigidTransform:
        return convert(self._rot, self._xyz, parameterization="se3_log_map")
