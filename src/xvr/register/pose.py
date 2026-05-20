import torch
from diffdrr.pose import RigidTransform, convert


class Pose(torch.nn.Module):
    """Optimizable pose defined on the global chart."""

    def __init__(
        self,
        init: RigidTransform,
        parameterization: str,
        convention: str | None,
    ):
        super().__init__()
        rot, xyz = init.convert(parameterization, convention)
        self._rot = torch.nn.Parameter(rot)
        self._xyz = torch.nn.Parameter(xyz)
        self.parameterization = parameterization
        self.convention = convention

    def forward(self) -> RigidTransform:
        return convert(
            self._rot,
            self._xyz,
            parameterization=self.parameterization,
            convention=self.convention,
        )
