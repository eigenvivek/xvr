import torch
from diffdrr.drr import DRR
from diffdrr.pose import RigidTransform, convert


class Registration(torch.nn.Module):
    def __init__(self, drr: DRR, pose: RigidTransform):
        super().__init__()
        self.drr = drr
        self.T = pose
        self._rot = torch.nn.Parameter(1e-8 * torch.randn((1, 3)))
        self._xyz = torch.nn.Parameter(1e-8 * torch.randn((1, 3)))

    @property
    def height(self) -> int:
        return self.drr.detector.height

    @property
    def width(self) -> int:
        return self.drr.detector.width

    @property
    def tau(self) -> RigidTransform:
        return convert(self._rot, self._xyz, parameterization="se3_log_map")

    @property
    def pose(self) -> RigidTransform:
        return self.tau.compose(self.T)

    def forward(self, **kwargs):
        return self.drr(self.pose, **kwargs)

    def rescale_(self, scale: float):
        self.drr.rescale_detector_(scale)


class LocalRegistration(Registration):
    @torch.no_grad()
    def retract(self):
        self.T = self.tau.compose(self.T)
        self._rot.copy_(1e-8 * torch.randn((1, 3)))
        self._xyz.copy_(1e-8 * torch.randn((1, 3)))


class GlobalRegistration(Registration):
    pass
