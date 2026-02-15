import torch
import torch.nn as nn
import torch.nn.functional as F


class XrayTransforms(nn.Module):
    def __init__(
        self,
        height: int,
        width: int = None,
        mean: float = 0.15,
        std: float = 0.1,
        equalize: bool = False,
    ):
        super().__init__()
        self.height = height
        self.width = height if width is None else width
        self.equalize = equalize

        self.standardize = Standardize()
        self.equalize_fn = Equalize() if equalize else None
        self.normalize = Normalize([mean], [std])

    def forward(self, x):
        x = self.standardize(x)
        if self.equalize_fn is not None:
            x = self.equalize_fn(x)
        x = F.interpolate(
            x, size=(self.height, self.width), mode="bilinear", align_corners=False
        )
        x = self.normalize(x)
        return x


class Standardize(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return (x - x.min()) / (x.max() - x.min() + self.eps)


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(-1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(-1, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


class Equalize(nn.Module):
    def __init__(self, n_bins: int = 256, tau: float = 0.01, eps: float = 1e-10):
        super().__init__()
        self.tau = tau
        self.eps = eps
        self.register_buffer("bins", torch.linspace(0, 1, n_bins)[None, None])

    def forward(self, x):
        B, _, H, W = x.shape

        diff = x.view(B, -1, 1) - self.bins
        weights = (-diff.square() / (2 * self.tau**2)).exp()

        histogram = weights.sum(dim=1)
        histogram = histogram / (histogram.sum(dim=1, keepdim=True) + self.eps)

        cdf = torch.cumsum(histogram, dim=1)
        cdf_min = cdf[:, 0:1]
        cdf_normalized = (cdf - cdf_min) / (1 - cdf_min + self.eps)

        weights_norm = weights / (weights.sum(dim=-1, keepdim=True) + self.eps)
        equalized_flat = (weights_norm * cdf_normalized[:, None]).sum(dim=-1)

        return equalized_flat.view(B, 1, H, W)
