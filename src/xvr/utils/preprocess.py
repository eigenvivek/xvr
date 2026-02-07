import torch
from torchvision.transforms import Compose, Normalize, Resize


def XrayTransforms(
    height: int,
    width: int = None,
    mean: float = 0.15,
    std: float = 0.1,
    equalize: bool = False,
):
    width = height if width is None else width
    return Compose(
        [
            Standardize(),
            Equalize() if equalize else Identity(),
            Resize((height, width)),
            Normalize([mean], [std]),
        ]
    )


class Standardize(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return (x - x.min()) / (x.max() - x.min() + self.eps)


class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Equalize(torch.nn.Module):
    def __init__(self, n_bins: int = 256, tau: float = 0.01, eps: float = 1e-10):
        super().__init__()
        self.n_bins = n_bins
        self.tau = tau
        self.eps = eps

    def forward(self, x):
        B, _, H, W = x.shape

        bins = torch.linspace(0, 1, self.n_bins, device=x.device)[None, None]
        diff = x.view(B, -1, 1) - bins
        weights = (-diff.square() / (2 * self.tau**2)).exp()

        histogram = weights.sum(dim=1)
        histogram = histogram / (histogram.sum(dim=1, keepdim=True) + self.eps)

        cdf = torch.cumsum(histogram, dim=1)
        cdf_min = cdf[:, 0:1]
        cdf_normalized = (cdf - cdf_min) / (1 - cdf_min + self.eps)

        weights_norm = weights / (weights.sum(dim=-1, keepdim=True) + self.eps)
        equalized_flat = (weights_norm * cdf_normalized[:, None]).sum(dim=-1)

        equalized_channel = equalized_flat.view(B, 1, H, W)

        return equalized_channel
