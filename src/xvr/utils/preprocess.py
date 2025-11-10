import torch
from torchvision.transforms import Compose, Normalize, Resize


def XrayTransforms(height, width=None, mean=0.15, std=0.1):
    width = height if width is None else width
    return Compose(
        [
            Standardize(),
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
