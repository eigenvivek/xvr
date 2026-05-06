import torch
from nanodrr.metrics import (
    GradientNormalizedCrossCorrelation2d,
    MultiscaleNormalizedCrossCorrelation2d,
)


class GradientMultiscaleNormalizedCrossCorrelation2d(torch.nn.Module):
    def __init__(
        self,
        mncc_patch_size: int = 9,
        gncc_patch_size: int = 11,
        sigma: float = 0.0,
        beta: float = 0.5,
    ):
        """A weighted average of gradient and multiscale NCC."""
        super().__init__()
        self.mncc = MultiscaleNormalizedCrossCorrelation2d([None, mncc_patch_size], [0.5, 0.5])
        self.gncc = GradientNormalizedCrossCorrelation2d(gncc_patch_size, sigma)
        self.beta = beta

    def forward(self, true, pred):
        return self.beta * self.mncc(true, pred) + (1 - self.beta) * self.gncc(true, pred)
