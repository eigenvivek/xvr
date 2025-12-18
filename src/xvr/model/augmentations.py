import kornia.augmentation as K
import torch

from ..utils import Standardize


def XrayAugmentations(
    p=0.333, max_crop=10, same_on_batch=False, transformation_matrix_mode="skip"
):
    return K.AugmentationSequential(
        Standardize(),
        K.RandomClahe(clip_limit=(1.0, 40.0), p=(2 * p)),
        K.RandomGamma(gamma=(0.7, 1.8), p=p),
        K.RandomBoxBlur(p=p),
        K.RandomGaussianNoise(std=0.01, p=p),
        K.RandomSharpness(p=p),
        K.RandomErasing(p=p),
        RandomCenterCrop(p=p, maxcrop=max_crop),
        keepdim=True,
        same_on_batch=same_on_batch,
        transformation_matrix_mode=transformation_matrix_mode,
    )


class RandomCenterCrop(K.IntensityAugmentationBase2D):
    """Simulate collimation."""

    def __init__(self, maxcrop: int, p: float = 0.5):
        super().__init__(p=p)
        self.maxcrop = maxcrop

    def apply_transform(
        self,
        input: torch.Tensor,
        params: dict[str, torch.Tensor],
        flags: dict[str, any],
        transform: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, _, H, W = input.shape
        crops = params["crop"].to(input.device).view(B, 1, 1)

        y = torch.arange(H, device=input.device).view(1, H, 1).expand(B, H, W)
        x = torch.arange(W, device=input.device).view(1, 1, W).expand(B, H, W)

        mask = (
            (y >= crops) & (y < H - crops) & (x >= crops) & (x < W - crops)
        ).unsqueeze(1)

        return torch.where(mask, input, torch.zeros_like(input))

    def generate_parameters(self, shape: tuple[int, ...]) -> dict[str, torch.Tensor]:
        B = shape[0]
        return {"crop": torch.randint(0, self.maxcrop + 1, (B,), device=self.device)}


class Clamp(torch.nn.Module):
    def __init__(self, mini, maxi):
        super().__init__()
        self.mini = mini
        self.maxi = maxi

    def forward(self, x):
        return x.clamp(self.mini, self.maxi)
