from random import randint, random

import kornia.augmentation as K
import torch
from torchvision.transforms.functional import center_crop, pad

from ..utils import Standardize


def XrayAugmentations(
    p=0.5, max_crop=10, same_on_batch=False, transformation_matrix_mode="skip"
):
    return K.AugmentationSequential(
        Standardize(),
        K.RandomEqualize(p=p / 3),
        K.RandomBrightness(brightness=0.2, p=p),
        K.RandomContrast(contrast=(0.8, 1.2), p=p),
        K.RandomGamma(gamma=(0.7, 1.3), p=p),
        K.RandomBoxBlur(p=p),
        K.RandomGaussianNoise(std=0.01, p=p),
        K.RandomInvert(p=p / 5),
        K.RandomAutoContrast(p=p / 2),
        K.RandomSharpness(p=p),
        K.RandomErasing(p=p),
        RandomCenterCrop(p=p, maxcrop=max_crop),
        keepdim=True,
        same_on_batch=same_on_batch,
        transformation_matrix_mode=transformation_matrix_mode,
    )


class RandomCenterCrop(torch.nn.Module):
    """Simulate collimation."""

    def __init__(self, p, maxcrop):
        super().__init__()
        self.p = p
        self.maxcrop = maxcrop

    def forward(self, x):
        if random() > self.p:
            return x
        *_, h, w = x.shape
        crop = randint(0, self.maxcrop)
        x = center_crop(x, [h - 2 * crop, w - 2 * crop])
        x = pad(x, [crop, crop, crop, crop])
        return x


class Clamp(torch.nn.Module):
    def __init__(self, mini, maxi):
        super().__init__()
        self.mini = mini
        self.maxi = maxi

    def forward(self, x):
        return x.clamp(self.mini, self.maxi)
