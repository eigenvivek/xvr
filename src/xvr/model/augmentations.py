from random import randint, random

import kornia.augmentation as K
import torch
from torchvision.transforms.functional import center_crop, pad

from ..utils import Standardize


def XrayAugmentations(p=0.5):
    class RandomCenterCrop(torch.nn.Module):
        def __init__(self, p, maxcrop):
            super().__init__()
            self.p = p
            self.maxcrop = maxcrop

        def forward(self, x):
            if random() > p:
                return x
            *_, h, w = x.shape
            crop = randint(0, self.maxcrop)
            x = center_crop(x, [h - 2 * crop, w - 2 * crop])
            x = pad(x, [crop, crop, crop, crop])
            return x

    return K.AugmentationSequential(
        Standardize(),
        K.RandomAutoContrast(p=p),
        K.RandomBoxBlur(p=p),
        K.RandomEqualize(p=p),
        K.RandomGaussianNoise(std=0.01, p=p),
        K.RandomInvert(p=p),
        K.RandomLinearIllumination(p=p),
        K.RandomSharpness(p=p),
        K.RandomErasing(p=p),
        RandomCenterCrop(p=p, maxcrop=10),
        K.RandomSaltAndPepperNoise(p=p),
        Clamp(0, 1),
        same_on_batch=False,
        transformation_matrix_mode="skip",
    )


class Clamp(torch.nn.Module):
    def __init__(self, mini, maxi):
        super().__init__()
        self.mini = mini
        self.maxi = maxi

    def forward(self, x):
        return x.clamp(self.mini, self.maxi)
