from typing import NamedTuple

import torch

from nanodrr.metrics import DoubleGeodesicSE3, MultiscaleNormalizedCrossCorrelation2d


class Metrics(NamedTuple):
    mncc: torch.Tensor
    dgeo: torch.Tensor
    rgeo: torch.Tensor
    tgeo: torch.Tensor
    dice: torch.Tensor


class PoseRegressionLoss(torch.nn.Module):
    def __init__(
        self,
        sdd: float,  # Source-to-detector distance (in mm)
        weight_ncc: float = 1e0,  # Weight for mNCC loss
        weight_geo: float = 1e-2,  # Weight for geodesic distance
        weight_dice: float = 1e0,  # Weight for Dice loss
    ):
        super().__init__()

        self.imagesim = MultiscaleNormalizedCrossCorrelation2d([None, 9], [0.5, 0.5])
        self.diceloss = DiceLoss()
        self.geodesic = DoubleGeodesicSE3(sdd)

        self.weight_ncc = weight_ncc
        self.weight_geo = weight_geo
        self.weight_dice = weight_dice

    def forward(self, img, mask, pose, pred_img, pred_mask, pred_pose):
        # Per-image losses
        mncc = self.imagesim(img, pred_img)
        dice = self.diceloss(mask, pred_mask)
        rgeo, tgeo, dgeo = self.geodesic(pose, pred_pose)
        loss = (
            self.weight_ncc * (1 - mncc)
            + self.weight_dice * dice
            + self.weight_geo * dgeo
        )

        metrics = Metrics(mncc, dgeo, rgeo, tgeo, dice)
        return loss, metrics


class DiceLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = DiceMetric()

    def forward(self, img1, img2):
        return 1 - self.dice(img1, img2).nanmean(dim=1).nan_to_num()


class DiceMetric(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        """
        Compute 2D Dice coefficient between to multi-channel labelmaps.
        Assumes the first channel in each image is background.

        Equivalent to monai.metrics.DiceMetric(include_background=False, reduction="none")
        """

        # Flatten spatial dimensions
        y_pred = y_pred.view(y_pred.shape[0], y_pred.shape[1], -1)  # (B, C, H*W)
        y_true = y_true.view(y_true.shape[0], y_true.shape[1], -1)  # (B, C, H*W)

        # Compute intersection and union
        intersection = (y_pred * y_true).sum(dim=2)  # (B, C)
        pred_sum = y_pred.sum(dim=2)  # (B, C)
        true_sum = y_true.sum(dim=2)  # (B, C)

        # Compute Dice coefficient
        dice = (2.0 * intersection) / (pred_sum + true_sum)

        # Exclude background (assume background is channel 0)
        dice = dice[:, 1:]

        return dice
