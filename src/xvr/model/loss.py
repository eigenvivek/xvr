import warnings

import torch
from diffdrr.metrics import DoubleGeodesicSE3, MultiscaleNormalizedCrossCorrelation2d
from monai.metrics import DiceMetric

warnings.filterwarnings("ignore", category=UserWarning, module="monai.metrics")


class PoseRegressionLoss(torch.nn.Module):
    def __init__(
        self,
        sdd: float,  # Source-to-detector distance (in mm)
        weight_geo: float,  # Balancing term for geodesic loss (in mm) and NCC [-1, 1]
        weight_dice: float,  # Balancing term for Dice loss and NCC
    ):
        super().__init__()
        self.imagesim = MultiscaleNormalizedCrossCorrelation2d([None, 9], [0.5, 0.5])
        self.diceloss = DiceLoss()
        self.geodesic = DoubleGeodesicSE3(sdd)
        self.weight_geo = weight_geo
        self.weight_dice = weight_dice

    def forward(self, img, mask, pose, pred_img, pred_mask, pred_pose):
        mncc = self.imagesim(img, pred_img)
        dice = self.diceloss(mask, pred_mask)
        rgeo, tgeo, dgeo = self.geodesic(pose, pred_pose)
        loss = 1 - mncc + self.weight_dice * dice + self.weight_geo * dgeo
        return loss, mncc, dgeo, rgeo, tgeo, dice


class DiceLoss(torch.nn.Module):
    def __init__(self, include_background=False, reduction="none"):
        super().__init__()
        self.dice = DiceMetric(
            include_background=include_background, reduction=reduction
        )

    def forward(self, img1, img2):
        return 1 - self.dice(img1, img2).nanmean(dim=1).nan_to_num()
