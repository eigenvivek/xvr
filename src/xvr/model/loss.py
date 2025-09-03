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
    ):
        super().__init__()
        self.imagesim = MultiscaleNormalizedCrossCorrelation2d([None, 9], [0.5, 0.5])
        self.diceloss = DiceLoss()
        self.geodesic = DoubleGeodesicSE3(sdd)
        self.weight_geo = weight_geo

    def forward(self, img, pose, pred_img, pred_pose):
        if img.shape[1] > 1 and pred_img.shape[1] > 1:  # Multi-channel       
            mask = img > 0
            pred_mask = pred_img > 0

            img = img.sum(dim=1, keepdim=True)
            pred_img = pred_img.sum(dim=1, keepdim=True)

            compute_dice = True
            
        mncc = self.imagesim(img, pred_img)
        rgeo, tgeo, dgeo = self.geodesic(pose, pred_pose)
        loss = 1 - mncc + self.weight_geo * dgeo
        
        if compute_dice:
            dice = self.diceloss(mask, pred_mask)
            loss += dice
        else:
            dice = None
        
        return loss, mncc, dgeo, rgeo, tgeo, dice



class DiceLoss(torch.nn.Module):
    def __init__(self, include_background=False, reduction="none"):
        super().__init__()
        self.dice = DiceMetric(include_background=include_background, reduction=reduction)

    def forward(self, img1, img2):
        return 1 - self.dice(img1, img2).nanmean(dim=1).nan_to_num()()