from dataclasses import dataclass

import torch
from diffdrr.metrics import DoubleGeodesicSE3

from ..register.losses import GradientMultiscaleNormalizedCrossCorrelation2d


@dataclass
class PoseRegressionMetrics:
    loss: torch.Tensor
    mncc: torch.Tensor
    dgeo: torch.Tensor
    rgeo: torch.Tensor
    tgeo: torch.Tensor
    dice: torch.Tensor
    haus: torch.Tensor


class PoseRegressionLoss(torch.nn.Module):
    def __init__(
        self,
        sdd: float,
        weight_ncc: float = 1e0,
        weight_geo: float = 1e-2,
        weight_dice: float = 1e0,
        weight_haus: float = 1e-1,
        device: str = "cuda",
    ):
        super().__init__()
        self.imagesim = GradientMultiscaleNormalizedCrossCorrelation2d().to(device)
        self.diceloss = DiceLoss()
        self.geodesic = DoubleGeodesicSE3(sdd)
        self.hausloss = HausdorffLoss()

        self.weight_ncc = weight_ncc
        self.weight_geo = weight_geo
        self.weight_dice = weight_dice
        self.weight_haus = weight_haus

    def forward(self, img, mask, pose, pred_img, pred_mask, pred_pose):
        mncc = self.imagesim(img, pred_img)
        dice = self.diceloss(mask, pred_mask)
        rgeo, tgeo, dgeo = self.geodesic(pose, pred_pose)
        haus = self.hausloss(mask, pred_mask)

        loss = (
            self.weight_ncc * (1 - mncc)
            + self.weight_dice * dice
            + self.weight_geo * dgeo
            + self.weight_haus * haus
        )

        return PoseRegressionMetrics(
            loss=loss,
            mncc=mncc,
            dgeo=dgeo,
            rgeo=rgeo,
            tgeo=tgeo,
            dice=dice,
            haus=haus,
        )


class DiceLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = DiceMetric()

    def forward(self, img1, img2):
        return 1 - self.dice(img1, img2).nanmean(dim=1).nan_to_num()


class DiceMetric(torch.nn.Module):
    def forward(self, y_pred, y_true):
        """
        Compute 2D Dice coefficient between to multi-channel labelmaps.
        Assumes the first channel in each image is background.

        Equivalent to monai.metrics.DiceMetric(include_background=False, reduction="none")
        """
        y_pred = y_pred.view(y_pred.shape[0], y_pred.shape[1], -1)
        y_true = y_true.view(y_true.shape[0], y_true.shape[1], -1)

        intersection = (y_pred * y_true).sum(dim=2)
        pred_sum = y_pred.sum(dim=2)
        true_sum = y_true.sum(dim=2)

        dice = (2.0 * intersection) / (pred_sum + true_sum)
        return dice[:, 1:]


class HausdorffLoss(torch.nn.Module):
    """
    Differentiable Hausdorff loss via the distance-transform formulation of
    Karimi & Salcudean (2019). The distance transform is approximated by
    iterated morphological erosion (max-pool on the inverted mask) and
    treated as a fixed spatial weight; gradients flow through
    (y_pred - y_true)^2.

        L = mean( (y_pred - y_true)^2 * (DT(y_true)^a + DT(y_pred)^a) )
    """

    def __init__(self, alpha: float = 2.0, kernel_size: int = 3, num_iters: int = 10):
        super().__init__()
        self.alpha = alpha
        self.kernel_size = kernel_size
        self.num_iters = num_iters

    @torch.no_grad()
    def _distance_transform(self, mask: torch.Tensor) -> torch.Tensor:
        pad = self.kernel_size // 2
        dt = torch.zeros_like(mask)
        eroded = mask
        for _ in range(self.num_iters):
            eroded = -torch.nn.functional.max_pool2d(
                -eroded, self.kernel_size, stride=1, padding=pad
            )
            dt = dt + eroded
        return dt

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        y_true = y_true.float()[:, 1:]
        y_pred = y_pred.float()[:, 1:]
        B, C = y_true.shape[:2]
        y_true = y_true.reshape(B * C, 1, *y_true.shape[2:])
        y_pred = y_pred.reshape(B * C, 1, *y_pred.shape[2:])

        dt_true = self._distance_transform(1 - y_true)
        dt_pred = self._distance_transform(1 - y_pred.detach())

        weight = dt_true**self.alpha + dt_pred**self.alpha
        diff = (y_pred - y_true) ** 2
        loss = (diff * weight).flatten(1).mean(dim=1)
        return loss.view(B, C).mean(dim=1)
