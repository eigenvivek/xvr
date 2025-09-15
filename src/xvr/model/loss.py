import torch
from diffdrr.metrics import DoubleGeodesicSE3, MultiscaleNormalizedCrossCorrelation2d


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


class DiceMetric(torch.nn.Module):
    def __init__(self, include_background=False, reduction="none", smooth=1e-6):
        super().__init__()
        self.include_background = include_background
        self.reduction = reduction
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        """
        Compute 2D Dice coefficient.

        Args:
            y_pred: Predicted tensor of shape (B, C, H, W) - probabilities or logits
            y_true: Ground truth tensor of shape (B, C, H, W) - one-hot encoded

        Returns:
            Dice coefficients of shape (B, C) when reduction="none"
        """
        # Convert predictions to probabilities if needed (sigmoid/softmax)
        if y_pred.dtype != torch.bool:
            y_pred = torch.sigmoid(y_pred)  # Assuming binary/multi-label case

        # Flatten spatial dimensions
        y_pred = y_pred.view(y_pred.shape[0], y_pred.shape[1], -1)  # (B, C, H*W)
        y_true = y_true.view(y_true.shape[0], y_true.shape[1], -1)  # (B, C, H*W)

        # Compute intersection and union
        intersection = (y_pred * y_true).sum(dim=2)  # (B, C)
        pred_sum = y_pred.sum(dim=2)  # (B, C)
        true_sum = y_true.sum(dim=2)  # (B, C)

        # Compute Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (pred_sum + true_sum + self.smooth)

        # Exclude background if specified (assume background is channel 0)
        if not self.include_background:
            dice = dice[:, 1:]  # Remove first channel

        # Apply reduction
        if self.reduction == "none":
            return dice
        elif self.reduction == "mean":
            return dice.mean()
        elif self.reduction == "sum":
            return dice.sum()
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")
