import torch
from diffdrr.metrics import DoubleGeodesicSE3, MultiscaleNormalizedCrossCorrelation2d


class PoseRegressionLoss(torch.nn.Module):
    def __init__(
        self, 
        sdd: float,  # Source-to-detector distance (in mm)
        weight_geo: float,  # Balancing term for geodesic loss (in mm) and NCC [-1, 1]
    ):
        super().__init__()
        self.imagesim = MultiscaleNormalizedCrossCorrelation2d([None, 9], [0.5, 0.5])
        self.geodesic = DoubleGeodesicSE3(sdd)
        self.weight_geo = weight_geo

    def forward(self, img, pose, pred_img, pred_pose):
        mncc = self.imagesim(img, pred_img)
        rgeo, tgeo, dgeo = self.geodesic(pose, pred_pose)
        loss = 1 - mncc + self.weight_geo * dgeo
        return loss, mncc, dgeo, rgeo, tgeo
