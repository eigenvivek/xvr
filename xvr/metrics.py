import torch
from diffdrr.drr import DRR
from diffdrr.metrics import DoubleGeodesicSE3
from diffdrr.pose import RigidTransform


class Evaluator:
    """
    Calculate four 2D/3D registration error metrics (all in mm).
    """
    
    def __init__(self, drr: DRR, fiducials: torch.Tensor):
        self.drr = drr
        self.fiducials = fiducials
        self.geodesic = DoubleGeodesicSE3(drr.detector.sdd)

    def __call__(self, true_pose: RigidTransform, pred_pose: RigidTransform):
        # Mean projection error (mPE)
        x = self.drr.perspective_projection(pred_pose, self.fiducials)
        y = self.drr.perspective_projection(true_pose, self.fiducials)
        mpe = (self.drr.detector.delx * (x - y)).norm(dim=-1).mean(dim=-1)
    
        # Mean reprojection error (mRPE)
        x = self.drr.inverse_projection(pred_pose, x)
        y = self.drr.inverse_projection(true_pose, y)
        mrpe = (x - y).norm(dim=-1).mean(dim=-1)
    
        # Mean target registration error (mTRE)
        x = pred_pose(self.fiducials)
        y = true_pose(self.fiducials)
        mtre = (x - y).norm(dim=-1).mean(dim=-1)
    
        # Double geodesic distance
        *_, dgeo = self.geodesic(true_pose, pred_pose)
    
        return torch.stack([mpe, mrpe, mtre, dgeo], dim=-1).squeeze().cpu().tolist()
