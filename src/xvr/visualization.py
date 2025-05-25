import matplotlib.pyplot as plt
import torch
from diffdrr.visualization import plot_drr

from .metrics import Evaluator
from .utils import XrayTransforms


def plot_registration(drr, fiducials, gt, pred_pose, true_pose):
    # Get the registration error
    evaluator = Evaluator(drr, fiducials)
    mtre = evaluator(true_pose, pred_pose)[2]

    # Compute true and predicted DRRs and fiducials
    with torch.no_grad():
        pred_pts = drr.perspective_projection(pred_pose, fiducials).cpu().squeeze()
        true_pts = drr.perspective_projection(true_pose, fiducials).cpu().squeeze()
        pred_img = drr(pred_pose).cpu()
        true_img = drr(true_pose).cpu()
        error = (true_img - pred_img)
    
    xt = XrayTransforms(drr.detector.height, drr.detector.width)
    gt = xt(gt)
    pred_img = xt(pred_img)

    # Plot the fiducials
    axs = plot_drr(torch.concat([pred_img, gt, error]))
    axs[1].scatter(true_pts[..., 0], true_pts[..., 1], color="dodgerblue", label="True")
    axs[1].scatter(pred_pts[..., 0], pred_pts[..., 1], color="darkorange", label="Pred")
    for x, y in zip(pred_pts, true_pts):
        axs[1].plot([x[0], y[0]], [x[1], y[1]], "w--")
    axs[1].legend()

    # Plot the predicted, true, and error images
    plot_drr(
        torch.concat([pred_img, gt, error]),
        title=["DRR from Predicted Pose", "Ground truth X-ray",
               f"Error (mTRE = {mtre:.2f} mm)"],
        ticks=False,
        axs=axs,
    )
    axs[2].imshow(error[0].permute(1, 2, 0), cmap="bwr", vmin=-error.abs().max(), vmax=error.abs().max())
    
    plt.tight_layout()
    plt.show()
