from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import torch
from diffdrr.pose import convert
from diffdrr.visualization import plot_drr
from imageio.v3 import imread, imwrite
from tqdm import tqdm

from ..io import read_xray
from ..registrar import _parse_scales
from ..renderer import initialize_drr
from ..utils import XrayTransforms


def animate(
    inpath: str | Path,  # Saved registration result from <xvr register>
    outpath: str | Path,  # Savepath for iterative optimization animation
    skip: int = 1,  # Animate every <skip> frames of the optimization
    dpi: int = 192,  # DPI of each frame of the animation
    fps: int = 30,  # FPS of the animation
):
    """Animate the trajectory of iterative optimization."""

    # Initialize the renderer
    run = torch.load(inpath, weights_only=False)
    drr = initialize_drr(**run["drr"])
    gt, *_ = read_xray(**run["xray"])
    scales = _parse_scales(run["optimization"]["scales"], run["xray"]["crop"], run["drr"]["height"])

    # Render all DRRs
    drrs = render(drr, gt, scales, run, skip)

    # Generate the animation
    frames = plot(drrs, dpi)
    imwrite(outpath, frames, fps=fps)


def render(drr, gt, scales, run, skip):
    lowest_lr = 0.0

    drrs = []
    for idx, row in tqdm(
        run["trajectory"].iterrows(),
        total=len(run["trajectory"]),
        desc="Rendering DRRs",
    ):
        # Animate every <skip> frames
        if idx % skip != 0:
            continue

        # If the learning rate has reset, rescale the detector
        if row.lr_rot > lowest_lr:
            scale = scales.pop(0)
            drr.rescale_detector_(scale)
            transform = XrayTransforms(drr.detector.height, drr.detector.width)
            true = transform(gt)
        lowest_lr = row.lr_rot

        # Render the current estimate
        pose = convert(
            torch.tensor([[row.r1, row.r2, row.r3]]),
            torch.tensor([[row.tx, row.ty, row.tz]]),
            parameterization=run["optimization"]["parameterization"],
            convention=run["optimization"]["convention"],
        ).to(dtype=torch.float32, device="cuda")
        with torch.no_grad():
            pred = drr(pose)
            pred = transform(pred)

        # Save the results
        drrs.append([true.cpu(), pred.cpu()])

    return drrs


def plot(drrs, dpi):
    # Plot every frame of the animation
    with TemporaryDirectory() as tmpdir:
        for idx, img in tqdm(enumerate(drrs), total=len(drrs), desc="Rendering frames"):
            imgs = [img[0], img[1], img[0] - img[1]]

            plt.figure(figsize=(9, 3), dpi=dpi)
            ax1 = plt.subplot(131)
            ax2 = plt.subplot(132)
            ax3 = plt.subplot(133)
            plot_drr(
                torch.concat(imgs),
                axs=[ax1, ax2, ax3],
                title=["Ground Truth", "Prediction", "Difference"],
                # ticks=False,
            )
            plt.gcf().set_size_inches(9, 3)
            plt.savefig(f"{tmpdir}/{idx:04d}.png", pad_inches=0, dpi=dpi)
            plt.close()

        # Read the images
        imgs = []
        for filepath in sorted(Path(tmpdir).glob("*.png")):
            img = imread(filepath)
            imgs.append(img)

    return imgs
