from pathlib import Path
from tempfile import TemporaryDirectory

import click
import matplotlib.pyplot as plt
import torch
from diffdrr.metrics import DoubleGeodesicSE3
from diffdrr.pose import RigidTransform, convert
from diffdrr.visualization import plot_drr
from imageio.v3 import imread, imwrite
from tqdm import tqdm

from ..utils import XrayTransforms
from .register import (
    _parse_scales,
    initialize_pose,
    initialize_registration,
)


@click.command(context_settings=dict(show_default=True, max_content_width=120))
@click.option(
    "-i",
    "--inpath",
    required=True,
    type=click.Path(exists=True),
    help="Saved registration result from `xvr register`",
)
@click.option(
    "-o",
    "--outpath",
    required=True,
    type=click.Path(),
    help="Savepath for iterative optimization animation",
)
@click.option(
    "--dpi",
    default=192,
    type=int,
    help="DPI of individual animation frames",
)
@click.option(
    "--fps",
    default=30,
    type=int,
    help="FPS of animation",
)
def animate(inpath, outpath, dpi, fps):
    """Animate the trajectory of iterative optimization."""
    
    # Initialize the renderer and ground truth data
    run = torch.load(inpath, weights_only=False)
    args = run["arguments"]
    drr, gt, gt_pose, double_geodesic, scales = initialize(args)

    # Render all DRRs
    drrs, nccs, dgeos = render(drr, gt, gt_pose, double_geodesic, scales, run, args)

    # Generate the animation
    frames = plot(drrs, nccs, dgeos, dpi)
    imwrite(outpath, frames, fps=fps)


def initialize(args):
    """Initialize the DRR and ground truth."""

    # Initialize the ground truth X-ray
    gt, sdd, delx, dely, x0, y0, init_pose, height, config, date = initialize_pose(
        args["i2d"],
        args["ckptpath"],
        args["crop"],
        args["subtract_background"],
        args["linearize"],
        args["warp"],
        args["volume"],
    )

    # Get the ground truth pose
    double_geodesic = DoubleGeodesicSE3(sdd)
    gt_pose = torch.load(str(args["i2d"]).split(".")[0] + ".pt")["pose"]
    gt_pose = RigidTransform(gt_pose)

    # Initialize the renderer
    drr = initialize_registration(
        args["volume"],
        args["mask"],
        args["labels"],
        config["orientation"],
        gt,
        sdd,
        dely,
        dely,
        x0,
        y0,
        args["reverse_x_axis"],
        args["renderer"],
        init_pose,
        args["parameterization"],
        args["convention"],
    ).drr

    # Parse the scales
    scales = _parse_scales(args["scales"], args["crop"], height)

    return drr, gt, gt_pose, double_geodesic, scales


def render(drr, gt, gt_pose, double_geodesic, scales, run, args):
    lowest_lr = 0.0

    drrs = []
    nccs = []
    dgeos = []
    for _, row in tqdm(run["trajectory"].iterrows(), total=len(run["trajectory"]), desc="Rendering DRRs"):
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
            parameterization=args["parameterization"],
            convention=args["convention"],
        ).to(dtype=torch.float32, device="cuda")
        with torch.no_grad():
            pred = drr(pose)
            pred = transform(pred)

        # Get the image similarity
        nccs.append(row.ncc)
        # Get the geodesic error
        _, _, dgeo = double_geodesic(pose.cpu(), gt_pose)
        dgeos.append(dgeo.item())

        # Save the results
        drrs.append([true.cpu(), pred.cpu()])

    return drrs, nccs, dgeos


def plot(drrs, nccs, dgeos, dpi):
    # Get ylim for the image similarity metric and geodesic distances
    plt.plot(nccs)
    ncc_ylim = plt.gca().get_ylim()
    plt.close()
    plt.plot(dgeos)
    plt.yscale("log")
    dgeo_ylim = plt.gca().get_ylim()
    plt.close()

    # Plot every frame of the animation
    with TemporaryDirectory() as tmpdir:
        running_nccs = []
        running_dgeos = []
        for idx, (ncc, dgeo, img) in tqdm(
            enumerate(zip(nccs, dgeos, drrs)), total=len(nccs), desc="Rendering frames"
        ):
            imgs = [img[0], img[1], img[0] - img[1]]
            running_nccs.append(ncc)
            running_dgeos.append(dgeo)

            plt.figure(
                figsize=(10, 6),
                dpi=dpi,
            )
            ax1 = plt.subplot(231)
            ax2 = plt.subplot(232)
            ax3 = plt.subplot(233)
            plot_drr(
                torch.concat(imgs),
                axs=[ax1, ax2, ax3],
                title=["Ground Truth", "Prediction", "Difference"],
                # ticks=False,
            )

            ax4 = plt.subplot(223)
            ax4.plot(running_nccs)
            ax4.set(
                xlim=(0, len(nccs)),
                ylim=ncc_ylim,
                xlabel="Iteration",
                ylabel="Image Similarity",
            )

            ax5 = plt.subplot(224)
            ax5.plot(running_dgeos)
            ax5.set(
                yscale="log",
                xlim=(0, len(nccs)),
                ylim=dgeo_ylim,
                xlabel="Iteration",
                ylabel="Error (mm)",
            )

            plt.gcf().set_size_inches(10, 6)
            plt.savefig(f"{tmpdir}/{idx:04d}.png", pad_inches=0, dpi=dpi)
            plt.close()

        # Read the images
        imgs = []
        for filepath in sorted(Path(tmpdir).glob("*.png")):
            img = imread(filepath)
            imgs.append(img)

    return imgs
