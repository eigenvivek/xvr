import click


@click.command(context_settings=dict(show_default=True, max_content_width=120))
@click.argument(
    "xray",
    nargs=-1,
    required=True,
    type=click.Path(exists=True),
)
@click.option(
    "-v",
    "--volume",
    required=True,
    type=click.Path(exists=True),
    help="Input CT volume (3D image)",
)
@click.option(
    "-m",
    "--mask",
    type=click.Path(exists=True),
    help="Labelmap for the CT volume (optional)",
)
@click.option(
    "-c",
    "--ckptpath",
    required=True,
    type=click.Path(exists=True),
    help="Checkpoint of a pretrained pose regressor",
)
@click.option(
    "-o",
    "--outpath",
    required=True,
    type=click.Path(),
    help="Directory for saving registration results",
)
@click.option(
    "--crop",
    default=0,
    type=int,
    help="Preprocessing: center crop the X-ray image",
)
@click.option(
    "--subtract_background",
    default=False,
    is_flag=True,
    help="Preprocessing: subtract mode X-ray image intensity",
)
@click.option(
    "--linearize",
    default=False,
    is_flag=True,
    help="Preprocessing: convert X-ray from exponential to linear form",
)
@click.option(
    "--warp",
    type=click.Path(exists=True),
    help="SimpleITK transform to warp input CT to template reference frame",
)
@click.option(
    "--invert",
    default=False,
    is_flag=True,
    help="Invert the warp",
)
@click.option(
    "--model_only",
    default=False,
    is_flag=True,
    help="Directly return the output of the pose regressor (no test-time optimization)",
)
@click.option(
    "--labels",
    type=str,
    help="Labels in mask to exclusively render (comma separated)",
)
@click.option(
    "--scales",
    default="8",
    type=str,
    help="Scales of downsampling for multiscale registration (comma separated)",
)
@click.option(
    "--reverse_x_axis",
    default=False,
    is_flag=True,
    help="Enable to obey radiologic convention (e.g., heart on right)",
)
@click.option(
    "--renderer",
    default="trilinear",
    type=click.Choice(["siddon", "trilinear"]),
    help="Rendering equation",
)
@click.option(
    "--parameterization",
    default="euler_angles",
    type=str,
    help="Parameterization of SO(3) for regression",
)
@click.option(
    "--convention",
    default="ZXY",
    type=str,
    help="If parameterization is Euler angles, specify order",
)
@click.option(
    "--lr_rot",
    default=1e-2,
    type=float,
    help="Initial step size for rotational parameters",
)
@click.option(
    "--lr_xyz",
    default=1e0,
    type=float,
    help="Initial step size for translational parameters",
)
@click.option(
    "--patience",
    default=10,
    type=int,
    help="Number of allowed epochs with no improvement after which the learning rate will be reduced",
)
@click.option(
    "--max_n_itrs",
    default=500,
    type=int,
    help="Maximum number of iterations to run at each scale",
)
@click.option(
    "--max_n_plateaus",
    default=3,
    type=int,
    help="Number of times loss can plateau before moving to next scale",
)
@click.option(
    "--pattern",
    default="*.dcm",
    type=str,
    help="Pattern rule for glob is XRAY is directory",
)
def register(
    xray,
    volume,
    mask,
    ckptpath,
    outpath,
    crop,
    subtract_background,
    linearize,
    warp,
    invert,
    model_only,
    labels,
    scales,
    reverse_x_axis,
    renderer,
    parameterization,
    convention,
    lr_rot,
    lr_xyz,
    patience,
    max_n_itrs,
    max_n_plateaus,
    pattern,
):
    """
    Use gradient-based optimization to register XRAY to a CT.

    Can pass multiple DICOM files or a directory in XRAY.
    """
    from pathlib import Path

    dcmfiles = []
    for xpath in xray:
        xpath = Path(xpath)
        if xpath.is_file():
            dcmfiles.append(xpath)
        else:
            dcmfiles += sorted(xpath.glob(pattern))

    for i2d in dcmfiles:
        print(f"\nRegistering {i2d} ...")
        run(
            Path(i2d).resolve(),
            Path(volume).resolve(),
            Path(mask).resolve() if mask is not None else None,
            Path(ckptpath).resolve(),
            Path(outpath).resolve(),
            crop,
            subtract_background,
            linearize,
            Path(warp).resolve() if warp is not None else None,
            invert,
            model_only,
            labels,
            scales,
            reverse_x_axis,
            renderer,
            parameterization,
            convention,
            lr_rot,
            lr_xyz,
            patience,
            max_n_itrs,
            max_n_plateaus,
        )


def run(
    i2d,
    volume,
    mask,
    ckptpath,
    outpath,
    crop,
    subtract_background,
    linearize,
    warp,
    invert,
    model_only,
    labels,
    scales,
    reverse_x_axis,
    renderer,
    parameterization,
    convention,
    lr_rot,
    lr_xyz,
    patience,
    max_n_itrs,
    max_n_plateaus,
):
    # Save input arguments
    arguments = locals()

    # Import required packages
    from pathlib import Path

    import torch
    from diffdrr.metrics import (
        GradientNormalizedCrossCorrelation2d,
        MultiscaleNormalizedCrossCorrelation2d,
    )
    from tqdm import tqdm

    from ..renderer import initialize_registration
    from ..utils import XrayTransforms

    # Make the savepath
    outpath = Path(outpath)
    savepath = outpath / f"{i2d.stem}.pt"

    # Get initial pose estimate and intrinsic parameters of the imaging system
    print("Predicting initial pose...")
    gt, sdd, delx, dely, x0, y0, init_pose, height, config, date = initialize_pose(
        i2d, ckptpath, crop, subtract_background, linearize, warp, volume, invert
    )
    if model_only:
        save(
            arguments,
            sdd,
            delx,
            dely,
            x0,
            y0,
            init_pose,
            None,
            config,
            date,
            savepath,
        )
        return

    # Initialize the Registration module
    print("Initializing iterative optimizer...")
    reg = initialize_registration(
        volume,
        mask,
        labels,
        config["orientation"],
        gt,
        sdd,
        delx,
        dely,
        x0,
        y0,
        reverse_x_axis,
        renderer,
        init_pose,
        parameterization,
        convention,
    )

    # Initialize the image similarity metrics
    imagesim1 = MultiscaleNormalizedCrossCorrelation2d([None, 9], [0.5, 0.5])
    imagesim2 = GradientNormalizedCrossCorrelation2d(patch_size=11, sigma=10).cuda()

    def imagesim(x, y):
        return 0.5 * imagesim1(x, y) + 0.5 * imagesim2(x, y)

    # Perform multiscale registration
    params = [
        torch.concat(reg.pose.convert("euler_angles", "ZXY"), dim=-1).squeeze().tolist()
    ]
    nccs = []
    alphas = [[lr_rot, lr_xyz]]

    step_size_scalar = 1.0
    scales = _parse_scales(scales, crop, height)
    for stage, scale in enumerate(scales, start=1):
        # Rescale DRR detector and ground truth image
        reg.drr.rescale_detector_(scale)
        transform = XrayTransforms(reg.drr.detector.height, reg.drr.detector.width)
        img = transform(gt).cuda()

        # Initialize the optimizer and scheduler
        step_size_scalar *= 2 ** (stage - 1)
        optimizer = torch.optim.Adam(
            [
                {"params": [reg.rotation], "lr": lr_rot / step_size_scalar},
                {"params": [reg.translation], "lr": lr_xyz / step_size_scalar},
            ],
            maximize=True,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1, patience=patience, mode="max"
        )

        # Iteratively optimize at this scale until improvements in image similarity plateau
        n_plateaus = 0
        current_lr = torch.inf
        for _ in (
            pbar := tqdm(range(max_n_itrs + 1), ncols=100, desc=f"Stage {stage}")
        ):
            optimizer.zero_grad()
            pred_img = reg()
            pred_img = transform(pred_img)
            loss = imagesim(img, pred_img)
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

            # Record current loss
            pbar.set_postfix_str(f"ncc = {loss.item():5.3f}")
            nccs.append(loss.item())
            params.append(
                torch.concat(reg.pose.convert("euler_angles", "ZXY"), dim=-1)
                .squeeze()
                .tolist()
            )

            # Determine update to the learning rate
            lr = scheduler.get_last_lr()
            alphas.append(lr)
            if lr[0] < current_lr:
                current_lr = lr[0]
                n_plateaus += 1
                tqdm.write("→ Plateaued... decreasing step size")
            if n_plateaus == max_n_plateaus:
                break

    # Record the final NCC value
    with torch.no_grad():
        pred_img = reg()
        pred_img = transform(pred_img)
        loss = imagesim(img, pred_img)
    nccs.append(loss.item())

    trajectory = _make_csv(
        params,
        nccs,
        alphas,
        columns=["r1", "r2", "r3", "tx", "ty", "tz", "ncc", "lr_rot", "lr_xyz"],
    )
    save(
        arguments,
        sdd,
        delx,
        dely,
        x0,
        y0,
        init_pose,
        reg.pose,
        config,
        date,
        savepath,
        trajectory=trajectory,
    )


def initialize_pose(
    i2d,
    ckptpath,
    crop,
    subtract_background,
    linearize,
    warp,
    volume,
    invert,
):
    """Get initial pose estimate and image intrinsics."""

    from ..dicom import _parse_dicom, _preprocess_xray
    from ..model import _correct_pose

    # Preprocess X-ray image and get imaging system intrinsics
    img, sdd, delx, dely, x0, y0 = _parse_dicom(i2d)
    img = _preprocess_xray(img, crop, subtract_background, linearize)

    # Get the predicted pose from the model
    init_pose, height, config, date = _predict_initial_pose(
        img, sdd, delx, dely, x0, y0, ckptpath
    )

    # Optionally, correct the pose by warping the CT volume to the template
    init_pose = _correct_pose(init_pose, warp, volume, invert)

    return img, sdd, delx, dely, x0, y0, init_pose, height, config, date


def _predict_initial_pose(img, sdd, delx, dely, x0, y0, ckptpath, meta=True):
    from ..model import load_model, predict_pose

    # Load the pretrained model
    model, config, date = load_model(ckptpath, meta)

    # Predict the pose of the X-ray image
    init_pose, height = predict_pose(model, config, img, sdd, delx, dely, x0, y0, meta)

    return init_pose, height, config, date


def _parse_scales(scales: str, crop: int, height: int):
    pyramid = [1.0] + [float(x) * (height / (height + crop)) for x in scales.split(",")]
    scales = []
    for idx in range(len(pyramid) - 1):
        scales.append(pyramid[idx] / pyramid[idx + 1])
    return scales


def save(
    arguments,
    sdd,
    delx,
    dely,
    x0,
    y0,
    init_pose,
    final_pose,
    config,
    date,
    savepath,
    **kwargs,
):
    import torch

    print("Saving registration results...")
    savepath.parent.mkdir(parents=True, exist_ok=True)
    init_pose = init_pose.matrix.detach().cpu()
    final_pose = final_pose.matrix.detach().cpu() if final_pose is not None else None
    torch.save(
        {
            "arguments": arguments,
            "intrinsics": dict(sdd=sdd, delx=delx, dely=dely, x0=x0, y0=y0),
            "init_pose": init_pose,
            "final_pose": final_pose,
            "config": config,
            "date": date,
            **kwargs,
        },
        savepath,
    )


def _make_csv(*metrics, columns):
    import numpy as np
    import pandas as pd

    ls = []
    for metric in metrics:
        metric = np.array(metric)
        if metric.ndim == 1:
            metric = metric[..., np.newaxis]
        ls.append(metric)
    ls = np.concatenate(ls, axis=1)
    df = pd.DataFrame(ls, columns=columns)
    return df
