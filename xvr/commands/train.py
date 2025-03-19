import click


@click.command(context_settings=dict(show_default=True, max_content_width=120))
@click.option(
    "-i",
    "--inpath",
    required=True,
    type=click.Path(exists=True),
    help="A single CT or a directory of CTs for pretraining",
)
@click.option(
    "-o",
    "--outpath",
    required=True,
    type=click.Path(),
    help="Directory in which to save model weights",
)
@click.option(
    "--r1",
    required=True,
    type=(float, float),
    help="Range for primary angle (in degrees)",
)
@click.option(
    "--r2",
    required=True,
    type=(float, float),
    help="Range for secondary angle (in degrees)",
)
@click.option(
    "--r3",
    required=True,
    type=(float, float),
    help="Range for tertiary angle (in degrees)",
)
@click.option(
    "--tx",
    required=True,
    type=(float, float),
    help="Range for x-offset (in millimeters)",
)
@click.option(
    "--ty",
    required=True,
    type=(float, float),
    help="Range for y-offset (in millimeters)",
)
@click.option(
    "--tz",
    required=True,
    type=(float, float),
    help="Range for z-offset (in millimeters)",
)
@click.option(
    "--sdd",
    required=True,
    type=float,
    help="Source-to-detector distance (in millimeters)",
)
@click.option(
    "--height",
    required=True,
    type=int,
    help="DRR height (in pixels)",
)
@click.option(
    "--delx",
    required=True,
    type=float,
    help="DRR pixel size (in millimeters / pixel)",
)
@click.option(
    "--renderer",
    default="trilinear",
    type=click.Choice(["siddon", "trilinear"]),
    help="Rendering equation",
)
@click.option(
    "--orientation",
    default="PA",
    type=click.Choice(["AP", "PA"]),
    help="Orientation of CT volumes",
)
@click.option(
    "--reverse_x_axis",
    default=False,
    is_flag=True,
    help="Enable to obey radiologic convention (e.g., heart on right)",
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
    "--model_name",
    default="resnet18",
    type=str,
    help="Name of model to instantiate",
)
@click.option(
    "--pretrained",
    default=False,
    is_flag=True,
    help="Load pretrained ImageNet-1k weights",
)
@click.option(
    "--norm_layer",
    default="groupnorm",
    type=str,
    help="Normalization layer",
)
@click.option(
    "--lr",
    default=5e-3,
    type=float,
    help="Maximum learning rate",
)
@click.option(
    "--weight-geo",
    default=1e-2,
    type=float,
    help="Weight on geodesic loss term",
)
@click.option(
    "--batch_size",
    default=116,
    type=int,
    help="Number of DRRs per batch",
)
@click.option(
    "--n_epochs",
    default=1000,
    type=int,
    help="Number of epochs",
)
@click.option(
    "--n_batches_per_epoch",
    default=100,
    type=int,
    help="Number of batches per epoch",
)
@click.option(
    "--name",
    default=None,
    type=str,
    help="WandB run name",
)
@click.option(
    "--project",
    default="xvr",
    type=str,
    help="WandB project name",
)
def train(
    inpath,
    outpath,
    r1,
    r2,
    r3,
    tx,
    ty,
    tz,
    sdd,
    height,
    delx,
    renderer,
    orientation,
    reverse_x_axis,
    parameterization,
    convention,
    model_name,
    pretrained,
    norm_layer,
    lr,
    weight_geo,
    batch_size,
    n_epochs,
    n_batches_per_epoch,
    name,
    project,
):
    """
    Train a pose regression model from scratch.
    """
    import os
    from pathlib import Path

    import wandb

    # Create the output directory for saving model weights
    Path(outpath).mkdir(parents=True, exist_ok=True)

    # Parse 6-DoF pose parameters
    alphamin, alphamax = r1
    betamin, betamax = r2
    gammamin, gammamax = r3
    txmin, txmax = tx
    tymin, tymax = ty
    tzmin, tzmax = tz

    # Parse configuration parameters
    config = dict(
        inpath=inpath,
        outpath=outpath,
        alphamin=alphamin,
        alphamax=alphamax,
        betamin=betamin,
        betamax=betamax,
        gammamin=gammamin,
        gammamax=gammamax,
        txmin=txmin,
        txmax=txmax,
        tymin=tymin,
        tymax=tymax,
        tzmin=tzmin,
        tzmax=tzmax,
        sdd=sdd,
        height=height,
        delx=delx,
        renderer=renderer,
        orientation=orientation,
        reverse_x_axis=reverse_x_axis,
        parameterization=parameterization,
        convention=convention,
        model_name=model_name,
        pretrained=pretrained,
        norm_layer=norm_layer,
        lr=lr,
        weight_geo=weight_geo,
        batch_size=batch_size,
        n_epochs=n_epochs,
        n_batches_per_epoch=n_batches_per_epoch,
    )

    # Set up logging and train the model
    wandb.login(key=os.environ["WANDB_API_KEY"])
    run = wandb.init(
        project=project,
        name=name if name is not None else project,
        config=config,
    )
    train_model(config, run)


def train_model(config, run):
    from datetime import datetime
    from pathlib import Path
    from random import choice

    import torch
    from diffdrr.data import read
    from diffdrr.metrics import (
        DoubleGeodesicSE3,
        MultiscaleNormalizedCrossCorrelation2d,
    )
    from timm.utils.agc import adaptive_clip_grad as adaptive_clip_grad_
    from tqdm import tqdm

    import wandb

    from ..utils import XrayAugmentations, get_random_pose, render

    # Load all CT volumes
    volumes = []
    inpath = Path(config["inpath"])
    niftis = [inpath] if inpath.is_file() else sorted(inpath.glob("*.nii.gz"))
    for filepath in tqdm(niftis, desc="Reading CTs..."):
        subject = read(filepath, orientation=config["orientation"])
        volumes.append(subject.volume.data.squeeze().to(dtype=torch.float32))

    # Initialize deep learning modules
    model, drr, transforms, optimizer, scheduler = initialize(config, subject)

    # Initialize the loss function
    imagesim = MultiscaleNormalizedCrossCorrelation2d([None, 9], [0.5, 0.5])
    geodesic = DoubleGeodesicSE3(config["sdd"])

    # Set up augmentations
    contrast_distribution = torch.distributions.Uniform(1.0, 10.0)
    augmentations = XrayAugmentations()

    # Train the model
    for epoch in range(config["n_epochs"] + 1):
        for _ in tqdm(range(config["n_batches_per_epoch"]), desc=f"Epoch {epoch}"):
            # Sample a random volume for this batch
            volume = choice(volumes).cuda()

            # Sample a batch of random poses
            pose = get_random_pose(config).cuda()

            # Render random DRRs and apply transforms
            contrast = contrast_distribution.sample().item()
            img, _, _ = render(drr, pose, volume, contrast)
            with torch.no_grad():
                img = augmentations(img)
            img = transforms(img)

            # Regress the poses and render the predicted DRRs
            pred_pose = model(img)
            pred_img, _, _ = render(drr, pred_pose, volume, contrast)

            # Compute the loss
            mncc = imagesim(img, pred_img)
            rgeo, tgeo, dgeo = geodesic(pose, pred_pose)
            loss = 1 - mncc + config["weight_geo"] * dgeo

            # Optimize the model
            optimizer.zero_grad()
            loss.mean().backward()
            adaptive_clip_grad_(model.parameters())
            optimizer.step()
            scheduler.step()

            # Log metrics
            wandb.log(
                {
                    "mncc": mncc.mean().item(),
                    "dgeo": dgeo.mean().item(),
                    "rgeo": rgeo.mean().item(),
                    "tgeo": tgeo.mean().item(),
                    "loss": loss.mean().item(),
                    "lr": scheduler.get_last_lr()[0],
                }
            )

        # Checkpoint the model every 5 epochs (and the first 100 epochs)
        if epoch % 5 == 0 or epoch < 100:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                    "date": datetime.now(),
                    "config": config,
                },
                f"{config['outpath']}/{run.name}_{epoch:04d}.pth",
            )


def initialize(config, subject):
    import torch
    from diffdrr.drr import DRR
    from diffdrr.registration import PoseRegressor

    from ..model import WarmupCosineSchedule
    from ..utils import XrayTransforms

    # Initialize the pose regression model
    model = PoseRegressor(
        model_name=config["model_name"],
        pretrained=config["pretrained"],
        parameterization=config["parameterization"],
        convention=config["convention"],
        norm_layer=config["norm_layer"],
        height=config["height"],
    ).cuda()

    # Initialize a DRR renderer with a placeholder subject
    drr = DRR(
        subject,
        sdd=config["sdd"],
        height=config["height"],
        delx=config["delx"],
        reverse_x_axis=config["reverse_x_axis"],
        renderer=config["renderer"],
    ).cuda()
    transforms = XrayTransforms(config["height"])

    # Initialize the optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = WarmupCosineSchedule(
        optimizer,
        5 * config["n_batches_per_epoch"],
        config["n_epochs"] * config["n_batches_per_epoch"]
        - 5 * config["n_batches_per_epoch"],
    )  # Warmup for 5 epochs, then taper off

    return model, drr, transforms, optimizer, scheduler
