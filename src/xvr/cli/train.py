import click


@click.command(context_settings=dict(show_default=True, max_content_width=120))
@click.option(
    "-v",
    "--volpath",
    required=True,
    type=click.Path(exists=True),
    help="A single CT or a directory of CTs for pretraining",
)
@click.option(
    "-m",
    "--maskpath",
    required=False,
    type=click.Path(exists=True),
    help="Labelmaps for the CTs in volpath",
)
@click.option(
    "-c",
    "--ckptpath",
    required=False,
    type=click.Path(exists=True),
    help="Checkpoint of a pretrained pose regressor",
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
    default="batchnorm",
    type=str,
    help="Normalization layer",
)
@click.option(
    "--p_augmentation",
    default=0.5,
    type=float,
    help="Probability of applying augmentations during training",
)
@click.option(
    "--lr",
    default=5e-3,
    type=float,
    help="Maximum learning rate",
)
@click.option(
    "--weight_geo",
    default=1e-2,
    type=float,
    help="Weight on geodesic loss term",
)
@click.option(
    "--weight_dice",
    default=1.0,
    type=float,
    help="Weight on Dice loss term",
)
@click.option(
    "--batch_size",
    default=116,
    type=int,
    help="Number of DRRs per batch",
)
@click.option(
    "--n_total_itrs",
    default=int(1e6),
    type=int,
    help="Number of iterations for training the model",
)
@click.option(
    "--n_warmup_itrs",
    default=int(1e3),
    type=int,
    help="Number of iterations for warming up the learning rate",
)
@click.option(
    "--n_grad_accum_itrs",
    default=4,
    type=int,
    help="Number of iterations for gradient accumulation",
)
@click.option(
    "--n_save_every_itrs",
    default=int(2.5e3),
    type=int,
    help="Number of iterations before saving a new model checkpoint",
)
@click.option(
    "--reuse_optimizer",
    default=False,
    is_flag=True,
    help="If ckptpath passed, initialize the previous optimizer's state",
)
@click.option(
    "--preload_volumes",
    default=False,
    is_flag=True,
    help="If directory of CTs are passed, load all into memory (speeds up training)",
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
    volpath,
    maskpath,
    ckptpath,
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
    p_augmentation,
    lr,
    weight_geo,
    weight_dice,
    batch_size,
    n_total_itrs,
    n_warmup_itrs,
    n_grad_accum_itrs,
    n_save_every_itrs,
    reuse_optimizer,
    preload_volumes,
    name,
    project,
):
    """
    Train a pose regression model from scratch.
    """
    import os
    from pathlib import Path

    import wandb

    from ..model import Trainer

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
        volpath=volpath,
        maskpath=maskpath,
        ckptpath=ckptpath,
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
        p_augmentation=p_augmentation,
        lr=lr,
        weight_geo=weight_geo,
        weight_dice=weight_dice,
        batch_size=batch_size,
        n_total_itrs=n_total_itrs,
        n_warmup_itrs=n_warmup_itrs,
        n_grad_accum_itrs=n_grad_accum_itrs,
        n_save_every_itrs=n_save_every_itrs,
        reuse_optimizer=reuse_optimizer,
        preload_volumes=preload_volumes,
    )

    # Set up logging
    wandb.login(key=os.environ["WANDB_API_KEY"])
    run = wandb.init(
        project=project,
        name=name if name is not None else project,
        config=config,
    )

    # Train the model
    trainer = Trainer(**config)
    trainer.train(run)
