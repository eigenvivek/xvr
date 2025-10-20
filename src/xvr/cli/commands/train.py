import click

from ..formatter import CategorizedCommand, categorized_option


@click.command(
    cls=CategorizedCommand,
    category_order=[
        "Required",
        "Data",
        "Model",
        "Checkpoint",
        "Sampling",
        "Renderer",
        "Optimizer",
        "Logging",
    ],
)
@categorized_option(
    "-v",
    "--volpath",
    required=True,
    type=click.Path(exists=True),
    help="A single CT or a directory of CTs for pretraining",
    category="Required",
)
@categorized_option(
    "-m",
    "--maskpath",
    required=False,
    type=click.Path(exists=True),
    help="Optional labelmaps for the CTs passed in `volpath`",
    category="Data",
)
@categorized_option(
    "-c",
    "--ckptpath",
    required=False,
    type=click.Path(exists=True),
    help="Checkpoint of a pretrained pose regressor",
    category="Checkpoint",
)
@categorized_option(
    "-o",
    "--outpath",
    required=True,
    type=click.Path(),
    help="Directory in which to save model weights",
    category="Required",
)
@categorized_option(
    "--r1",
    required=True,
    type=(float, float),
    help="Range for primary angle (in degrees)",
    category="Sampling",
)
@categorized_option(
    "--r2",
    required=True,
    type=(float, float),
    help="Range for secondary angle (in degrees)",
    category="Sampling",
)
@categorized_option(
    "--r3",
    required=True,
    type=(float, float),
    help="Range for tertiary angle (in degrees)",
    category="Sampling",
)
@categorized_option(
    "--tx",
    required=True,
    type=(float, float),
    help="Range for x-offset (in millimeters)",
    category="Sampling",
)
@categorized_option(
    "--ty",
    required=True,
    type=(float, float),
    help="Range for y-offset (in millimeters)",
    category="Sampling",
)
@categorized_option(
    "--tz",
    required=True,
    type=(float, float),
    help="Range for z-offset (in millimeters)",
    category="Sampling",
)
@categorized_option(
    "--sdd",
    required=True,
    type=float,
    help="Source-to-detector distance (in millimeters)",
    category="Renderer",
)
@categorized_option(
    "--height",
    required=True,
    type=int,
    help="DRR height (in pixels)",
    category="Renderer",
)
@categorized_option(
    "--delx",
    required=True,
    type=float,
    help="DRR pixel size (in millimeters / pixel)",
    category="Renderer",
)
@categorized_option(
    "--renderer",
    default="trilinear",
    type=click.Choice(["siddon", "trilinear"]),
    help="Rendering equation",
    category="Renderer",
)
@categorized_option(
    "--orientation",
    default="AP",
    type=click.Choice(["AP", "PA"]),
    help="Orientation of CT volumes",
    category="Renderer",
)
@categorized_option(
    "--reverse_x_axis",
    default=False,
    is_flag=True,
    help="Enable to obey radiologic convention (e.g., heart on right)",
    category="Renderer",
)
@categorized_option(
    "--model_name",
    default="resnet18",
    type=str,
    help="Name of model to instantiate from the timm library",
    category="Model",
)
@categorized_option(
    "--norm_layer",
    default="groupnorm",
    type=str,
    help="Normalization layer",
    category="Model",
)
@categorized_option(
    "--pretrained",
    default=False,
    is_flag=True,
    help="Load pretrained ImageNet-1k weights",
    category="Model",
)
@categorized_option(
    "--parameterization",
    default="euler_angles",
    type=str,
    help="Parameterization of SO(3) for regression",
    category="Model",
)
@categorized_option(
    "--convention",
    default="ZXY",
    type=str,
    help="If `parameterization='euler_angles'`, specify order",
    category="Model",
)
@categorized_option(
    "--unit_conversion_factor",
    default=1000.0,
    type=float,
    help="Scale factor for translation prediction (e.g., from m to mm)",
    category="Model",
)
@categorized_option(
    "--p_augmentation",
    default=0.5,
    type=float,
    help="Base probability of image augmentations during training",
    category="Model",
)
@categorized_option(
    "--lr",
    default=5e-3,
    type=float,
    help="Maximum learning rate",
    category="Optimizer",
)
@categorized_option(
    "--weight_geo",
    default=1e-2,
    type=float,
    help="Weight on geodesic loss term",
    category="Optimizer",
)
@categorized_option(
    "--weight_dice",
    default=1e-1,
    type=float,
    help="Weight on Dice loss term",
    category="Optimizer",
)
@categorized_option(
    "--batch_size",
    default=116,
    type=int,
    help="Number of DRRs per batch",
    category="Sampling",
)
@categorized_option(
    "--n_total_itrs",
    default=int(1e6),
    type=int,
    help="Number of iterations for training the model",
    category="Optimizer",
)
@categorized_option(
    "--n_warmup_itrs",
    default=int(1e3),
    type=int,
    help="Number of iterations for warming up the learning rate",
    category="Optimizer",
)
@categorized_option(
    "--n_grad_accum_itrs",
    default=4,
    type=int,
    help="Number of iterations for gradient accumulation",
    category="Optimizer",
)
@categorized_option(
    "--n_save_every_itrs",
    default=int(2.5e3),
    type=int,
    help="Number of iterations before saving a new model checkpoint",
    category="Optimizer",
)
@categorized_option(
    "--reuse_optimizer",
    default=False,
    is_flag=True,
    help="If ckptpath passed, initialize the previous optimizer's state",
    category="Checkpoint",
)
@categorized_option(
    "--lora_target_modules",
    default=None,
    type=str,
    help="Target modules for which to create LoRA adapters",
    category="Model",
)
@categorized_option(
    "-w",
    "--warp",
    type=click.Path(exists=True),
    help="SimpleITK transform to warp input CT to the checkpoint's reference frame",
    category="Checkpoint",
)
@categorized_option(
    "--invert",
    default=False,
    is_flag=True,
    help="Whether to invert the warp or not",
    category="Checkpoint",
)
@categorized_option(
    "--preload_volumes",
    default=False,
    is_flag=True,
    help="If directory of CTs are passed, try to load all into memory (speeds up training)",
    category="Data",
)
@categorized_option(
    "--name",
    default=None,
    type=str,
    help="WandB run name",
    category="Logging",
)
@categorized_option(
    "--project",
    default="xvr",
    type=str,
    help="WandB project name",
    category="Logging",
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
    model_name,
    norm_layer,
    pretrained,
    parameterization,
    convention,
    unit_conversion_factor,
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
    lora_target_modules,
    warp,
    invert,
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

    from ...model import Trainer

    # Create the output directory for saving model weights
    Path(outpath).mkdir(parents=True, exist_ok=True)

    # If ckptpath is a directory, get the last saved model
    ckptpath = Path(ckptpath)
    if ckptpath.is_dir():
        ckptpath = sorted(ckptpath.glob("*.pth"))[-1]
    ckptpath = str(ckptpath)

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
        unit_conversion_factor=unit_conversion_factor,
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
        lora_target_modules=lora_target_modules,
        preload_volumes=preload_volumes,
        warp=warp,
        invert=invert,
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
