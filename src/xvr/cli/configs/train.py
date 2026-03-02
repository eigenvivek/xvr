from dataclasses import dataclass
from typing import Annotated

from cyclopts import Group, Parameter

_DATA = Group("Data", sort_key=2)
_SAMPLING = Group("Sampling", sort_key=3)
_RENDERER = Group("Renderer", sort_key=4)
_MODEL = Group("Model", sort_key=5)
_OPTIMIZER = Group("Optimizer", sort_key=6)
_CHECKPOINT = Group("Checkpoint", sort_key=7)
_LOGGING = Group("Logging", sort_key=8)


@Parameter(name="*")
@dataclass
class TrainParams:
    # Data
    volpath: Annotated[str, Parameter(help="CT or directory of CTs for pretraining", group=_DATA)]
    outpath: Annotated[str, Parameter(help="Directory in which to save model weights", group=_DATA)]

    # Renderer
    sdd: Annotated[
        float, Parameter(help="Source-to-detector distance (in millimeters)", group=_RENDERER)
    ]
    height: Annotated[int, Parameter(help="DRR height (in pixels)", group=_RENDERER)]
    delx: Annotated[
        float, Parameter(help="DRR pixel size (in millimeters / pixel)", group=_RENDERER)
    ]

    # Sampling
    r1: Annotated[
        tuple[float, float], Parameter(help="Range for primary angle (in degrees)", group=_SAMPLING)
    ]
    r2: Annotated[
        tuple[float, float],
        Parameter(help="Range for secondary angle (in degrees)", group=_SAMPLING),
    ]
    r3: Annotated[
        tuple[float, float],
        Parameter(help="Range for tertiary angle (in degrees)", group=_SAMPLING),
    ]
    tx: Annotated[
        tuple[float, float], Parameter(help="Range for x-offset (in millimeters)", group=_SAMPLING)
    ]
    ty: Annotated[
        tuple[float, float], Parameter(help="Range for y-offset (in millimeters)", group=_SAMPLING)
    ]
    tz: Annotated[
        tuple[float, float], Parameter(help="Range for z-offset (in millimeters)", group=_SAMPLING)
    ]
    batch_size: Annotated[int, Parameter(help="Number of DRRs per batch", group=_SAMPLING)] = 116

    # Data (optional)
    maskpath: Annotated[
        str | None, Parameter(help="Optional labelmaps corresponding to the CTs", group=_DATA)
    ] = None
    patch_size: Annotated[
        str | None,
        Parameter(
            help="Optional random crop size e.g. 'h,w,d'; if None, return entire volume",
            group=_DATA,
        ),
    ] = None
    sample_weights: Annotated[
        str | None, Parameter(help="Probability for sampling each volume in volpath", group=_DATA)
    ] = None
    num_workers: Annotated[
        int, Parameter(help="Number of subprocesses to use in the dataloader", group=_DATA)
    ] = 4
    pin_memory: Annotated[
        bool, Parameter(help="Copy volumes into CUDA pinned memory before returning", group=_DATA)
    ] = False

    # Renderer (optional)
    orientation: Annotated[str, Parameter(help="Orientation of CT volumes", group=_RENDERER)] = "AP"
    reverse_x_axis: Annotated[
        bool, Parameter(help="Obey radiologic convention (e.g., heart on right)", group=_RENDERER)
    ] = False

    # Model
    model_name: Annotated[
        str, Parameter(help="Name of model to instantiate from the timm library", group=_MODEL)
    ] = "resnet18"
    norm_layer: Annotated[str, Parameter(help="Normalization layer", group=_MODEL)] = "groupnorm"
    pretrained: Annotated[
        bool, Parameter(help="Load pretrained ImageNet-1k weights", group=_MODEL)
    ] = False
    parameterization: Annotated[
        str, Parameter(help="Parameterization of SO(3) for regression", group=_MODEL)
    ] = "quaternion_adjugate"
    convention: Annotated[
        str, Parameter(help="If parameterization='euler_angles', specify order", group=_MODEL)
    ] = "ZXY"
    unit_conversion_factor: Annotated[
        float,
        Parameter(
            help="Scale factor for translation prediction (e.g., from m to mm)", group=_MODEL
        ),
    ] = 1000.0
    p_augmentation: Annotated[
        float,
        Parameter(help="Base probability of image augmentations during training", group=_MODEL),
    ] = 0.333
    use_compile: Annotated[
        bool, Parameter(help="Compile forward pass with max-autotune-no-cudagraphs", group=_MODEL)
    ] = False
    use_bf16: Annotated[bool, Parameter(help="Run all ops in bf16", group=_MODEL)] = False

    # Optimizer
    lr: Annotated[float, Parameter(help="Maximum learning rate", group=_OPTIMIZER)] = 2e-4
    weight_ncc: Annotated[float, Parameter(help="Weight on mNCC loss term", group=_OPTIMIZER)] = 1e0
    weight_geo: Annotated[
        float, Parameter(help="Weight on geodesic loss term", group=_OPTIMIZER)
    ] = 1e-2
    weight_dice: Annotated[float, Parameter(help="Weight on Dice loss term", group=_OPTIMIZER)] = (
        1e0
    )
    n_total_itrs: Annotated[
        int, Parameter(help="Number of iterations for training the model", group=_OPTIMIZER)
    ] = 1_000_000
    n_warmup_itrs: Annotated[
        int,
        Parameter(help="Number of iterations for warming up the learning rate", group=_OPTIMIZER),
    ] = 1_000
    n_grad_accum_itrs: Annotated[
        int, Parameter(help="Number of iterations for gradient accumulation", group=_OPTIMIZER)
    ] = 4
    n_save_every_itrs: Annotated[
        int,
        Parameter(
            help="Number of iterations before saving a new model checkpoint", group=_OPTIMIZER
        ),
    ] = 1_000
    disable_scheduler: Annotated[
        bool, Parameter(help="Turn off cosine learning rate scheduler", group=_OPTIMIZER)
    ] = False

    # Checkpoint
    ckptpath: Annotated[
        str | None, Parameter(help="Checkpoint of a pretrained pose regressor", group=_CHECKPOINT)
    ] = None
    reuse_optimizer: Annotated[
        bool, Parameter(help="Initialize the previous optimizer's state", group=_CHECKPOINT)
    ] = False
    warp: Annotated[
        str | None,
        Parameter(
            help="SimpleITK transform to warp input CT to checkpoint's reference frame",
            group=_CHECKPOINT,
        ),
    ] = None
    invert: Annotated[
        bool, Parameter(help="Whether to invert the warp or not", group=_CHECKPOINT)
    ] = False

    # Logging
    project: Annotated[str, Parameter(help="WandB project name", group=_LOGGING)] = "xvr"
    name: Annotated[str | None, Parameter(help="WandB run name", group=_LOGGING)] = None
    id: Annotated[
        str | None,
        Parameter(help="WandB run ID (useful when restarting from a checkpoint)", group=_LOGGING),
    ] = None
