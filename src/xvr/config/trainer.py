from dataclasses import dataclass


@dataclass
class TrainerArgs:
    """Default arguments for training."""

    # Renderer
    orientation: str = "AP"
    reverse_x_axis: bool = False

    # Model
    model_name: str = "resnet18"
    norm_layer: str = "groupnorm"
    pretrained: bool = False
    parameterization: str = "quaternion_adjugate"
    convention: str = "ZXY"
    unit_conversion_factor: float = 1000.0
    p_augmentation: float = 0.333

    # Optimizer
    lr: float = 2e-4
    weight_ncc: float = 1e0
    weight_geo: float = 1e-2
    weight_dice: float = 1e0
    batch_size: int = 116
    n_total_itrs: int = 1_000_000
    n_warmup_itrs: int = 1_000
    n_grad_accum_itrs: int = 4
    n_save_every_itrs: int = 1_000
    disable_scheduler: bool = False

    # Checkpoint
    reuse_optimizer: bool = False
    invert: bool = False

    # Data
    num_workers: int = 4
    pin_memory: bool = False

    # Logging
    project: str = "xvr"


args = TrainerArgs()
