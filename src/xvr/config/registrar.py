from dataclasses import dataclass


@dataclass
class RegistrarArgs:
    """Default arguments for registration."""

    # Preprocessing
    crop: int = 0
    subtract_background: bool = False
    linearize: bool = False
    equalize: bool = False
    reducefn: str = "max"
    pattern: str = "*.dcm"

    # Renderer
    reverse_x_axis: bool = False
    renderer: str = "trilinear"
    voxel_shift: float = 0.0

    # Optimizer
    scales: str = "8"
    n_itrs: str = "500"
    parameterization: str = "euler_angles"
    convention: str = "ZXY"
    lr_rot: float = 1e-2
    lr_xyz: float = 1e0
    patience: int = 10
    threshold: float = 1e-4
    max_n_plateaus: int = 3

    # Logging
    init_only: bool = False
    saveimg: bool = False
    verbose: int = 1


args = RegistrarArgs()
