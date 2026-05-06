from dataclasses import dataclass, field
from typing import Annotated

from cyclopts import Group, Parameter


def _non_empty(type_, value):
    if value is not None and len(value) == 0:
        raise ValueError("Must provide at least one value")


_DATA = Group("Data", sort_key=2)
_OPTIMIZER = Group("Optimizer", sort_key=3)
_PREPROCESSING = Group("Preprocessing", sort_key=4)
_MISC = Group("Miscellaneous", sort_key=5)


@Parameter(name="*")
@dataclass
class BaseParams:
    labelpath: Annotated[str | None, Parameter(help="Path to the segmentation label map. If None, uses the full image", group=_DATA)] = None
    labels: Annotated[list[int] | None, Parameter(help="Label indices to include in the DRR. If None, uses all labels", group=_DATA, validator=_non_empty)] = None
    metric: Annotated[str, Parameter(help="Image similarity metric", group=_OPTIMIZER)] = "gmncc"
    scales: Annotated[list[float], Parameter(help="Downsampling scale(s) for multiscale registration", group=_OPTIMIZER, validator=_non_empty)] = field(default_factory=lambda: [8.0])
    n_itrs: Annotated[list[int], Parameter(help="Number of optimization iterations per scale", group=_OPTIMIZER, validator=_non_empty)] = field(default_factory=lambda: [500])
    lr_rot: Annotated[float, Parameter(help="Learning rate for rotation parameters", group=_OPTIMIZER)] = 1e-2
    lr_xyz: Annotated[float, Parameter(help="Learning rate for translation parameters", group=_OPTIMIZER)] = 1e-0
    lr_reduce_factor: Annotated[float, Parameter(help="Factor by which to reduce the learning rate on plateau", group=_OPTIMIZER)] = 0.1
    patience: Annotated[int, Parameter(help="Number of steps with no improvement before reducing the learning rate", group=_OPTIMIZER)] = 5
    threshold: Annotated[float, Parameter(help="Minimum change to qualify as an improvement", group=_OPTIMIZER)] = 1e-4
    max_n_plateaus: Annotated[int, Parameter(help="Number of learning rate reductions before early stopping", group=_OPTIMIZER)] = 2
    device: Annotated[str, Parameter(help="Torch device to run on", group=_MISC)] = "cuda"


@Parameter(name="*")
@dataclass
class RunParams:
    crop: Annotated[int, Parameter(help="Number of pixels to crop from the image border", group=_PREPROCESSING)] = 0
    linearize: Annotated[bool, Parameter(help="Convert image to linear attenuation values", group=_PREPROCESSING)] = True
    subtract_background: Annotated[bool, Parameter(help="Subtract background from the image", group=_PREPROCESSING)] = False
    equalize: Annotated[bool, Parameter(help="Apply histogram equalization during optimization", group=_PREPROCESSING)] = False
    reducefn: Annotated[str, Parameter(help="Reduction function for multi-frame images", group=_PREPROCESSING)] = "max"
    reverse_x_axis: Annotated[bool, Parameter(help="Flip the image horizontally", group=_PREPROCESSING)] = False
    savepath: Annotated[str | None, Parameter(help="Location to save the registration results and an optimization GIF", group=_MISC)] = None
