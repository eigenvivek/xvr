from pathlib import Path

import torch
from attrs import define
from jaxtyping import Float

from ..io import Intrinsics


@define
class XrayContext:
    """Everything a protocol may need to initialize a pose estimate."""

    filename: str | Path
    img: Float[torch.Tensor, "1 1 H W"]
    intrinsics: Intrinsics
