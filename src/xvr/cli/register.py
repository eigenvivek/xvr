from dataclasses import asdict
from pathlib import Path
from typing import Annotated

from cyclopts import App, Group, Parameter

from .configs.register import _DATA, BaseParams, RunParams

register = App(name="register", help="Use gradient-based optimization to register XRAY to a CT/MR.")

_POSE = Group("POSE", sort_key=1)
_MODEL = Group("MODEL", sort_key=1)


@register.command
def model(
    files: Annotated[list[Path], Parameter(help="X-ray images to register")],
    ckpt: Annotated[str, Parameter(help="Path to model checkpoint", group=_MODEL)],
    imagepath: Annotated[str, Parameter(help="Path to the CT image", group=_DATA)],
    *,
    base: BaseParams = BaseParams(),
    run: RunParams = RunParams(),
):
    """Register using a neural network initial pose estimate."""
    from ..register import RegisterModel

    reg = RegisterModel(ckpt=ckpt, imagepath=imagepath, **asdict(base))
    for f in files:
        reg(str(f), **asdict(run))


@register.command
def fixed(
    files: Annotated[list[Path], Parameter(help="X-ray images to register")],
    imagepath: Annotated[str, Parameter(help="Path to the CT image", group=_DATA)],
    rot: Annotated[
        tuple[float, float, float], Parameter(help="Rotation angles in degrees", group=_POSE)
    ],
    xyz: Annotated[tuple[float, float, float], Parameter(help="Translation in mm", group=_POSE)],
    *,
    base: BaseParams = BaseParams(),
    run: RunParams = RunParams(),
    orientation: Annotated[
        str, Parameter(help="Starting orientation (AP, PA, or None)", group=_POSE)
    ] = "AP",
    isocenter: Annotated[
        bool, Parameter(help="Center pose at subject isocenter", group=_POSE)
    ] = True,
):
    """Register using a fixed initial pose."""
    from ..register import RegisterFixed

    reg = RegisterFixed(imagepath=imagepath, **asdict(base))
    for f in files:
        reg(
            str(f),
            **asdict(run),
            rot=rot,
            xyz=xyz,
            orientation=orientation,
            isocenter=isocenter,
        )
