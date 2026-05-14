from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any

from cyclopts import App, Group, Parameter

from .configs.register import BaseParams, RunParams

register = App(name="register", help="Use gradient-based optimization to register XRAY to a CT/MR.")

_POSE = Group("POSE", sort_key=1)
_MODEL = Group("MODEL", sort_key=1)


@register.command
def model(
    ckpt: Annotated[str, Parameter(help="Path to model checkpoint", group=_MODEL)],
    *,
    base: BaseParams,
    run: RunParams = RunParams(),
) -> None:
    """Register using a neural network initial pose estimate."""
    from ..register import ModelPose

    initializer = ModelPose(ckpt=ckpt, device=base.device)
    _run_registration(initializer, base, run)


@register.command
def fixed(
    rot: Annotated[tuple[float, float, float], Parameter(help="Rotations in degrees", group=_POSE)],
    xyz: Annotated[tuple[float, float, float], Parameter(help="Translations in mm", group=_POSE)],
    *,
    orientation: Annotated[
        str, Parameter(help="Patient orientation for the DRR", group=_POSE)
    ] = "AP",
    reverse_x_axis: Annotated[
        bool, Parameter(help="Horizontally flip the rendered DRRs", group=_POSE)
    ] = False,
    base: BaseParams,
    run: RunParams = RunParams(),
) -> None:
    """Register using a fixed initial pose."""
    from ..register import FixedPose

    initializer = FixedPose(
        rot=rot,
        xyz=xyz,
        orientation=orientation,
        reverse_x_axis=reverse_x_axis,
        device=base.device,
    )
    _run_registration(initializer, base, run)


@register.command
def dicom(
    *,
    orientation: Annotated[
        str, Parameter(help="Patient orientation for the DRR", group=_POSE)
    ] = "AP",
    reverse_x_axis: Annotated[
        bool, Parameter(help="Horizontally flip the rendered DRRs", group=_POSE)
    ] = False,
    base: BaseParams,
    run: RunParams = RunParams(),
) -> None:
    """Register using an initial pose parsed from DICOM metadata."""
    from ..register import DicomPose

    initializer = DicomPose(
        orientation=orientation,
        reverse_x_axis=reverse_x_axis,
        device=base.device,
    )
    _run_registration(initializer, base, run)


def _run_registration(
    initializer: Any,
    base: BaseParams,
    run: RunParams,
) -> None:
    """Helper to run registration on multiple files."""
    from ..register import Register

    base_dict = asdict(base)
    files = _expand_files(base_dict.pop("files"))

    reg = Register(initializer=initializer, **base_dict)
    for f in files:
        print(f"Registering {f}")
        reg(str(f), **asdict(run))


def _expand_files(files: list[Path]) -> list[Path]:
    """Expand any folders into their *.dcm contents."""
    out: list[Path] = []
    for f in files:
        out.extend(sorted(f.glob("*.dcm")) if f.is_dir() else [f])
    return out
