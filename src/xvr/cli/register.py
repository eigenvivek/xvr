from collections.abc import Callable
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
    from ..register import RegisterModel

    _run_registration(RegisterModel, base, run, init_kwargs={"ckpt": ckpt})


@register.command
def fixed(
    rot: Annotated[tuple[float, float, float], Parameter(help="Rotations in degrees", group=_POSE)],
    xyz: Annotated[tuple[float, float, float], Parameter(help="Translations in mm", group=_POSE)],
    *,
    base: BaseParams,
    run: RunParams = RunParams(),
) -> None:
    """Register using a fixed initial pose."""
    from ..register import RegisterFixed

    _run_registration(RegisterFixed, base, run, run_kwargs={"rot": rot, "xyz": xyz})


def _run_registration(
    registrator: Callable[..., Any],
    base: BaseParams,
    run: RunParams,
    init_kwargs: dict[str, Any] = {},
    run_kwargs: dict[str, Any] = {},
) -> None:
    """Helper to run registration on multiple files."""
    base_dict = asdict(base)
    files = _expand_files(base_dict.pop("files"))

    reg = registrator(**base_dict, **init_kwargs)
    for f in files:
        print(f"Registering {f}")
        reg(str(f), **asdict(run), **run_kwargs)


def _expand_files(files: list[Path]) -> list[Path]:
    """Expand any folders into their *.dcm contents."""
    out: list[Path] = []
    for f in files:
        out.extend(sorted(f.glob("*.dcm")) if f.is_dir() else [f])
    return out
