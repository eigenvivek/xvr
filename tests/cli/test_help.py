import subprocess
import sys
import typing
from pathlib import Path

import pytest

from xvr.cli._help import _metavar


@pytest.mark.parametrize(
    "annotation, expected",
    [
        (None, ""),  # command rows carry no type
        (str, "TEXT"),
        (int, "INTEGER"),
        (float, "FLOAT"),
        (bool, ""),  # a flag takes no argument
        (Path, "PATH"),
        (str | None, "TEXT"),  # Optional unwraps to its one real member
        (int | str, "TEXT"),  # a genuine union degrades to TEXT
        (list[int], "INTEGER..."),
        (list[Path], "PATH..."),
        (tuple[float, float, float], "<FLOAT FLOAT FLOAT>..."),
        (tuple[int, ...], "INTEGER..."),  # variadic is a collection, not a fixed slot list
        (typing.Literal["a", "b"], "TEXT"),
        (typing.Literal[1, 2], "INTEGER"),
        (typing.Annotated[float, "ignored"], "FLOAT"),
        (typing.Annotated[list[float], "ignored"], "FLOAT..."),
    ],
)
def test_metavar_rendering(annotation, expected):
    """`_help` reproduces Click's metavar column; these are the shapes the CLI actually uses."""
    assert _metavar(annotation) == expected


@pytest.mark.parametrize(
    "argv",
    [
        ["--help"],
        ["train", "--help"],
        ["restart", "--help"],
        ["register", "--help"],
        ["register", "fixed", "--help"],
        ["register", "model", "--help"],
        ["register", "dicom", "--help"],
        ["register", "restart", "--help"],
    ],
)
def test_help_renders_for_every_command(argv):
    """Help is the one code path every user hits, and a formatter crash would break all of it."""
    result = subprocess.run(
        [sys.executable, "-m", "xvr", *argv], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert "Usage:" in result.stdout


def test_fixed_length_vectors_do_not_get_a_negative_flag():
    """`negative_iterable=()` suppresses cyclopts' `--empty-rot`, which makes no sense here."""
    result = subprocess.run(
        [sys.executable, "-m", "xvr", "register", "fixed", "--help"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert "--empty-" not in result.stdout


def test_importing_the_cli_does_not_import_torch():
    """Every heavy import in `cli/` is function-local on purpose: torch costs ~3s per call."""
    code = "import xvr.cli, sys; print('torch' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"
