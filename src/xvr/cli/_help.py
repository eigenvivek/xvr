import types
import typing
from pathlib import Path

from attrs import evolve
from cyclopts.help.formatters import DefaultFormatter
from cyclopts.help.specs import (
    AsteriskColumn,
    ColumnSpec,
    DescriptionColumn,
    NameRenderer,
    TableSpec,
    get_default_parameter_columns,
)
from rich.text import Text

# Beyond this many, the choice list is dropped from `--help` and only the default
# is shown. Passing an invalid value still names every choice, and the reference
# pages always list them in full.
_MAX_CHOICES = 4

# Anything longer wraps onto its own line, as Click did. Wider than this and
# the descriptions get squeezed at 80 columns.
_NAME_WIDTH = 30

# A flag takes no argument, so bool deliberately maps to an empty metavar.
_SCALARS: dict[type, str] = {str: "TEXT", int: "INTEGER", float: "FLOAT", bool: "", Path: "PATH"}


def _metavar(annotation) -> str:
    """Render an annotation the way Click rendered its metavars."""
    # Command entries carry no type; they share these columns with parameters.
    if annotation is None:
        return ""

    # Cyclopts hands back the raw annotation, which may still be wrapped.
    while hasattr(annotation, "__metadata__"):
        annotation = annotation.__origin__

    origin, args = typing.get_origin(annotation), typing.get_args(annotation)

    if origin is typing.Literal:
        # Choices are already spelled out in the description column.
        return _metavar(type(args[0])) if args else "TEXT"
    if origin in (typing.Union, types.UnionType):
        optional = [arg for arg in args if arg is not type(None)]
        return _metavar(optional[0]) if len(optional) == 1 else "TEXT"
    if origin is tuple and args and Ellipsis not in args:
        # Fixed-length: name every slot, e.g. tuple[float, float].
        return "<" + " ".join(_metavar(arg) for arg in args) + ">..."
    if origin in (list, set, frozenset, tuple):
        return (_metavar(args[0]) if args else "TEXT") + "..."

    return _SCALARS.get(annotation, getattr(annotation, "__name__", "TEXT").upper())


def _is_command_panel(entries) -> bool:
    """Whether a panel lists commands rather than parameters.

    Command entries carry no type, and no required marker, so they keep the
    cyclopts defaults instead of the parameter layout below.
    """
    return all(entry.type is None for entry in entries)


def _columns(console, options, entries):
    """Build the three columns: required marker, name + metavar, description.

    The metavar goes inside the name cell rather than in a column of its own. A
    separate column would reserve its width in every panel, and at 80 columns
    that comes straight out of the descriptions.
    """
    default = get_default_parameter_columns(console, options, entries)

    if _is_command_panel(entries):
        return default

    names = NameRenderer(max_width=_NAME_WIDTH)

    def render(entry):
        rendered = names(entry)
        text = rendered if isinstance(rendered, Text) else Text.from_markup(str(rendered))
        metavar = _metavar(entry.type)
        if not metavar:
            return text
        text = text.copy()
        text.append(f" {metavar}", style="dim")
        return text

    def describe(entry):
        choices = entry.choices
        if choices and len(choices) > _MAX_CHOICES:
            entry = evolve(entry, choices=None)
        return DescriptionColumn.renderer(entry)

    return (
        AsteriskColumn,
        ColumnSpec(
            renderer=render,
            header="Option",
            justify="left",
            style="cyan",
            width=_NAME_WIDTH,
        ),
        evolve(default[-1], renderer=describe),
    )


class _FlatFormatter(DefaultFormatter):
    """Render each panel as a left-aligned block instead of a rounded box.

    Only panel rendering changes; the usage line and description keep the
    inherited behaviour, which is what supplies the `Usage: ` prefix.
    """

    def __call__(self, console, options, panel) -> None:
        if not panel.entries:
            return

        console.print(Text(f"{panel.title}:", style="bold"))

        description = panel.description
        if description is not None and getattr(description, "plain", ""):
            console.print(description)
            console.print()

        # A command panel has no marker column, so one space would leave its
        # longest name touching its description.
        gap = 2 if _is_command_panel(panel.entries) else 1
        console.print(
            TableSpec(padding=(0, gap, 0, 0), pad_edge=False).build(
                _columns(console, options, panel.entries), panel.entries
            )
        )
        console.print()


formatter = _FlatFormatter()
