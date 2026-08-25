"""Generate the reference pages from the package itself.

The CLI reference is rendered from the live `cyclopts` app, and the API
reference from whatever modules exist under `src/xvr`, so page *contents*
cannot drift from the code. Zensical calls `define_env` at build time; nothing
generated here is checked in.

What contents alone cannot catch is a page that was never created — a new
subpackage or a new top-level command needs a page and a nav entry. Running
this module directly checks exactly that:

    python tools/macros.py

It lives outside `docs/` because everything under `docs/` is copied into the
built site. Zensical resolves `module_name` as a path below the project root,
so a subdirectory works as long as `zensical.toml` names it.
"""

import html
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
DOCS = ROOT / "docs"
CONFIG = ROOT / "zensical.toml"
PACKAGE = "xvr"

# Width the captured `--help` output is rendered at, chosen so a code block
# never needs a horizontal scrollbar. The panels span the full width, so this
# is the block's exact width, not an upper bound. The theme's content column is
# 61rem of grid minus two 12.1rem sidebars, and `html` sets 1rem to 20px; after
# the inner margins and the block's own 1.25em padding that leaves about 86
# characters of JetBrains Mono at 12.8px. Rounding down keeps a margin for
# fallback fonts, which are not all exactly 0.6em wide.
HELP_WIDTH = 84

# `cli` holds thin argument-parsing wrappers and the dataclasses behind them;
# the CLI reference already renders every one of their fields from the live
# cyclopts app, so an API page for it would document the same thing twice.
UNDOCUMENTED = {"cli"}

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def subpackages() -> list[str]:
    """Names of the subpackages under `src/xvr` that need an API page.

    Alphabetical order, minus the ones documented elsewhere (`UNDOCUMENTED`).
    """
    return sorted(
        p.name
        for p in (SRC / PACKAGE).iterdir()
        if p.is_dir()
        and (p / "__init__.py").is_file()
        and not p.name.startswith(".")
        and p.name not in UNDOCUMENTED
    )


def module_names(pkg: str) -> list[str]:
    """Dotted paths of every public module under a subpackage, depth-first.

    Skips private modules and the `.ipynb_checkpoints` directories scattered
    through `src/`. Package `__init__` files are re-export lists with no
    docstrings of their own, so they are skipped too.
    """
    modules = []
    for path in sorted((SRC / PACKAGE / pkg).rglob("*.py")):
        parts = path.relative_to(SRC).with_suffix("").parts
        if any(part.startswith((".", "_")) for part in parts):
            continue
        modules.append(".".join(parts))
    return modules


def load_app():
    """Import the `xvr` app fresh.

    `zensical serve` keeps one Python process alive across rebuilds, so an
    already-imported `xvr.cli` would shadow later edits to the CLI. Dropping
    the package from `sys.modules` first keeps the served CLI reference in
    step with the source. The CLI imports nothing heavier than cyclopts.
    """
    for name in [n for n in sys.modules if n == PACKAGE or n.startswith(f"{PACKAGE}.")]:
        del sys.modules[name]

    from xvr.cli import xvr as app

    return app


def commands() -> list[str]:
    """Names of the top-level `xvr` commands, excluding help/version flags."""
    return sorted(name for name in load_app() if not name.startswith("-"))


def subcommands(node) -> list[str]:
    """Names of a command's subcommands, excluding help/version flags."""
    return sorted(name for name in node if not name.startswith("-"))


def _tidy(text: str) -> str:
    """Drop rich's full-width padding, which would widen the block past its text."""
    return "\n".join(line.rstrip() for line in text.splitlines()).strip("\n")


def help_html(app, tokens: list[str]) -> str:
    """Capture one command's `--help`, colours and all, as inline-styled HTML."""
    from rich.console import Console
    from rich.terminal_theme import SVG_EXPORT_THEME

    console = Console(
        file=io.StringIO(),
        record=True,
        width=HELP_WIDTH,
        force_terminal=True,
        color_system="truecolor",
        highlight=False,
    )
    app.help_print(tokens, console=console)

    # Inlined styles cannot react to the theme toggle, so the palette has to work
    # on both backgrounds; this one is mid-tone, the default is not.
    return _tidy(
        console.export_html(inline_styles=True, code_format="{code}", theme=SVG_EXPORT_THEME)
    )


# ---------------------------------------------------------------------------
# Macros
# ---------------------------------------------------------------------------


def define_env(env) -> None:
    """Register the macros used by the pages under `docs/`."""

    @env.macro
    def cli(command: str | None = None, heading_level: int = 2) -> str:
        """Render `--help` for a command exactly as the terminal prints it.

        The page shows the real thing — same panels, same type metavars, same
        grouping — rather than a second rendering that can drift from it.
        Subcommands are included recursively, so `register` documents its four
        initializers without pages of their own.
        """
        app = load_app()

        def section(tokens: list[str], level: int) -> list[str]:
            name = " ".join([PACKAGE, *tokens])
            node = app
            for token in tokens:
                node = node[token]

            # Raw HTML rather than a fenced block, because the colours are
            # carried by `<span>`s. The theme's own `highlight` wrapper supplies
            # the background, padding, and horizontal scrolling.
            block = [
                f"{'#' * level} {name}",
                "",
                '<div class="highlight"><pre><code>'
                + html.escape(f"$ {name} --help")
                + "\n"
                + help_html(app, tokens)
                + "</code></pre></div>",
                "",
            ]
            for sub in subcommands(node):
                block += section([*tokens, sub], level + 1)
            return block

        return "\n".join(section([] if command is None else [command], heading_level))

    @env.macro
    def modules(pkg: str) -> list[str]:
        """Dotted paths of the modules to document for a subpackage."""
        return module_names(pkg)


# ---------------------------------------------------------------------------
# Coverage check
# ---------------------------------------------------------------------------


def _nav_targets(nav) -> list[str]:
    """Flatten the nav into the list of page paths it points at."""
    if isinstance(nav, str):
        return [nav]
    if isinstance(nav, list):
        return [target for item in nav for target in _nav_targets(item)]
    if isinstance(nav, dict):
        return [target for value in nav.values() for target in _nav_targets(value)]
    return []


def check() -> list[str]:
    """Return a list of pages the docs are missing, empty if none."""
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        import tomli as tomllib

    with CONFIG.open("rb") as f:
        config = tomllib.load(f)["project"]

    errors = []

    # Every subpackage needs a page that expands its modules
    for pkg in subpackages():
        page = DOCS / "reference" / f"{pkg}.md"
        if not page.is_file():
            errors.append(
                f"{PACKAGE}.{pkg} has no API reference page: create "
                f"docs/reference/{pkg}.md and add it to the nav in zensical.toml"
            )
        elif f'modules("{pkg}")' not in page.read_text():
            errors.append(
                f'docs/reference/{pkg}.md does not call modules("{pkg}"), so it '
                f"will not pick up new modules in {PACKAGE}/{pkg}"
            )

    # Every top-level command needs a page that renders it
    pages = [p.read_text() for p in (DOCS / "cli").glob("*.md")]
    for name in commands():
        if not any(f'cli("{name}")' in text for text in pages):
            errors.append(
                f"`xvr {name}` is not documented: add a page under docs/cli "
                f'containing {{{{ cli("{name}") }}}} and add it to the nav'
            )

    # Every nav entry must resolve to a page that exists
    for target in _nav_targets(config.get("nav", [])):
        if not (DOCS / target).is_file():
            errors.append(f"nav points at a missing page: docs/{target}")

    return errors


def main() -> int:
    """Report missing pages, exiting non-zero if there are any."""
    errors = check()
    if errors:
        print("Documentation is out of date:\n", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print("Documentation covers every subpackage and command.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
