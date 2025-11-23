import click

from ..formatter import CategorizedCommand, categorized_option


@click.command(cls=CategorizedCommand)
@categorized_option(
    "-i",
    "--inpath",
    required=True,
    type=click.Path(exists=True),
    help="Saved registration result from <xvr register>",
)
@categorized_option(
    "-o",
    "--outpath",
    required=True,
    type=click.Path(),
    help="Savepath for iterative optimization animation",
)
@categorized_option(
    "--skip",
    default=1,
    type=int,
    help="Animate every <skip> frames of the optimization",
)
@categorized_option(
    "--dpi",
    default=192,
    type=int,
    help="DPI of individual animation frames",
)
@categorized_option(
    "--fps",
    default=30,
    type=int,
    help="FPS of animation",
)
def animate(inpath, outpath, skip, dpi, fps):
    """Animate the trajectory of iterative optimization."""

    from ...visualization import animate as _animate

    _animate(inpath, outpath, skip, dpi, fps)
