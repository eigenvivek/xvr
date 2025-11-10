import click

from ..formatter import CategorizedCommand


@click.command(cls=CategorizedCommand)
@click.argument("inpath", type=click.Path(exists=True))
@click.argument("outpath", type=click.Path())
def dcm2nii(inpath, outpath):
    """Convert a DICOMDIR to a NIfTI file."""

    from torchio import ScalarImage

    click.echo(f"Converting {inpath} to {outpath}")

    volume = ScalarImage(inpath)
    volume.save(outpath)
