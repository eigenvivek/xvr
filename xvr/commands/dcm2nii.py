import click


@click.command(context_settings=dict(show_default=True, max_content_width=120))
@click.option(
    "-i",
    "--inpath",
    required=True,
    type=click.Path(exists=True),
    help="Input path to DICOMDIR for conversion",
)
@click.option(
    "-o",
    "--outpath",
    required=True,
    type=click.Path(),
    help="Savepath for the NIfTI file",
)
def dcm2nii(inpath, outpath):
    """Convert a DICOMDIR to a NIfTI file."""

    from torchio import ScalarImage

    volume = ScalarImage(inpath)
    volume.save(outpath)
