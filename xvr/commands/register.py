import click


@click.command(context_settings=dict(show_default=True, max_content_width=120))
@click.argument(
    "xray",
    nargs=-1,
    required=True,
    type=click.Path(exists=True),
)
@click.option(
    "-v",
    "--volume",
    required=True,
    type=click.Path(exists=True),
    help="Input CT volume (3D image)",
)
@click.option(
    "-m",
    "--mask",
    type=click.Path(exists=True),
    help="Labelmap for the CT volume (optional)",
)
@click.option(
    "-c",
    "--ckptpath",
    required=True,
    type=click.Path(exists=True),
    help="Checkpoint of a pretrained pose regressor",
)
@click.option(
    "-o",
    "--outpath",
    required=True,
    type=click.Path(),
    help="Directory for saving registration results",
)
@click.option(
    "--crop",
    default=0,
    type=int,
    help="Preprocessing: center crop the X-ray image",
)
@click.option(
    "--subtract_background",
    default=False,
    is_flag=True,
    help="Preprocessing: subtract mode X-ray image intensity",
)
@click.option(
    "--linearize",
    default=False,
    is_flag=True,
    help="Preprocessing: convert X-ray from exponential to linear form",
)
@click.option(
    "--warp",
    type=click.Path(exists=True),
    help="SimpleITK transform to warp input CT to template reference frame",
)
@click.option(
    "--invert",
    default=False,
    is_flag=True,
    help="Invert the warp",
)
@click.option(
    "--model_only",
    default=False,
    is_flag=True,
    help="Directly return the output of the pose regressor (no test-time optimization)",
)
@click.option(
    "--labels",
    type=str,
    help="Labels in mask to exclusively render (comma separated)",
)
@click.option(
    "--scales",
    default="8",
    type=str,
    help="Scales of downsampling for multiscale registration (comma separated)",
)
@click.option(
    "--reverse_x_axis",
    default=False,
    is_flag=True,
    help="Enable to obey radiologic convention (e.g., heart on right)",
)
@click.option(
    "--renderer",
    default="trilinear",
    type=click.Choice(["siddon", "trilinear"]),
    help="Rendering equation",
)
@click.option(
    "--parameterization",
    default="euler_angles",
    type=str,
    help="Parameterization of SO(3) for regression",
)
@click.option(
    "--convention",
    default="ZXY",
    type=str,
    help="If parameterization is Euler angles, specify order",
)
@click.option(
    "--lr_rot",
    default=1e-2,
    type=float,
    help="Initial step size for rotational parameters",
)
@click.option(
    "--lr_xyz",
    default=1e0,
    type=float,
    help="Initial step size for translational parameters",
)
@click.option(
    "--patience",
    default=10,
    type=int,
    help="Number of allowed epochs with no improvement after which the learning rate will be reduced",
)
@click.option(
    "--threshold",
    default=1e-4,
    type=float,
    help="Threshold for measuring the new optimum",
)
@click.option(
    "--max_n_itrs",
    default=500,
    type=int,
    help="Maximum number of iterations to run at each scale",
)
@click.option(
    "--max_n_plateaus",
    default=3,
    type=int,
    help="Number of times loss can plateau before moving to next scale",
)
@click.option(
    "--pattern",
    default="*.dcm",
    type=str,
    help="Pattern rule for glob is XRAY is directory",
)
def register(
    xray,
    volume,
    mask,
    ckptpath,
    outpath,
    crop,
    subtract_background,
    linearize,
    warp,
    invert,
    model_only,
    labels,
    scales,
    reverse_x_axis,
    renderer,
    parameterization,
    convention,
    lr_rot,
    lr_xyz,
    patience,
    threshold,
    max_n_itrs,
    max_n_plateaus,
    pattern,
):
    """
    Use gradient-based optimization to register XRAY to a CT.

    Can pass multiple DICOM files or a directory in XRAY.
    """
    from pathlib import Path

    from ..registrar import Registrar

    # Construct a list of all X-ray DICOMs to register
    dcmfiles = []
    for xpath in xray:
        xpath = Path(xpath)
        if xpath.is_file():
            dcmfiles.append(xpath)
        else:
            dcmfiles += sorted(xpath.glob(pattern))

    # Initialize a registration module
    registrar = Registrar(
        volume,
        mask,
        ckptpath,
        crop,
        subtract_background,
        linearize,
        warp,
        invert,
        model_only,
        labels,
        scales,
        reverse_x_axis,
        renderer,
        parameterization,
        convention,
        lr_rot,
        lr_xyz,
        patience,
        threshold,
        max_n_itrs,
        max_n_plateaus,
    )

    for i2d in dcmfiles:
        print(f"\nRegistering {i2d} ...")
        registrar(i2d, outpath)
