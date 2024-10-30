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
    "--init_only",
    default=False,
    is_flag=True,
    help="Directly return the initial pose estimate (no iterative pose refinement)",
)
@click.option(
    "--pattern",
    default="*.dcm",
    type=str,
    help="Pattern rule for glob is XRAY is directory",
)
@click.option(
    "--verbose",
    default=2,
    type=click.IntRange(0, 3),
    help="Verbosity level for logging",
)
def model(
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
    init_only,
    pattern,
    verbose,
):
    """
    Initialize from a pose regression model.
    """
    from ..registrar import RegistrarModel

    registrar = RegistrarModel(
        volume,
        mask,
        ckptpath,
        labels,
        crop,
        subtract_background,
        linearize,
        warp,
        invert,
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
        init_only,
        verbose,
    )

    run(registrar, xray, pattern, verbose, outpath)


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
    "--orientation",
    type=click.Choice(["AP", "PA"]),
    help="Orientation of the C-arm",
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
    "--init_only",
    default=False,
    is_flag=True,
    help="Directly return the initial pose estimate (no iterative pose refinement)",
)
@click.option(
    "--pattern",
    default="*.dcm",
    type=str,
    help="Pattern rule for glob is XRAY is directory",
)
@click.option(
    "--verbose",
    default=2,
    type=click.IntRange(0, 3),
    help="Verbosity level for logging",
)
def dicom(
    xray,
    volume,
    mask,
    orientation,
    outpath,
    crop,
    subtract_background,
    linearize,
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
    init_only,
    pattern,
    verbose,
):
    """
    Initialize from the DICOM parameters.
    """
    from ..registrar import RegistrarDicom

    registrar = RegistrarDicom(
        volume,
        mask,
        orientation,
        labels,
        crop,
        subtract_background,
        linearize,
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
        init_only,
        verbose,
    )

    run(registrar, xray, pattern, verbose, outpath)


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
    "--orientation",
    type=click.Choice(["AP", "PA"]),
    help="Orientation of the C-arm",
)
@click.option(
    "--rot",
    type=str,
    help="",
)
@click.option(
    "--xyz",
    type=str,
    help="",
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
    "--init_only",
    default=False,
    is_flag=True,
    help="Directly return the initial pose estimate (no iterative pose refinement)",
)
@click.option(
    "--pattern",
    default="*.dcm",
    type=str,
    help="Pattern rule for glob is XRAY is directory",
)
@click.option(
    "--verbose",
    default=2,
    type=click.IntRange(0, 3),
    help="Verbosity level for logging",
)
def fixed(
    xray,
    volume,
    mask,
    orientation,
    rot,
    xyz,
    outpath,
    crop,
    subtract_background,
    linearize,
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
    init_only,
    pattern,
    verbose,
):
    """
    Initialize from a fixed pose.
    """

    from ..registrar import RegistrarFixed

    rot = [[float(x) for x in rot.split(",")]]
    xyz = [[float(x) for x in xyz.split(",")]]

    registrar = RegistrarFixed(
        volume,
        mask,
        orientation,
        rot,
        xyz,
        labels,
        crop,
        subtract_background,
        linearize,
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
        init_only,
        verbose,
    )

    run(registrar, xray, pattern, verbose, outpath)


def run(registrar, xray, pattern, verbose, outpath):
    from tqdm import tqdm

    dcmfiles = parse_dcmfiles(xray, pattern)
    if verbose == 0:
        dcmfiles = tqdm(dcmfiles, desc="DICOMs")

    for i2d in dcmfiles:
        if verbose > 0:
            print(f"\nRegistering {i2d} ...")
        registrar(i2d, outpath)


def parse_dcmfiles(xray, pattern):
    from pathlib import Path

    dcmfiles = []
    for xpath in xray:
        xpath = Path(xpath)
        if xpath.is_file():
            dcmfiles.append(xpath)
        else:
            dcmfiles += sorted(xpath.glob(pattern))
    return dcmfiles
