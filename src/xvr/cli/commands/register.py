import click


class BaseRegistrar(click.Command):
    default_context_settings = {"show_default": True, "max_content_width": 120}

    default_params = [
        click.Argument(
            ["xray"],
            nargs=-1,
            required=True,
            type=click.Path(exists=True),
        ),
        click.Option(
            ["-v", "--volume"],
            required=True,
            type=click.Path(exists=True),
            help="Input CT volume (3D image)",
        ),
        click.Option(
            ["-m", "--mask"],
            type=click.Path(exists=True),
            help="Labelmap for the CT volume (optional)",
        ),
        click.Option(
            ["-o", "--outpath"],
            required=True,
            type=click.Path(),
            help="Directory for saving registration results",
        ),
        click.Option(
            ["--crop"],
            default=0,
            type=int,
            help="Preprocessing: center crop the X-ray image",
        ),
        click.Option(
            ["--subtract_background"],
            default=False,
            is_flag=True,
            help="Preprocessing: subtract mode X-ray image intensity",
        ),
        click.Option(
            ["--linearize"],
            default=False,
            is_flag=True,
            help="Preprocessing: convert X-ray from exponential to linear form",
        ),
        click.Option(
            ["--reducefn"],
            default="max",
            help="If DICOM is multiframe, method to extract a single 2D image",
        ),
        click.Option(
            ["--labels"],
            type=str,
            help="Labels in mask to exclusively render (comma separated)",
        ),
        click.Option(
            ["--scales"],
            default="8",
            type=str,
            help="Scales of downsampling for multiscale registration (comma separated)",
        ),
        click.Option(
            ["--n_itrs"],
            default=500,
            type=str,
            help="Number of iterations to run at each scale (comma separated)",
        ),
        click.Option(
            ["--reverse_x_axis"],
            default=False,
            is_flag=True,
            help="Enable to obey radiologic convention (e.g., heart on right)",
        ),
        click.Option(
            ["--renderer"],
            default="trilinear",
            type=click.Choice(["siddon", "trilinear"]),
            help="Rendering equation",
        ),
        click.Option(
            ["--parameterization"],
            default="euler_angles",
            type=str,
            help="Parameterization of SO(3) for regression",
        ),
        click.Option(
            ["--convention"],
            default="ZXY",
            type=str,
            help="If parameterization is Euler angles, specify order",
        ),
        click.Option(
            ["--voxel_shift"],
            default=0.0,
            type=float,
            help="Position of voxel (top left corner or center)",
        ),
        click.Option(
            ["--lr_rot"],
            default=1e-2,
            type=float,
            help="Initial step size for rotational parameters",
        ),
        click.Option(
            ["--lr_xyz"],
            default=1e0,
            type=float,
            help="Initial step size for translational parameters",
        ),
        click.Option(
            ["--patience"],
            default=10,
            type=int,
            help="Number of itrs without improvement before decreasing the learning rate",
        ),
        click.Option(
            ["--threshold"],
            default=1e-4,
            type=float,
            help="Threshold for measuring the new optimum",
        ),
        click.Option(
            ["--max_n_plateaus"],
            default=3,
            type=int,
            help="Number of times loss can plateau before moving to next scale",
        ),
        click.Option(
            ["--init_only"],
            default=False,
            is_flag=True,
            help="Directly return the initial pose estimate (no iterative pose refinement)",
        ),
        click.Option(
            ["--saveimg"],
            default=False,
            is_flag=True,
            help="Save ground truth X-ray and predicted DRRs",
        ),
        click.Option(
            ["--pattern"],
            default="*.dcm",
            type=str,
            help="Pattern rule for glob is XRAY is directory",
        ),
        click.Option(
            ["--verbose"],
            default=1,
            type=click.IntRange(0, 3),
            help="Verbosity level for logging",
        ),
    ]

    def __init__(self, *args, **kwargs):
        # Apply default context settings
        kwargs["context_settings"] = self.default_context_settings.copy()

        super().__init__(*args, **kwargs)

        # Add shared parameters for all registration methods
        self.params.extend(self.default_params.copy())


@click.command(cls=BaseRegistrar)
@click.option(
    "-c",
    "--ckptpath",
    required=True,
    type=click.Path(exists=True),
    help="Checkpoint of a pretrained pose regressor",
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
def model(
    xray,
    volume,
    mask,
    outpath,
    crop,
    subtract_background,
    linearize,
    reducefn,
    labels,
    scales,
    n_itrs,
    reverse_x_axis,
    renderer,
    parameterization,
    convention,
    voxel_shift,
    lr_rot,
    lr_xyz,
    patience,
    threshold,
    max_n_plateaus,
    init_only,
    saveimg,
    pattern,
    verbose,
    ckptpath,
    warp,
    invert,
):
    """Initialize from a pose regression model."""
    from ...registrar import RegistrarModel

    registrar = RegistrarModel(
        volume,
        mask,
        ckptpath,
        labels,
        crop,
        subtract_background,
        linearize,
        reducefn,
        warp,
        invert,
        scales,
        n_itrs,
        reverse_x_axis,
        renderer,
        parameterization,
        convention,
        voxel_shift,
        lr_rot,
        lr_xyz,
        patience,
        threshold,
        max_n_plateaus,
        init_only,
        saveimg,
        verbose,
    )

    run(registrar, xray, pattern, verbose, outpath)


@click.command(cls=BaseRegistrar)
@click.option(
    "--orientation",
    type=click.Choice(["AP", "PA"]),
    help="Orientation of the C-arm",
)
def dicom(
    xray,
    volume,
    mask,
    outpath,
    crop,
    subtract_background,
    linearize,
    reducefn,
    labels,
    scales,
    n_itrs,
    reverse_x_axis,
    renderer,
    parameterization,
    convention,
    voxel_shift,
    lr_rot,
    lr_xyz,
    patience,
    threshold,
    max_n_plateaus,
    init_only,
    saveimg,
    pattern,
    verbose,
    orientation,
):
    """Initialize from pose parameters in the DICOM header."""
    from ...registrar import RegistrarDicom

    registrar = RegistrarDicom(
        volume,
        mask,
        orientation,
        labels,
        crop,
        subtract_background,
        linearize,
        scales,
        n_itrs,
        reverse_x_axis,
        renderer,
        reducefn,
        parameterization,
        convention,
        voxel_shift,
        lr_rot,
        lr_xyz,
        patience,
        threshold,
        max_n_plateaus,
        init_only,
        saveimg,
        verbose,
    )

    run(registrar, xray, pattern, verbose, outpath)


@click.command(cls=BaseRegistrar)
@click.option(
    "--orientation",
    type=click.Choice(["AP", "PA"]),
    help="Orientation of the C-arm",
)
@click.option(
    "--rot",
    type=str,
    help="Rotational parameters (comma separated); see `parameterization` and `convention`",
)
@click.option(
    "--xyz",
    type=str,
    help="Translational parameters (comma separated); see `parameterization` and `convention`",
)
def fixed(
    xray,
    volume,
    mask,
    outpath,
    crop,
    subtract_background,
    linearize,
    reducefn,
    labels,
    scales,
    reverse_x_axis,
    renderer,
    parameterization,
    convention,
    voxel_shift,
    lr_rot,
    lr_xyz,
    patience,
    threshold,
    max_n_itrs,
    max_n_plateaus,
    init_only,
    saveimg,
    pattern,
    verbose,
    orientation,
    rot,
    xyz,
):
    """Initialize from a fixed pose."""
    from ...registrar import RegistrarFixed

    rot = [float(x) for x in rot.split(",")]
    xyz = [float(x) for x in xyz.split(",")]

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
        reducefn,
        scales,
        reverse_x_axis,
        renderer,
        parameterization,
        convention,
        voxel_shift,
        lr_rot,
        lr_xyz,
        patience,
        threshold,
        max_n_itrs,
        max_n_plateaus,
        init_only,
        saveimg,
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
            print(f"\nRegistering {i2d} ....")
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
