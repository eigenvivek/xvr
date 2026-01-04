import click

from ...config import register as cfg
from ..formatter import CategorizedCommand, CategorizedOption, categorized_option


class BaseRegistrar(CategorizedCommand):
    default_params = [
        click.Argument(
            ["xray"],
            nargs=-1,
            required=True,
            type=click.Path(exists=True),
        ),
        CategorizedOption(
            ["-v", "--volume"],
            required=True,
            type=click.Path(exists=True),
            help="Input CT volume (3D image)",
            category="Required",
        ),
        CategorizedOption(
            ["-m", "--mask"],
            type=click.Path(exists=True),
            help="Labelmap for the CT volume",
            category="Renderer",
        ),
        CategorizedOption(
            ["-o", "--outpath"],
            required=True,
            type=click.Path(),
            help="Directory for saving registration results",
            category="Required",
        ),
        CategorizedOption(
            ["--crop"],
            default=cfg.crop,
            type=int,
            help="Center crop the X-ray image",
            category="Preprocessing",
        ),
        CategorizedOption(
            ["--subtract_background"],
            default=cfg.subtract_background,
            is_flag=True,
            help="Subtract mode X-ray image intensity",
            category="Preprocessing",
        ),
        CategorizedOption(
            ["--linearize"],
            default=cfg.linearize,
            is_flag=True,
            help="Convert X-ray from exponential to linear form",
            category="Preprocessing",
        ),
        CategorizedOption(
            ["--equalize"],
            default=cfg.equalize,
            is_flag=True,
            help="Apply histogram equalization to X-rays/DRRs during optimization",
            category="Preprocessing",
        ),
        CategorizedOption(
            ["--reducefn"],
            default=cfg.reducefn,
            help="If DICOM is multiframe, method to extract a single 2D image",
            category="Preprocessing",
        ),
        CategorizedOption(
            ["--labels"],
            type=str,
            help="Labels in mask to exclusively render (comma-separated)",
            category="Renderer",
        ),
        CategorizedOption(
            ["--scales"],
            default=cfg.scales,
            type=str,
            help="Scales of downsampling for multiscale registration (comma-separated)",
            category="Optimizer",
        ),
        CategorizedOption(
            ["--n_itrs"],
            default=cfg.n_itrs,
            type=str,
            help="Number of iterations to run at each scale (comma-separated)",
            category="Optimizer",
        ),
        CategorizedOption(
            ["--reverse_x_axis"],
            default=cfg.reverse_x_axis,
            is_flag=True,
            help="Enable to obey radiologic convention (e.g., heart on right)",
            category="Renderer",
        ),
        CategorizedOption(
            ["--renderer"],
            default=cfg.renderer,
            type=click.Choice(["siddon", "trilinear"]),
            help="Renderer equation",
            category="Renderer",
        ),
        CategorizedOption(
            ["--parameterization"],
            default=cfg.parameterization,
            type=str,
            help="Parameterization of SO(3) for regression",
            category="Optimizer",
        ),
        CategorizedOption(
            ["--convention"],
            default=cfg.convention,
            type=str,
            help="If parameterization is Euler angles, specify order",
            category="Optimizer",
        ),
        CategorizedOption(
            ["--voxel_shift"],
            default=0.0,
            type=float,
            help="Position of voxel (top left corner or center)",
            category="Renderer",
        ),
        CategorizedOption(
            ["--lr_rot"],
            default=cfg.lr_rot,
            type=float,
            help="Initial step size for rotational parameters",
            category="Optimizer",
        ),
        CategorizedOption(
            ["--lr_xyz"],
            default=cfg.lr_xyz,
            type=float,
            help="Initial step size for translational parameters",
            category="Optimizer",
        ),
        CategorizedOption(
            ["--patience"],
            default=cfg.patience,
            type=int,
            help="Number of itrs without improvement before decreasing the learning rate",
            category="Optimizer",
        ),
        CategorizedOption(
            ["--threshold"],
            default=cfg.threshold,
            type=float,
            help="Threshold for measuring the new optimum",
            category="Optimizer",
        ),
        CategorizedOption(
            ["--max_n_plateaus"],
            default=cfg.max_n_plateaus,
            type=int,
            help="Number of times loss can plateau before moving to next scale",
            category="Optimizer",
        ),
        CategorizedOption(
            ["--init_only"],
            default=cfg.init_only,
            is_flag=True,
            help="Directly return the initial pose estimate (no iterative pose refinement)",
            category="Logging",
        ),
        CategorizedOption(
            ["--saveimg"],
            default=cfg.saveimg,
            is_flag=True,
            help="Save ground truth X-ray and predicted DRRs",
            category="Logging",
        ),
        CategorizedOption(
            ["--pattern"],
            default="*.dcm",
            type=str,
            help="Pattern rule for glob is XRAY is directory",
            category="Preprocessing",
        ),
        CategorizedOption(
            ["--verbose"],
            default=cfg.verbose,
            type=click.IntRange(0, 3),
            help="Verbosity level for logging",
            category="Logging",
        ),
    ]

    def __init__(self, *args, **kwargs):
        category_order = [
            "Required",
            "Model",
            "Renderer",
            "Preprocessing",
            "Optimizer",
            "Logging",
        ]
        super().__init__(category_order=category_order, *args, **kwargs)
        self.params.extend(self.default_params.copy())


@click.command(cls=BaseRegistrar)
@categorized_option(
    "-c",
    "--ckptpath",
    required=True,
    type=click.Path(exists=True),
    help="Checkpoint of a pretrained pose regressor",
    category="Required",
)
@categorized_option(
    "--warp",
    type=click.Path(exists=True),
    help="SimpleITK transform to warp input CT to a template reference frame",
)
@categorized_option(
    "--invert",
    default=False,
    is_flag=True,
    help="Whether to invert the warp or not",
)
@categorized_option(
    "--antipodal",
    default=False,
    is_flag=True,
    help="Initialize from antipode of predicted pose",
)
def model(
    xray,
    volume,
    mask,
    outpath,
    crop,
    subtract_background,
    linearize,
    equalize,
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
    antipodal,
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
        equalize,
        reducefn,
        warp,
        invert,
        antipodal,
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
@categorized_option(
    "--orientation",
    required=True,
    type=click.Choice(["AP", "PA"]),
    category="Required",
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
    equalize,
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
        equalize,
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
@categorized_option(
    "--orientation",
    required=True,
    type=click.Choice(["AP", "PA"]),
    category="Required",
    help="Orientation of the C-arm",
)
@categorized_option(
    "--rot",
    required=True,
    type=str,
    help="Rotation (comma-separated); see `parameterization` and `convention`",
    category="Required",
)
@categorized_option(
    "--xyz",
    required=True,
    type=str,
    help="Translation (comma-separated); see `parameterization` and `convention`",
    category="Required",
)
def fixed(
    xray,
    volume,
    mask,
    outpath,
    crop,
    subtract_background,
    linearize,
    equalize,
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
        equalize,
        reducefn,
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
@categorized_option(
    "--orientation",
    required=True,
    type=click.Choice(["AP", "PA"]),
    category="Required",
    help="Orientation of the C-arm",
)
@categorized_option(
    "-c",
    "--ckpt",
    required=True,
    type=click.Path(exists=True),
    help="Path to `parameters.pt` for previous iterative optimization run",
    category="Required",
)
def restart(
    xray,
    volume,
    mask,
    outpath,
    crop,
    subtract_background,
    linearize,
    equalize,
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
    ckpt,
):
    """Initialize from a previous final pose estimate."""
    import torch
    from diffdrr.pose import RigidTransform

    from ...registrar import RegistrarRestart

    ckpt = torch.load(ckpt, weights_only=False)
    pose = RigidTransform(ckpt["final_pose"])

    registrar = RegistrarRestart(
        volume,
        mask,
        orientation,
        pose,
        labels,
        crop,
        subtract_background,
        linearize,
        equalize,
        reducefn,
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
