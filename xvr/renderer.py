from diffdrr.data import read
from diffdrr.drr import DRR
from diffdrr.registration import Registration


def initialize_drr(
    volume,
    mask,
    labels,
    orientation,
    img,
    sdd,
    delx,
    dely,
    x0,
    y0,
    reverse_x_axis,
    renderer,
    read_kwargs={},
    drr_kwargs={},
):
    # Load the CT volume
    if labels is not None:
        labels = [int(x) for x in labels.split(",")]
    subject = read(volume, mask, labels, orientation, **read_kwargs)

    # Initialize the DRR module at full resolution
    *_, height, width = img.shape
    drr = DRR(
        subject,
        sdd,
        height,
        delx,
        width,
        dely,
        x0,
        y0,
        reverse_x_axis=reverse_x_axis,
        renderer=renderer,
        **drr_kwargs,
    ).cuda()

    return drr


def initialize_registration(
    volume,
    mask,
    labels,
    orientation,
    img,
    sdd,
    delx,
    dely,
    x0,
    y0,
    reverse_x_axis,
    renderer,
    init_pose,
    parameterization,
    convention,
):
    # Initialize the DRR module at full resolution
    drr = initialize_drr(
        volume,
        mask,
        labels,
        orientation,
        img,
        sdd,
        delx,
        dely,
        x0,
        y0,
        reverse_x_axis,
        renderer,
    )

    # Initialize the registration module
    rot, xyz = init_pose.convert(parameterization, convention)
    return Registration(drr, rot, xyz, parameterization, convention)
