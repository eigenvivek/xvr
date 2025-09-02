from diffdrr.data import read
from diffdrr.drr import DRR


def initialize_drr(
    volume,
    mask,
    labels,
    orientation,
    height,
    width,
    sdd,
    delx,
    dely,
    x0,
    y0,
    reverse_x_axis,
    renderer,
    read_kwargs={},
    drr_kwargs={},
    device="cuda",
):
    # Load the CT volume
    if labels is not None:
        labels = [int(x) for x in labels.split(",")]
    subject = read(volume, mask, labels, orientation, **read_kwargs)

    # Initialize the DRR module at full resolution
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
    ).to(device)

    return drr
