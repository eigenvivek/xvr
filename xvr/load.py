import torch

from xvr.dicom import read_xray
from xvr.model import load_model
from xvr.renderer import initialize_drr


def load(ckptpath, read_kwargs={}, drr_kwargs={}):
    # Load the model
    ckpt = torch.load(ckptpath, weights_only=False)
    args = ckpt["arguments"]
    model, config = load_model(args["ckptpath"])

    # Load the ground truth image
    img, *intrinsics = read_xray(
        args["i2d"],
        args["crop"],
        args["subtract_background"],
        args["linearize"],
    )

    # Load the DRR module
    drr = initialize_drr(
        args["volume"],
        args["mask"],
        args["labels"],
        config["orientation"],
        img,
        *intrinsics,
        config["reverse_x_axis"],
        config["renderer"],
        read_kwargs,
        drr_kwargs,
    )

    return model, config, drr, img, intrinsics
