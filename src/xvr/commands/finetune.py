import click


@click.command(context_settings=dict(show_default=True, max_content_width=120))
@click.option(
    "-i",
    "--inpath",
    required=True,
    type=click.Path(exists=True),
    help="Input CT volume for patient-specific pretraining",
)
@click.option(
    "-o",
    "--outpath",
    required=True,
    type=click.Path(),
    help="Output directory for finetuned model weights",
)
@click.option(
    "-c",
    "--ckptpath",
    required=True,
    type=click.Path(exists=True),
    help="Checkpoint of a pretrained pose regressor",
)
@click.option(
    "--lr",
    default=5e-3,
    type=float,
    help="Maximum learning rate",
)
@click.option(
    "--batch_size",
    default=116,
    type=int,
    help="Number of DRRs per batch",
)
@click.option(
    "--n_epochs",
    default=10,
    type=int,
    help="Number of epochs",
)
@click.option(
    "--n_batches_per_epoch",
    default=25,
    type=int,
    help="Number of batches per epoch",
)
@click.option(
    "--rescale",
    default=1.0,
    type=float,
    help="Rescale the virtual detector plane",
)
@click.option(
    "--name",
    default=None,
    type=str,
    help="WandB run name",
)
@click.option(
    "--project",
    type=str,
    default="xvr",
    help="WandB project name",
)
def finetune(
    inpath,
    outpath,
    ckptpath,
    lr,
    batch_size,
    n_epochs,
    n_batches_per_epoch,
    rescale,
    name,
    project,
):
    """
    Optimize a pose regression model for a specific patient.
    """
    import os
    from pathlib import Path

    import torch

    import wandb

    # Create the output directory for saving model weights
    Path(outpath).mkdir(parents=True, exist_ok=True)

    # Load the previous model checkpoint
    ckpt = torch.load(ckptpath, weights_only=False)
    model_state_dict = ckpt["model_state_dict"]

    # Overwrite the config with the new parameters
    config = ckpt["config"]
    config["inpath"] = inpath
    config["outpath"] = outpath
    config["ckptpath"] = ckptpath
    config["lr"] = lr
    config["batch_size"] = batch_size
    config["n_epochs"] = n_epochs
    config["n_batches_per_epoch"] = n_batches_per_epoch
    config["rescale"] = rescale
    config["height"] = int(config["height"] * rescale)
    config["delx"] /= rescale

    # Set up logging and fine-tune the model
    wandb.login(key=os.environ["WANDB_API_KEY"])
    run = wandb.init(
        project=project,
        name=name if name is not None else project, 
        config=config,
    )
    train_model(config, model_state_dict, run)


def train_model(config, model_state_dict, run):
    from datetime import datetime

    import torch
    from diffdrr.data import read
    from diffdrr.metrics import (
        DoubleGeodesicSE3,
        MultiscaleNormalizedCrossCorrelation2d,
    )
    from timm.utils.agc import adaptive_clip_grad as adaptive_clip_grad_
    from tqdm import tqdm

    import wandb

    from ..utils import XrayAugmentations, get_random_pose, render

    # Load the subject-specific CT volume
    subject = read(config["inpath"], orientation=config["orientation"])
    volume = subject.volume.data.squeeze().to(device="cuda", dtype=torch.float32)

    # Initialize deep learning modules
    model, drr, transforms, optimizer, scheduler = initialize(
        config, model_state_dict, subject
    )

    # Initialize the loss function
    imagesim = MultiscaleNormalizedCrossCorrelation2d([None, 9], [0.5, 0.5])
    geodesic = DoubleGeodesicSE3(config["sdd"])

    # Set up augmentations
    contrast_distribution = torch.distributions.Uniform(1.0, 10.0)
    augmentations = XrayAugmentations()

    # Train the model
    for epoch in range(config["n_epochs"]):
        for _ in tqdm(range(config["n_batches_per_epoch"]), desc=f"Epoch {epoch}"):
            # Sample a batch of random poses
            pose = get_random_pose(config).cuda()

            # Render random DRRs and apply transforms
            contrast = contrast_distribution.sample().item()
            img, _, _ = render(drr, pose, volume, contrast)
            with torch.no_grad():
                img = augmentations(img)
            img = transforms(img)

            # Regress the poses and render the predicted DRRs
            pred_pose = model(img)
            pred_img, _, _ = render(drr, pred_pose, volume, contrast)

            # Compute the loss
            mncc = imagesim(img, pred_img)
            rgeo, tgeo, dgeo = geodesic(pose, pred_pose)
            loss = 1 - mncc + config["weight_geo"] * dgeo

            # Optimize the model
            optimizer.zero_grad()
            loss.mean().backward()
            adaptive_clip_grad_(model.parameters())
            optimizer.step()
            scheduler.step()

            # Log metrics
            wandb.log(
                {
                    "mncc": mncc.mean().item(),
                    "dgeo": dgeo.mean().item(),
                    "rgeo": rgeo.mean().item(),
                    "tgeo": tgeo.mean().item(),
                    "loss": loss.mean().item(),
                    "lr": scheduler.get_last_lr()[0],
                }
            )

        # Checkpoint the model every epoch
        if epoch % 1 == 0:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                    "date": datetime.now(),
                    "config": config,
                },
                f"{config['outpath']}/{run.name}_{epoch:04d}.pth",
            )


def initialize(config, model_state_dict, subject):
    import torch
    from diffdrr.drr import DRR
    from diffdrr.registration import PoseRegressor

    from ..utils import XrayTransforms

    # Load the pretrained pose regression model
    model = PoseRegressor(
        model_name=config["model_name"],
        pretrained=config["pretrained"],
        parameterization=config["parameterization"],
        convention=config["convention"],
        norm_layer=config["norm_layer"],
        height=config["height"],
    ).cuda()
    model.load_state_dict(model_state_dict)
    model.train()

    # Initialize the subject-specific DRR module
    drr = DRR(
        subject,
        sdd=config["sdd"],
        height=config["height"],
        delx=config["delx"],
        reverse_x_axis=config["reverse_x_axis"],
        renderer=config["renderer"],
    ).cuda()
    transforms = XrayTransforms(drr.detector.height)
    print(drr.detector.height)

    # Set up the optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
    )
    scheduler = torch.optim.lr_scheduler.PolynomialLR(
        optimizer, total_iters=config["n_epochs"] * config["n_batches_per_epoch"]
    )

    return model, drr, transforms, optimizer, scheduler
