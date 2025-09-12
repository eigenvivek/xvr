import click


@click.command(context_settings=dict(show_default=True, max_content_width=120))
@click.option(
    "-c",
    "--ckptpath",
    required=True,
    type=click.Path(exists=True),
    help="Checkpoint of a pretrained pose regressor",
)
@click.option(
    "--rescale",
    default=1.0,
    type=float,
    help="Rescale the virtual detector plane",
)
@click.option(
    "--project",
    type=str,
    default=None,
    help="WandB project name",
)
def restart(
    ckptpath,
    rescale,
    project,
):
    """
    Restart model training from a checkpoint.
    """
    import os

    import torch
    import wandb

    from ..model import Trainer

    # Load the config from the previous model checkpoint
    config = torch.load(ckptpath, weights_only=False)["config"]
    config["ckptpath"] = ckptpath

    # Rescale the detector plane
    config["batch_size"] = int(config["batch_size"] / (rescale**2))
    config["height"] = int(config["height"] * rescale)
    config["delx"] /= rescale

    # Set up logging
    addendum = f"-rescale{rescale}" if rescale != 1 else ""
    project = config["project"] if project is None else project
    name = ckptpath.split("/")[-1].split("_")[0] + addendum
    wandb.login(key=os.environ["WANDB_API_KEY"])
    run = wandb.init(project=project, name=name, config=config)

    # Train the model
    trainer = Trainer(**config)
    trainer.train(run)
