import click

from ..formatter import CategorizedCommand, categorized_option


@click.command(cls=CategorizedCommand)
@categorized_option(
    "-c",
    "--ckptpath",
    required=True,
    type=click.Path(exists=True),
    help="Checkpoint of a pretrained pose regressor",
)
@categorized_option(
    "--rescale",
    default=1.0,
    type=float,
    help="Rescale the virtual detector plane",
)
@categorized_option(
    "--name",
    default=None,
    type=str,
    help="WandB run name",
)
@categorized_option(
    "--project",
    type=str,
    default=None,
    help="WandB project name",
)
def restart(
    ckptpath: str,
    rescale: float,
    name: str,
    project: str,
):
    """
    Restart model training from a checkpoint.
    """
    import os

    import torch
    import wandb

    from ...model import Trainer

    # Load the config from the previous model checkpoint
    config = torch.load(ckptpath, weights_only=False)["config"]
    config["ckptpath"] = ckptpath
    config["reuse_optimizer"] = True

    # Rescale the detector plane
    config["batch_size"] = int(config["batch_size"] / (rescale**2))
    config["height"] = int(config["height"] * rescale)
    config["delx"] /= rescale

    # Set up logging
    project = config["project"] if project is None else project
    name += f"-rescale{rescale}" if rescale != 1 else ""
    wandb.login(key=os.environ["WANDB_API_KEY"])
    run = wandb.init(project=project, name=name, config=config)

    # Train the model
    trainer = Trainer(**config)
    trainer.train(run)
