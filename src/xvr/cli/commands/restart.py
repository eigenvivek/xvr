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
    "--id",
    default=None,
    type=str,
    help="WandB run ID",
)
@categorized_option(
    "--project",
    type=str,
    default=None,
    help="WandB project name",
)
def restart(
    ckptpath: str,
    id: str,
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

    # Set up logging
    wandb.login(key=os.environ["WANDB_API_KEY"])
    project = config["project"] if project is None else project
    run = wandb.init(project=project, id=id, config=config, resume="must")

    # Train the model
    trainer = Trainer(**config)
    trainer.train(run)
