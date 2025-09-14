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
    "--name",
    default=None,
    type=str,
    help="WandB run name",
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
    name,
    project,
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
