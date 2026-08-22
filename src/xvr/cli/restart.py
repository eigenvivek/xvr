from typing import Annotated

from cyclopts import App, Parameter

restart = App(name="restart", help="Restart model training from a checkpoint.")


@restart.default
def _restart(
    ckptpath: Annotated[str, Parameter(help="Checkpoint of a pretrained pose regressor.")],
    *,
    id: Annotated[str | None, Parameter(help="WandB run ID.")] = None,
    project: Annotated[str | None, Parameter(help="WandB project name.")] = None,
):
    """Restart model training from a checkpoint."""
    import os
    from pathlib import Path

    import torch
    import wandb

    from ..model import Trainer

    ckptpath = Path(ckptpath)
    if ckptpath.is_dir():
        ckptpath = sorted(ckptpath.glob("*.pth"))[-1]
    ckptpath = str(ckptpath)

    config = torch.load(ckptpath, weights_only=False)["config"]
    config["ckptpath"] = ckptpath
    config["reuse_optimizer"] = True

    wandb.login(key=os.environ["WANDB_API_KEY"])
    project = config["project"] if project is None else project
    run = wandb.init(project=project, id=id, config=config, resume="must")

    trainer = Trainer(**config)
    trainer.train(run)
