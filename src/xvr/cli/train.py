from dataclasses import asdict

from cyclopts import App

from .configs.train import TrainParams

train = App(name="train", help="Train a pose regression model.")


@train.default
def _train(params: TrainParams):
    """Train a pose regression model."""
    import os
    from pathlib import Path

    import wandb

    from ..model import Trainer

    Path(params.outpath).mkdir(parents=True, exist_ok=True)

    config = asdict(params)

    # Unpack pose ranges
    for key, (lo, hi) in zip(
        ("r1", "r2", "r3", "tx", "ty", "tz"),
        (params.r1, params.r2, params.r3, params.tx, params.ty, params.tz),
    ):
        prefix = {"r1": "alpha", "r2": "beta", "r3": "gamma"}.get(key, key)
        config.pop(key)
        config[f"{prefix}min"], config[f"{prefix}max"] = lo, hi

    # Parse ckptpath
    ckptpath = params.ckptpath
    if ckptpath is not None:
        ckptpath = Path(ckptpath)
        if ckptpath.is_dir():
            ckptpath = max(ckptpath.glob("*.pth"), key=lambda p: p.name)
        config["ckptpath"] = str(ckptpath)

    # Parse patch_size
    config["patch_size"] = (
        tuple(int(x) for x in params.patch_size.split(","))
        if params.patch_size is not None
        else None
    )

    # Parse sample weights
    config["weights"] = (
        [float(line) for line in Path(params.sample_weights).read_text().splitlines()]
        if params.sample_weights is not None
        else None
    )
    config.pop("sample_weights")

    # Strip logging fields not needed by Trainer
    project = config.pop("project")
    name = config.pop("name")
    id_ = config.pop("id")

    wandb.login(key=os.environ["WANDB_API_KEY"])
    run = wandb.init(
        project=project,
        name=name if name is not None else project,
        config=config,
        id=id_,
        resume="must" if id_ is not None else None,
    )

    trainer = Trainer(**config)
    trainer.train(run)
