import dataclasses
import inspect


def test_train_params_defaults_match_trainer():
    from xvr.cli.configs.train import TrainParams
    from xvr.model.trainer import Trainer

    trainer_sig = inspect.signature(Trainer.__init__)
    trainer_defaults = {
        name: param.default
        for name, param in trainer_sig.parameters.items()
        if name != "self" and param.default is not inspect.Parameter.empty
    }

    dataclass_defaults = {
        f.name: f.default if f.default is not dataclasses.MISSING else f.default_factory()
        for f in dataclasses.fields(TrainParams)
        if f.default is not dataclasses.MISSING or f.default_factory is not dataclasses.MISSING
    }

    # Fields only in CLI (logging, pose ranges, required fields) — not in Trainer
    cli_only = {"project", "name", "id", "r1", "r2", "r3", "tx", "ty", "tz", "sample_weights"}

    # Fields only in Trainer (transformed or internal) — not directly in CLI
    trainer_only = {
        "alphamin",
        "alphamax",
        "betamin",
        "betamax",
        "gammamin",
        "gammamax",
        "txmin",
        "txmax",
        "tymin",
        "tymax",
        "tzmin",
        "tzmax",
        "img_threshold",
        "mask_threshold",
        "weights",
    }

    comparable_trainer = {k: v for k, v in trainer_defaults.items() if k not in trainer_only}
    comparable_cli = {k: v for k, v in dataclass_defaults.items() if k not in cli_only}

    # Check for unexpected drift in shared keys
    shared_keys = comparable_trainer.keys() & comparable_cli.keys()
    mismatches = {
        k: (comparable_trainer[k], comparable_cli[k])
        for k in shared_keys
        if comparable_trainer[k] != comparable_cli[k]
    }
    assert not mismatches, "Default value mismatches:\n" + "\n".join(
        f"  {k}: Trainer={v[0]!r}, TrainParams={v[1]!r}" for k, v in mismatches.items()
    )

    # Check for unexpected fields in one but not the other
    unexpected_drift = (comparable_trainer.keys() - comparable_cli.keys()) | (
        comparable_cli.keys() - comparable_trainer.keys()
    )
    assert not unexpected_drift, f"Fields present in one but not the other: {unexpected_drift}"
