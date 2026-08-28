import dataclasses


def _dataclass_defaults(cls):
    """Default value for every field of a dataclass that declares one."""
    defaults = {}
    for field in dataclasses.fields(cls):
        if field.default is not dataclasses.MISSING:
            defaults[field.name] = field.default
        elif field.default_factory is not dataclasses.MISSING:
            defaults[field.name] = field.default_factory()
    return defaults


def test_register_defaults_match_the_cli():
    """`Register` and `BaseParams` maintain the same defaults in two different files."""
    import attrs

    from xvr.cli.configs.register import BaseParams
    from xvr.register import Register
    from xvr.register.register import _to_list

    api = {
        (field.alias or field.name): field.default
        for field in Register.__attrs_attrs__
        if field.init and field.default is not attrs.NOTHING
    }
    cli = _dataclass_defaults(BaseParams)

    # `metric_kwargs` takes arbitrary keyword arguments, which has no CLI surface.
    api_only = set(api) - set(cli) - {"metric_kwargs"}
    assert not api_only, f"exposed on Register but not the CLI: {sorted(api_only)}"
    assert not set(cli) - set(api), (
        f"exposed on the CLI but not Register: {sorted(set(cli) - set(api))}"
    )

    # The per-scale schedules go through the `_to_list` converter, so a scalar default on
    # `Register` is the same value as the singleton list the CLI declares.
    mismatches = {
        key: (api[key], cli[key])
        for key in set(api) & set(cli)
        if _to_list(api[key]) != _to_list(cli[key])
    }
    assert not mismatches, "Default value mismatches:\n" + "\n".join(
        f"  {key}: Register={api!r}, BaseParams={cli!r}" for key, (api, cli) in mismatches.items()
    )


def test_run_parameter_defaults_match_the_cli():
    """The same contract for the per-image options on `Register.__call__` and `RunParams`."""
    import inspect

    from xvr.cli.configs.register import RunParams
    from xvr.register import Register

    signature = inspect.signature(Register.__call__)
    api = {
        name: param.default
        for name, param in signature.parameters.items()
        if param.default is not inspect.Parameter.empty
    }
    cli = _dataclass_defaults(RunParams)

    assert set(api) == set(cli), f"symmetric difference: {sorted(set(api) ^ set(cli))}"
    mismatches = {key: (api[key], cli[key]) for key in api if api[key] != cli[key]}
    assert not mismatches, "Default value mismatches:\n" + "\n".join(
        f"  {key}: __call__={a!r}, RunParams={c!r}" for key, (a, c) in mismatches.items()
    )
