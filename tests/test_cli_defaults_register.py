import dataclasses
import inspect


def test_register_params_defaults_match_register_base():
    from xvr.cli.configs.register import BaseParams, RunParams
    from xvr.cli.register import fixed as fixed_cmd
    from xvr.register.base import RegisterBase

    init_sig = inspect.signature(RegisterBase.__init__)
    call_sig = inspect.signature(RegisterBase.__call__)
    fixed_impl_sig = inspect.signature(RegisterBase._registry["fixed"].get_initial_pose_estimate)
    fixed_cmd_sig = inspect.signature(fixed_cmd)

    def extract_sig_defaults(sig, exclude=()):
        defaults = {
            name: param.default
            for name, param in sig.parameters.items()
            if name not in {"self", "kwargs", *exclude}
            and param.default is not inspect.Parameter.empty
        }
        # RegisterBase.__init__ accepts scalar or list for these; BaseParams always uses list
        for key in ("scales", "n_itrs"):
            if key in defaults and not isinstance(defaults[key], list):
                defaults[key] = [defaults[key]]
        return defaults

    def extract_dataclass_defaults(cls):
        result = {}
        for f in dataclasses.fields(cls):
            if f.default is not dataclasses.MISSING:
                result[f.name] = f.default
            elif f.default_factory is not dataclasses.MISSING:
                result[f.name] = f.default_factory()
        return result

    def assert_in_sync(left, right, label):
        mismatches = {
            k: (left[k], right[k]) for k in left.keys() & right.keys() if left[k] != right[k]
        }
        assert not mismatches, f"[{label}] Default value mismatches:\n" + "\n".join(
            f"  {k}: left={v[0]!r}, right={v[1]!r}" for k, v in mismatches.items()
        )
        drift = left.keys() ^ right.keys()
        assert not drift, f"[{label}] Fields present in one but not the other: {drift}"

    # BaseParams <-> RegisterBase.__init__
    # imagepath is required — not in BaseParams
    assert_in_sync(
        extract_sig_defaults(init_sig, exclude={"imagepath"}),
        extract_dataclass_defaults(BaseParams),
        "BaseParams vs RegisterBase.__init__",
    )

    # RunParams <-> RegisterBase.__call__
    # filename is required — not in RunParams
    assert_in_sync(
        extract_sig_defaults(call_sig, exclude={"filename"}),
        extract_dataclass_defaults(RunParams),
        "RunParams vs RegisterBase.__call__",
    )

    # fixed CLI extras <-> RegisterFixed.get_initial_pose_estimate
    # _img, _intrinsics are stubs; rot, xyz are required — none in CLI
    # files, imagepath, base, run are structural CLI args — not in impl
    assert_in_sync(
        extract_sig_defaults(fixed_impl_sig, exclude={"_img", "_intrinsics", "rot", "xyz"}),
        extract_sig_defaults(
            fixed_cmd_sig, exclude={"files", "imagepath", "rot", "xyz", "base", "run"}
        ),
        "fixed CLI extras vs RegisterFixed.get_initial_pose_estimate",
    )
