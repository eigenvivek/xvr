import pytest
import torch

from xvr.register import FixedPose, Register, RestartPose


def pose_error(predicted, target):
    """Geodesic rotation error (degrees) and translation error (mm) between two poses."""
    delta = torch.linalg.inv(target.matrix[0]) @ predicted.matrix[0]
    cos = torch.clamp((torch.diagonal(delta[:3, :3]).sum() - 1) / 2, -1.0, 1.0)
    return torch.arccos(cos).rad2deg().item(), delta[:3, 3].norm().item()


def make_register(phantom_ct, start, **kwargs):
    """A CPU registration with the plateau early-stop disabled, so iteration counts are fixed."""
    options = dict(
        scales=[2.0, 1.0], n_itrs=[50, 50], patience=[10**9, 10**9], max_n_plateaus=10**9
    )
    options.update(kwargs)
    return Register(
        initializer=FixedPose(**start, orientation="AP", device="cpu"),
        imagepath=str(phantom_ct),
        device="cpu",
        **options,
    )


def test_registration_recovers_the_pose_the_xray_was_rendered_from(
    phantom_ct, xray_dicom, true_pose, offset_pose
):
    """The headline property: the answer is known by construction, so assert it is found.

    `linearize=False` because the X-ray is a DRR, not a detector reading -- see `xray_dicom`.
    """
    result = make_register(phantom_ct, offset_pose)(str(xray_dicom), linearize=False)

    init_rot, init_xyz = pose_error(result.init_pose, true_pose)
    final_rot, final_xyz = pose_error(result.final_pose, true_pose)

    assert final_rot < 1.5, f"rotation error {final_rot:.2f} deg (started at {init_rot:.2f})"
    assert final_xyz < 20.0, f"translation error {final_xyz:.2f} mm (started at {init_xyz:.2f})"
    assert final_rot < init_rot / 3
    assert final_xyz < init_xyz / 3


def test_the_metric_increases_over_the_run(phantom_ct, xray_dicom, offset_pose):
    """The optimizer maximizes similarity (`maximize=True`), so the loss must climb."""
    log = make_register(phantom_ct, offset_pose, n_itrs=[20, 20])(
        str(xray_dicom), linearize=False
    ).log
    assert log.losses[-1] > log.losses[0]
    assert all(map(torch.isfinite, map(torch.tensor, log.losses)))


def test_the_log_records_one_entry_per_iteration(phantom_ct, xray_dicom, offset_pose):
    """With the plateau break disabled the trace length is exactly sum(n_itrs)."""
    log = make_register(
        phantom_ct, offset_pose, scales=[2.0, 1.0], n_itrs=[7, 11], patience=[10**9, 10**9]
    )(str(xray_dicom), linearize=False).log
    assert len(log.losses) == 18
    assert len(log.scales) == len(log.rescale_factors) == len(log.times) == 18
    assert log.rots.shape == (18, 3)
    assert log.xyzs.shape == (18, 3)


def test_two_identical_runs_produce_identical_trajectories(phantom_ct, xray_dicom, offset_pose):
    """Nothing in the registration path is seeded, so it must be deterministic on CPU."""
    register = make_register(phantom_ct, offset_pose, n_itrs=[10, 10])
    first = register(str(xray_dicom), linearize=False).log
    second = register(str(xray_dicom), linearize=False).log
    assert first.losses == second.losses
    torch.testing.assert_close(first.rots, second.rots, atol=0.0, rtol=0.0)
    torch.testing.assert_close(first.xyzs, second.xyzs, atol=0.0, rtol=0.0)


def test_init_only_skips_optimization(phantom_ct, xray_dicom, offset_pose):
    """`--init-only` is how a model's raw prediction is inspected without refinement."""
    result = make_register(phantom_ct, offset_pose)(
        str(xray_dicom), linearize=False, init_only=True
    )
    assert result.log is None
    torch.testing.assert_close(result.final_pose.matrix, result.init_pose.matrix)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(scales=[8.0, 4.0], n_itrs=[10]), "scales and n_itrs"),
        (dict(scales=[8.0, 4.0], n_itrs=[10, 10], patience=[5]), "scales and patience"),
    ],
)
def test_mismatched_schedule_lengths_are_rejected(phantom_ct, offset_pose, kwargs, message):
    """The three per-stage schedules are zipped together, so they must line up."""
    with pytest.raises(ValueError, match=message):
        make_register(phantom_ct, offset_pose, **kwargs)


def test_scalar_patience_is_broadcast_across_scales(phantom_ct, offset_pose):
    """A single `patience` applies to every stage."""
    assert make_register(
        phantom_ct, offset_pose, scales=[8.0, 4.0], n_itrs=[1, 1], patience=5
    ).patience == [5, 5]


def test_a_pose_that_misses_the_volume_warns_instead_of_optimizing(
    phantom_ct, xray_dicom, offset_pose
):
    """A blank DRR has no gradient signal, so the loop bails out rather than diverging.

    Translating the source 500 mm laterally aims the cone clear of the 96 mm phantom.
    """
    off_target = {"rot": [0.0, 0.0, 0.0], "xyz": [500.0, 850.0, 0.0]}
    register = make_register(phantom_ct, off_target, n_itrs=[5, 5])
    with pytest.warns(UserWarning, match="blank"):
        assert register(str(xray_dicom), linearize=False).log is None


# --- saving and restarting ------------------------------------------------------------


def test_saved_results_carry_the_schema_restart_depends_on(
    phantom_ct, xray_dicom, offset_pose, tmp_path
):
    """`RestartPose` and `experiments/evaluate.py` both read this dict by key."""
    make_register(phantom_ct, offset_pose, n_itrs=[5, 5])(
        str(xray_dicom), linearize=False, savepath=tmp_path
    )
    saved = torch.load(tmp_path / "phantom.pth", weights_only=False)

    assert set(saved) == {"detector", "runtime", "init_pose", "final_pose", "gt", "log"}
    assert set(saved["detector"]) == {"sdd", "height", "width", "delx", "dely", "reverse_x_axis"}
    assert set(saved["log"]) == {"losses", "scales", "rescale_factors", "times", "rots", "xyzs"}
    assert saved["init_pose"].shape == saved["final_pose"].shape == (1, 4, 4)


def test_a_saved_pose_round_trips_through_restart(phantom_ct, xray_dicom, offset_pose, tmp_path):
    """Restarting must resume from exactly where the previous run stopped."""
    original = make_register(phantom_ct, offset_pose, n_itrs=[5, 5])(
        str(xray_dicom), linearize=False, savepath=tmp_path
    )

    resumed = Register(
        initializer=RestartPose(ckpt=str(tmp_path / "phantom.pth"), device="cpu"),
        imagepath=str(phantom_ct),
        device="cpu",
    )(str(xray_dicom), linearize=False, init_only=True)

    torch.testing.assert_close(resumed.init_pose.matrix, original.final_pose.matrix)


def test_init_only_runs_record_zero_runtime(phantom_ct, xray_dicom, offset_pose, tmp_path):
    """There is no optimization log to sum times from."""
    make_register(phantom_ct, offset_pose)(
        str(xray_dicom), linearize=False, init_only=True, savepath=tmp_path
    )
    saved = torch.load(tmp_path / "phantom.pth", weights_only=False)
    assert saved["runtime"] == 0.0
    assert saved["log"] is None
