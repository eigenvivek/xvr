import pytest
import torch

from xvr.model.scheduler import IdentitySchedule, WarmupCosineSchedule


def make_optimizer(lr: float = 1.0):
    """A one-parameter optimizer, enough to drive a LambdaLR."""
    return torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=lr)


def test_warmup_ramps_linearly_from_zero():
    schedule = WarmupCosineSchedule(make_optimizer(), warmup_steps=10, t_total=100)
    assert schedule.lr_lambda(0) == 0.0
    assert schedule.lr_lambda(5) == pytest.approx(0.5)
    assert schedule.lr_lambda(10) == pytest.approx(1.0)


def test_cosine_decays_to_zero_at_the_end_of_training():
    schedule = WarmupCosineSchedule(make_optimizer(), warmup_steps=10, t_total=100)
    assert schedule.lr_lambda(55) == pytest.approx(0.5, abs=1e-6)
    assert schedule.lr_lambda(100) == pytest.approx(0.0, abs=1e-6)


def test_the_multiplier_never_leaves_the_unit_interval():
    schedule = WarmupCosineSchedule(make_optimizer(), warmup_steps=10, t_total=100)
    values = [schedule.lr_lambda(step) for step in range(0, 130)]
    assert min(values) >= 0.0 and max(values) <= 1.0


def test_float_bounds_are_accepted():
    """`initialize_modules` divides iteration counts by the accumulation factor, so these are floats."""
    schedule = WarmupCosineSchedule(make_optimizer(), warmup_steps=12.5, t_total=250.0)
    assert schedule.lr_lambda(0) == 0.0
    assert 0.0 <= schedule.lr_lambda(125) <= 1.0


def test_zero_warmup_does_not_divide_by_zero():
    schedule = WarmupCosineSchedule(make_optimizer(), warmup_steps=0, t_total=100)
    assert schedule.lr_lambda(0) == pytest.approx(1.0)


def test_identity_schedule_holds_the_learning_rate_constant():
    """`--disable-scheduler` must leave the optimizer's lr untouched."""
    optimizer = make_optimizer(lr=0.003)
    schedule = IdentitySchedule(optimizer)
    for _ in range(5):
        optimizer.step()
        schedule.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.003)
