import pytest
import torch
from diffdrr.metrics import (
    GradientNormalizedCrossCorrelation2d,
    MultiscaleNormalizedCrossCorrelation2d,
)

from xvr.register.loss import load_loss_function
from xvr.register.losses import GradientMultiscaleNormalizedCrossCorrelation2d


def make_image_pair(seed: int = 0, size: int = 32):
    """Two uncorrelated random images, the generic case for a similarity metric."""
    torch.manual_seed(seed)
    return torch.rand(1, 1, size, size), torch.rand(1, 1, size, size)


def test_gmncc_is_symmetric():
    """Image similarity must not depend on argument order."""
    a, b = make_image_pair()
    metric = GradientMultiscaleNormalizedCrossCorrelation2d()
    torch.testing.assert_close(metric(a, b), metric(b, a))


def test_gmncc_of_an_image_with_itself_is_one():
    """A perfectly registered DRR is the optimum the loop maximizes toward."""
    a, _ = make_image_pair()
    metric = GradientMultiscaleNormalizedCrossCorrelation2d()
    assert metric(a, a).item() == pytest.approx(1.0, abs=1e-3)


def test_gmncc_beta_one_reduces_to_mncc():
    """beta weights the two terms; at 1.0 the gradient term must drop out entirely."""
    a, b = make_image_pair()
    expected = MultiscaleNormalizedCrossCorrelation2d([None, 9], [0.5, 0.5])(a, b)
    actual = GradientMultiscaleNormalizedCrossCorrelation2d(beta=1.0)(a, b)
    torch.testing.assert_close(actual, expected)


def test_gmncc_beta_zero_reduces_to_gncc():
    """The mirror of the beta=1 case: at 0.0 only the gradient term survives."""
    a, b = make_image_pair()
    expected = GradientNormalizedCrossCorrelation2d(11, 0.0)(a, b)
    actual = GradientMultiscaleNormalizedCrossCorrelation2d(beta=0.0)(a, b)
    torch.testing.assert_close(actual, expected)


def test_gmncc_is_differentiable_with_respect_to_the_prediction():
    """`loss.backward()` in the optimization loop needs a gradient back through the metric."""
    a, b = make_image_pair()
    b.requires_grad_(True)
    GradientMultiscaleNormalizedCrossCorrelation2d()(a, b).backward()
    assert b.grad is not None
    assert torch.isfinite(b.grad).all()
    assert b.grad.abs().sum() > 0


@pytest.mark.parametrize("name", ["mncc", "gncc", "gmncc"])
def test_load_loss_function_resolves_every_documented_metric(name):
    """These three strings are what the CLI's `--metric` Literal offers."""
    assert isinstance(load_loss_function(name), torch.nn.Module)


def test_load_loss_function_passes_through_a_custom_module():
    """A user-supplied nn.Module is used as-is rather than looked up by name."""
    custom = torch.nn.Identity()
    assert load_loss_function(custom) is custom


def test_load_loss_function_rejects_an_unknown_name():
    """The error message lists the valid keys, so it must name all three."""
    with pytest.raises(ValueError, match="Unknown metric"):
        load_loss_function("not-a-metric")
