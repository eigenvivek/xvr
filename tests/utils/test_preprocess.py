import torch

from xvr.utils.preprocess import Equalize, Identity, Standardize, XrayTransforms


def make_image(seed: int = 0, size: int = 32):
    """A random image on an arbitrary intensity range, as read off a detector."""
    torch.manual_seed(seed)
    return torch.rand(1, 1, size, size) * 250.0 - 30.0


def test_standardize_maps_onto_the_unit_interval():
    """Every downstream transform assumes intensities have already been rescaled to [0, 1]."""
    out = Standardize()(make_image())
    assert out.min().item() == 0.0
    assert out.max().item() <= 1.0
    assert out.max().item() > 0.99


def test_standardize_is_invariant_to_affine_intensity_changes():
    """Detector gain and offset must not change the standardized image."""
    img = make_image()
    torch.testing.assert_close(
        Standardize()(img), Standardize()(3.0 * img + 17.0), atol=1e-5, rtol=1e-5
    )


def test_standardize_does_not_divide_by_zero_on_a_constant_image():
    """A blank frame is degenerate but must not produce NaNs."""
    out = Standardize()(torch.full((1, 1, 8, 8), 4.0))
    assert torch.isfinite(out).all()


def test_identity_returns_its_input_unchanged():
    """`Identity` stands in for `Equalize` when equalization is off."""
    img = make_image()
    assert Identity()(img) is img


def test_equalize_stays_within_the_unit_interval():
    """Equalization redistributes intensities but must not leave [0, 1]."""
    out = Equalize()(Standardize()(make_image()))
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0


def test_equalize_is_differentiable():
    """The soft-histogram formulation exists so `--equalize` stays usable during optimization."""
    img = Standardize()(make_image())
    img.requires_grad_(True)
    Equalize()(img).sum().backward()
    assert img.grad is not None and torch.isfinite(img.grad).all()


def test_xray_transforms_resizes_to_the_detector_shape():
    """The transform is built from `drr.detector.height/width` and must match it exactly."""
    out = XrayTransforms(16, 24)(make_image(size=64))
    assert out.shape == (1, 1, 16, 24)


def test_xray_transforms_defaults_to_a_square_output():
    """`width=None` mirrors `height`."""
    assert XrayTransforms(16)(make_image(size=64)).shape == (1, 1, 16, 16)


def test_xray_transforms_normalizes_to_the_declared_mean_and_std():
    """Standardize -> Resize -> Normalize([mean], [std]) must undo to roughly zero mean."""
    out = XrayTransforms(32, mean=0.15, std=0.1)(make_image(size=32))
    raw = Standardize()(make_image(size=32))
    torch.testing.assert_close(out, (raw - 0.15) / 0.1, atol=1e-4, rtol=1e-4)
