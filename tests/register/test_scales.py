import math

import pytest

from xvr.register.register import _to_list, parse_scales


def test_parse_scales_converts_absolute_factors_to_sequential_ratios():
    """A [8, 4, 2, 1] pyramid steps down to 1/8, then halves the downsampling each stage."""
    assert parse_scales([8, 4, 2, 1], crop=0, height=100) == [0.125, 2.0, 2.0, 2.0]


def test_parse_scales_ratios_multiply_to_one():
    """The detector must land back at native resolution, so the ratios telescope to 1."""
    for crop in [0, 20, 57]:
        factors = parse_scales([8, 4, 2, 1], crop=crop, height=100)
        assert math.prod(factors) == pytest.approx(1.0)


def test_parse_scales_crop_shrinks_the_first_step():
    """Cropping shrinks the image, so the first downsample is gentler by height/(height+crop)."""
    uncropped = parse_scales([8, 1], crop=0, height=100)
    cropped = parse_scales([8, 1], crop=20, height=100)
    assert cropped[0] == pytest.approx(uncropped[0] * (100 + 20) / 100)


def test_parse_scales_returns_one_ratio_per_input_scale():
    """`_run_multiscale` appends a terminal 1.0, so it always gets len(scales) + 1 ratios."""
    assert len(parse_scales([8, 4, 2, 1], crop=0, height=100)) == 4


@pytest.mark.parametrize(
    "value, expected",
    [
        (8.0, [8.0]),
        (5, [5]),
        ("euler_angles", ["euler_angles"]),  # strings are scalars, not iterables
        ([1, 2], [1, 2]),
        ((1, 2), [1, 2]),
    ],
)
def test_to_list(value, expected):
    """Scalars and strings wrap; other iterables coerce to list."""
    assert _to_list(value) == expected
