import pytest

from xvr.io.intrinsics import Intrinsics


def test_fields_are_coerced_to_float():
    """DICOM tags arrive as pydicom DSfloat/int; the DRR constructor needs plain floats."""
    intrinsics = Intrinsics(1000, 2, 2, 0, 0)
    assert all(isinstance(getattr(intrinsics, f), float) for f in intrinsics.keys())


def test_supports_mapping_unpacking():
    """`Register.__call__` splats these straight into `DRR(**intrinsics)`."""
    intrinsics = Intrinsics(1000.0, 2.0, 3.0, 0.5, -0.5)
    assert dict(**intrinsics) == {
        "sdd": 1000.0,
        "delx": 2.0,
        "dely": 3.0,
        "x0": 0.5,
        "y0": -0.5,
    }


def test_getitem_matches_attribute_access():
    intrinsics = Intrinsics(1000.0, 2.0, 3.0, 0.5, -0.5)
    for key in intrinsics.keys():
        assert intrinsics[key] == getattr(intrinsics, key)


def test_unknown_key_raises():
    with pytest.raises((KeyError, AttributeError)):
        Intrinsics(1000.0, 2.0, 3.0, 0.0, 0.0)["nope"]
