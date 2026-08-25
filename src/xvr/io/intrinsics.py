from dataclasses import dataclass, fields


@dataclass
class Intrinsics:
    """Intrinsic parameters of an imaging system.

    Supports mapping-style access (``keys`` and ``__getitem__``) so an instance
    can be splatted into a renderer with ``**intrinsics``.

    Attributes:
        sdd: Source-to-detector distance in millimeters.
        delx: Pixel spacing along the x-axis in millimeters.
        dely: Pixel spacing along the y-axis in millimeters.
        x0: Detector origin offset along the x-axis in millimeters.
        y0: Detector origin offset along the y-axis in millimeters.
    """

    sdd: float
    delx: float
    dely: float
    x0: float
    y0: float

    def __post_init__(self):
        for f in fields(self):
            setattr(self, f.name, float(getattr(self, f.name)))

    def keys(self):
        return [f.name for f in fields(self)]

    def __getitem__(self, key):
        return getattr(self, key)
