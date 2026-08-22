from dataclasses import dataclass, fields


@dataclass
class Intrinsics:
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
