"""Annotated types shared by the command argument schemas.

Keep this module free of heavy imports: `cli/configs` is loaded by
`xvr --help`, and pulling in torch there costs ~3s per CLI call.
"""

from typing import Annotated, Literal

from cyclopts import Parameter

# Cyclopts emits a `--empty-<name>` flag for every collection-typed parameter so
# an empty collection is expressible. A fixed-length tuple can never validly be
# empty, so that flag only ever produces a value the command rejects.
Range = Annotated[tuple[float, float], Parameter(negative_iterable=())]
Vector3 = Annotated[tuple[float, float, float], Parameter(negative_iterable=())]

Orientation = Literal["AP", "PA"]

# Keys of `diffdrr.registration.N_ANGULAR_COMPONENTS`, which is indexed directly.
Parameterization = Literal[
    "axis_angle",
    "euler_angles",
    "se3_log_map",
    "quaternion",
    "rotation_6d",
    "rotation_9d",
    "rotation_10d",
    "quaternion_adjugate",
]

# Three letters from XYZ, with the middle axis differing from both outer ones.
Convention = Literal[
    "XYX", "XYZ", "XZX", "XZY",
    "YXY", "YXZ", "YZX", "YZY",
    "ZXY", "ZXZ", "ZYX", "ZYZ",
]
