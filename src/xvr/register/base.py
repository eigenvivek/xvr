from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, ClassVar, Iterable

import torch
from attrs import define, field
from diffdrr.data import read
from diffdrr.drr import DRR
from diffdrr.pose import RigidTransform
from jaxtyping import Float
from tqdm import tqdm

from ..io import Intrinsics, read_xray
from ..utils import XrayTransforms
from .logging import OptimizationLogger, RegistrationResult
from .loss import load_loss_function
from .models import Pose


def _to_list(value) -> list:
    """Wrap a scalar in a list; coerce iterables (except strings) to lists."""
    if isinstance(value, str) or not isinstance(value, Iterable):
        return [value]
    return list(value)


@define(kw_only=True, slots=False, eq=False)
class RegisterBase(ABC):
    """Abstract base class for 2D/3D image registration.

    Performs multiscale gradient-based registration of a CT volume to an X-ray
    image by optimizing a differentiable image similarity metric over the pose
    parameters of a DRR renderer.

    Subclasses must implement `get_initial_pose_estimate` to define how the
    initial pose is computed before optimization begins. Subclasses should be
    decorated with `@define`.

    Args:
        imagepath: Path to the CT image.
        labelpath: Path to the segmentation label map. If None, uses the full image.
        labels: Label indices to include in the DRR. If None, uses all labels.
        orientation: Patient orientation for the DRR.
        metric: Image similarity metric, either a string key or a custom nn.Module.
        metric_kwargs: Additional keyword arguments passed to the loss function.
        scales: Downsampling scale(s) for multiscale registration.
        n_itrs: Number of optimization iterations per scale.
        lr_rot: Learning rate for rotation parameters.
        lr_xyz: Learning rate for translation parameters.
        lr_reduce_factor: Factor by which to reduce the learning rate on plateau.
        patience: Number of steps without improvement before reducing the learning rate (can be defined per-scale).
        threshold: Minimum change to qualify as an improvement.
        max_n_plateaus: Number of learning rate reductions before early stopping.
        device: Torch device to run on.
    """

    _registry: ClassVar[dict] = {}

    imagepath: str
    labelpath: str | None = field(default=None)
    labels: list[int] | None = field(default=None)
    orientation: str = field(default="AP")

    metric: str | torch.nn.Module = field(default="gmncc")
    metric_kwargs: dict = field(factory=dict)

    scales: list[float] = field(default=8.0, converter=_to_list)
    n_itrs: list[int] = field(default=500, converter=_to_list)
    lr_rot: float = field(default=1e-2)
    lr_xyz: float = field(default=1e-0)
    factor: float = field(default=0.1, alias="lr_reduce_factor")
    patience: list[int] | int = field(default=5)
    threshold: float = field(default=1e-4)
    max_n_plateaus: int = field(default=2)
    device: str = field(default="cuda")

    subject: Any = field(init=False, repr=False)
    imagesim: Any = field(init=False, repr=False)
    max_scale_len: int = field(init=False, repr=False)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        RegisterBase._registry[cls.__name__.replace("Register", "").lower()] = cls

    @classmethod
    def create(cls, method: str, **kwargs):
        if method not in cls._registry:
            raise ValueError(f"Unknown method '{method}'. Choose from {list(cls._registry)}")
        return cls._registry[method](**kwargs)

    def __attrs_post_init__(self):
        # Normalize patience by expanding scalar to a per-scale list
        if not isinstance(self.patience, Iterable):
            self.patience = [self.patience] * len(self.scales)
        else:
            self.patience = list(self.patience)

        # Validate schedule lengths
        if len(self.scales) != len(self.n_itrs):
            raise ValueError("scales and n_itrs must have the same length")
        if len(self.scales) != len(self.patience):
            raise ValueError("scales and patience must have the same length")

        self.max_scale_len = max(len(str(s)) for s in self.scales)

        # Load the CT subject and image similarity metric
        self.subject = read(self.imagepath, self.labelpath, self.labels, self.orientation)
        self.imagesim = load_loss_function(self.metric, **self.metric_kwargs).to(self.device)

    @abstractmethod
    def get_initial_pose_estimate(
        self,
        img: Float[torch.Tensor, "1 1 H W"],
        intrinsics: Intrinsics,
        **kwargs,
    ) -> RigidTransform:
        """Compute an initial pose estimate for registration.

        Args:
            img: Preprocessed ground truth X-ray image.
            intrinsics: Camera intrinsics for the X-ray.
            **kwargs: Additional arguments for specific implementations.

        Returns:
            A 4x4 rigid transformation matrix representing the initial pose.
        """
        ...

    def __call__(
        self,
        filename: str,
        crop: int = 0,
        linearize: bool = True,
        subtract_background: bool = False,
        equalize: bool = False,
        reducefn: str | int | Callable = "max",
        reverse_x_axis: bool = False,
        savepath: Path | str | None = None,
        **kwargs,
    ) -> RegistrationResult:
        """Run registration on an X-ray image.

        Preprocesses the X-ray, computes an initial pose estimate, and runs
        multiscale gradient-based optimization to refine the pose.

        Args:
            filename: Path to the X-ray image.
            crop: Number of pixels to crop from the image border.
            linearize: Convert image to linear attenuation values.
            subtract_background: Subtract background from the image.
            equalize: Apply histogram equalization during optimization.
            reducefn: Reduction function for multi-frame images.
            reverse_x_axis: Flip the image horizontally.
            savepath: Location to save the registration results.
            **kwargs: Additional keyword arguments passed to `get_initial_pose_estimate`.

        Returns:
            A RegistrationResult with the optimized pose and full optimization log.
        """
        # Read and preprocess the ground truth X-ray
        gt, intrinsics, _ = read_xray(filename, crop, subtract_background, linearize, reducefn)
        gt = gt.to(self.device)
        *_, height, width = gt.shape

        # Get the initial pose estimate
        with torch.no_grad():
            init_pose = self.get_initial_pose_estimate(gt, intrinsics, **kwargs).to(self.device)

        # Build the DRR renderer and pose model
        drr = DRR(
            subject=self.subject,
            height=height,
            width=width,
            reverse_x_axis=reverse_x_axis,
            renderer="trilinear",
            **intrinsics,
        ).to(self.device)
        pose = Pose(init_pose).to(self.device)

        # Perform multiscale registration
        log = self._run_multiscale(drr, pose, gt, equalize, crop)
        result = RegistrationResult(drr, pose, init_pose, gt, log)

        # Optionally save the registration result
        if savepath is not None:
            savepath = Path(savepath) / Path(filename).stem
            result.save(savepath.with_suffix(".pth"))

        return result

    def _run_multiscale(
        self,
        drr: DRR,
        pose: Pose,
        gt: Float[torch.Tensor, "1 1 H W"],
        equalize: bool,
        crop: int,
    ) -> OptimizationLogger:
        """Run coarse-to-fine multiscale optimization."""
        # Compute sequential rescale ratios (with a terminal reset-to-full-res step)
        factors = parse_scales(self.scales + [1], crop, gt.shape[2])

        losses, scales, rescale_factors, rots, xyzs = [], [], [], [], []
        for stage, (scale, rescale_factor, n_itrs, patience) in enumerate(
            zip(self.scales, factors, self.n_itrs, self.patience)
        ):
            pbar = tqdm(range(n_itrs), ncols=100, desc=f"Scale {scale:>{self.max_scale_len}}")
            drr.rescale_detector_(rescale_factor)
            optimizer, scheduler, transform = self._setup_stage(
                drr, pose, stage, patience, equalize
            )
            current_lr, n_plateaus = torch.inf, 0
            true = transform(gt)
            for _ in pbar:
                optimizer.zero_grad()
                pred = transform(drr(pose()))
                loss = self.imagesim(true, pred)
                loss.backward()
                optimizer.step()
                scheduler.step(loss.detach())

                pbar.set_postfix_str(f"loss = {loss.item():5.3f}")
                losses.append(loss.item())
                scales.append(scale)
                rescale_factors.append(rescale_factor)
                rots.append(pose._rot.detach().clone())
                xyzs.append(pose._xyz.detach().clone())

                lr = min(scheduler.get_last_lr())
                if lr < current_lr:
                    current_lr = lr
                    n_plateaus += 1
                if n_plateaus > self.max_n_plateaus:
                    break

        # Reset detector to native resolution
        drr.rescale_detector_(factors[-1])

        rots, xyzs = torch.cat(rots).cpu(), torch.cat(xyzs).cpu()
        return OptimizationLogger(losses, scales, rescale_factors, rots, xyzs)

    def _setup_stage(self, drr: DRR, pose: Pose, stage: int, patience: int, equalize: bool):
        """Configure the optimizer, scheduler, and transforms for a single scale stage."""
        step_size_scalar = 2**stage
        optimizer = torch.optim.Adam(
            [
                {"params": [pose._rot], "lr": self.lr_rot / step_size_scalar},
                {"params": [pose._xyz], "lr": self.lr_xyz / step_size_scalar},
            ],
            maximize=True,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "max", self.factor, patience, self.threshold
        )
        transform = XrayTransforms(drr.detector.height, drr.detector.width, equalize=equalize)
        return optimizer, scheduler, transform


def parse_scales(scales: list[float], crop: int, height: int) -> list[float]:
    """Convert absolute downscale factors to sequential rescale ratios for an image pyramid.

    Args:
        scales: Absolute downscale factors relative to the original image (e.g. [8, 4, 2, 1]).
            A scale of 1.0 snaps back to full cropped resolution.
        crop: Total pixels cropped from the original image (crop/2 from each side).
        height: Height of the cropped image in pixels.

    Returns:
        Sequential rescale ratios between consecutive pyramid levels.
    """
    pyramid = [1.0] + [1.0 if x == 1.0 else x * (height / (height + crop)) for x in scales]
    return [pyramid[idx] / pyramid[idx + 1] for idx in range(len(pyramid) - 1)]
