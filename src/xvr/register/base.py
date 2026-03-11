from abc import ABC, abstractmethod
from inspect import Signature, signature
from pathlib import Path
from typing import Callable, Iterable

import torch
from jaxtyping import Float
from nanodrr.camera import make_k_inv
from nanodrr.registration import Registration
from tqdm import tqdm

from ..io import Intrinsics, read_xray
from ..utils import XrayTransforms
from .logging import OptimizationLogger, RegistrationResult
from .loss import load_loss_function
from .subject import load_subject


class RegisterBase(ABC):
    """Abstract base class for 2D/3D image registration.

    Performs multiscale gradient-based registration of a CT volume to an X-ray
    image by optimizing a differentiable image similarity metric over the pose
    parameters of a DRR renderer.

    Subclasses must implement `get_initial_pose_estimate` to define how the
    initial pose is computed before optimization begins.

    Args:
        imagepath: Path to the CT image.
        labelpath: Path to the segmentation label map. If None, uses the full image.
        labels: Label indices to include in the DRR. If None, uses all labels.
        metric: Image similarity metric, either a string key or a custom nn.Module.
        scales: Downsampling scale(s) for multiscale registration.
        n_itrs: Number of optimization iterations per scale.
        lr_rot: Learning rate for rotation parameters.
        lr_xyz: Learning rate for translation parameters.
        lr_reduce_factor: Factor by which to reduce the learning rate on plateau.
        patience: Number of steps without improvement before reducing the learning rate (can be defined per-scale).
        threshold: Minimum change to qualify as an improvement.
        max_n_plateaus: Number of learning rate reductions before early stopping.
        device: Torch device to run on.
        **metric_kwargs: Additional keyword arguments passed to the loss function.
    """

    _registry = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        RegisterBase._registry[cls.__name__.replace("Register", "").lower()] = cls

        base_sig = signature(RegisterBase.__init__)
        cls.__init__.__signature__ = _merge_signatures(signature(cls.__init__), base_sig)

        if "__call__" in cls.__dict__:
            cls.__call__.__signature__ = _merge_signatures(
                signature(cls.__call__),
                signature(RegisterBase.__call__),
            )

    @classmethod
    def create(cls, method: str, **kwargs):
        if method not in cls._registry:
            raise ValueError(f"Unknown method '{method}'. Choose from {list(cls._registry)}")
        return cls._registry[method](**kwargs)

    def __init__(
        self,
        imagepath: str,
        labelpath: str | None = None,
        labels: list[int] | None = None,
        metric: str | torch.nn.Module = "gmncc",
        scales: list[float] | float = 8.0,
        n_itrs: list[int] | int = 500,
        lr_rot: float = 1e-2,
        lr_xyz: float = 1e-0,
        lr_reduce_factor: float = 0.1,
        patience: int = 5,
        threshold: float = 1e-4,
        max_n_plateaus: int = 2,
        device: str = "cuda",
        **metric_kwargs,
    ):
        # Load the subject
        self.subject = load_subject(imagepath, labelpath, labels).to(device)

        # Load the loss function
        self.imagesim = load_loss_function(metric, **metric_kwargs).to(device)

        # Save the optimization hyperparameters
        self.scales = scales if isinstance(scales, Iterable) else [scales]
        self.n_itrs = n_itrs if isinstance(n_itrs, Iterable) else [n_itrs]
        if len(self.scales) != len(self.n_itrs):
            raise ValueError("scales and n_itrs must have the same length")
        self.max_scale_len = max(len(str(s)) for s in self.scales)

        # Optimization hyperparameters
        self.lr_rot = lr_rot
        self.lr_xyz = lr_xyz
        self.factor = lr_reduce_factor
        self.threshold = threshold
        self.max_n_plateaus = max_n_plateaus

        self.patience = (
            patience if isinstance(patience, Iterable) else [patience] * len(self.scales)
        )
        if len(self.scales) != len(self.patience):
            raise ValueError("scales and patience must have the same length")

        self.device = device

    @abstractmethod
    def get_initial_pose_estimate(
        self,
        img: Float[torch.Tensor, "1 1 H W"],
        intrinsics: Intrinsics,
        **kwargs,
    ) -> Float[torch.Tensor, "1 4 4"]:
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
            savepath: Location to save the registration results and an optimization GIF.
            **kwargs: Additional keyword arguments passed to `get_initial_pose_estimate`.

        Returns:
            reg: The Registration object with the optimized pose.
            gt: The preprocessed ground truth X-ray.
            losses: Loss value at each iteration.
            scales: Downsampling scale at each iteration.
            rescale_factors: Rescale factor at each iteration.
            rots: Rotation parameters at each iteration, shape (N, 3).
            xyzs: Translation parameters at each iteration, shape (N, 3).
        """
        # Read and preprocess the ground truth X-ray
        gt, intrinsics, _ = read_xray(filename, crop, subtract_background, linearize, reducefn)
        gt = gt.to(self.device)
        *_, height, width = gt.shape
        if reverse_x_axis:
            gt = gt.flip(-1)

        # Get the initial pose estimate
        with torch.no_grad():
            init_pose = self.get_initial_pose_estimate(gt, intrinsics, **kwargs).to(self.device)

        # Make the Registration object
        k_inv = make_k_inv(**intrinsics, height=height, width=width)
        sdd = torch.tensor([intrinsics.sdd])
        reg = Registration(self.subject, init_pose, k_inv, sdd, height, width).to(self.device)

        # Perform multiscale registration
        log = self._run_multiscale(reg, gt, equalize, crop)
        result = RegistrationResult(reg, gt, log)

        # Save and animate the registration results
        if savepath is not None:
            from ..plot import animate

            savepath = Path(savepath) / Path(filename).stem
            result.save(savepath.with_suffix(".pth"))
            animate(result, savepath=savepath.with_suffix(".gif"))

        return result

    def _run_multiscale(
        self,
        reg: Registration,
        gt: Float[torch.Tensor, "1 1 H W"],
        equalize: bool,
        crop: int,
    ) -> OptimizationLogger:
        """Run coarse-to-fine multiscale optimization."""
        # Compute the rescale factors for each scale (and append a reset factor)
        factors = parse_scales(self.scales + [1], crop, gt.shape[2])

        # Run the optimization and log iterations
        losses, scales, rescale_factors, rots, xyzs = [], [], [], [], []
        for stage, (scale, rescale_factor, n_itrs, patience) in enumerate(
            zip(self.scales, factors, self.n_itrs, self.patience)
        ):
            pbar = tqdm(range(n_itrs), ncols=100, desc=f"Scale {scale:>{self.max_scale_len}}")
            reg.rescale_(rescale_factor)
            optimizer, scheduler, transform = self._setup_stage(reg, stage, patience, equalize)
            current_lr, n_plateaus = torch.inf, 0
            true = transform(gt)
            for _ in pbar:
                optimizer.zero_grad()
                pred = transform(reg())
                loss = self.imagesim(true, pred)
                loss.backward()
                optimizer.step()
                scheduler.step(loss.detach())

                pbar.set_postfix_str(f"loss = {loss.item():5.3f}")
                losses.append(loss.item())
                scales.append(scale)
                rescale_factors.append(rescale_factor)
                rots.append(reg._rot.detach().clone())
                xyzs.append(reg._xyz.detach().clone())

                lr = min(scheduler.get_last_lr())
                if lr < current_lr:
                    current_lr = lr
                    n_plateaus += 1
                if n_plateaus > self.max_n_plateaus:
                    break

        # Reset the intrinsics to their native resolution
        reg.rescale_(factors[-1])

        # Return the optimization log
        rots, xyzs = torch.cat(rots).cpu(), torch.cat(xyzs).cpu()
        return OptimizationLogger(losses, scales, rescale_factors, rots, xyzs)

    def _setup_stage(self, reg: Registration, stage: int, patience: int, equalize: bool):
        """Configure the optimizer, scheduler, and transforms for a single scale stage."""
        step_size_scalar = 2**stage
        optimizer = torch.optim.Adam(
            [
                {"params": [reg._rot], "lr": self.lr_rot / step_size_scalar},
                {"params": [reg._xyz], "lr": self.lr_xyz / step_size_scalar},
            ],
            maximize=True,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "max", self.factor, patience, self.threshold
        )
        transform = XrayTransforms(reg.height, reg.width, equalize=equalize).to(self.device)
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


def _is_var_keyword(p) -> bool:
    return p.kind == p.VAR_KEYWORD


def _merge_signatures(sub_sig: Signature, base_sig: Signature) -> Signature:
    sub_params = [p for p in sub_sig.parameters.values() if not _is_var_keyword(p)]
    base_params = [
        p
        for p in base_sig.parameters.values()
        if p.name not in sub_sig.parameters and not _is_var_keyword(p)
    ]
    var_keyword = [p for p in base_sig.parameters.values() if _is_var_keyword(p)]
    return sub_sig.replace(parameters=sub_params + base_params + var_keyword)
