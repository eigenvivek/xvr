import base64
from io import BytesIO
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from diffdrr.visualization import plot_drr
from PIL import Image, ImageFilter

from .logging import OptimizationLogger

VizMethod = Literal["overlay", "checkerboard", "edges"]


class RegistrationOutput:
    """Image output that renders inline in Jupyter and can be saved to disk."""

    def __init__(self, data: bytes, fmt: str, width: int):
        self._data = data
        self._fmt = fmt
        self._width = width

    def _repr_html_(self) -> str:
        mime = "image/gif" if self._fmt == "gif" else "image/png"
        b64 = base64.b64encode(self._data).decode()
        return f'<img src="data:{mime};base64,{b64}" width="{self._width}"/>'

    def save(self, path: str) -> None:
        """Save the image to disk.

        Args:
            path: Output file path, e.g. "result.png" or "result.gif".
        """
        with open(path, "wb") as f:
            f.write(self._data)


def gif(
    logger: OptimizationLogger,
    width: int = 600,
    duration: int = 500,
) -> RegistrationOutput:
    """Animate between GT and predictions as an infinitely looping GIF.

    Args:
        logger: OptimizationLogger with .gt, .init_pose, .final_pose, and .drr.
        width: Display width in pixels when rendered in Jupyter.
        duration: Time per frame in milliseconds.

    Returns:
        RegistrationOutput that renders inline in Jupyter and can be saved to disk.
    """
    with torch.inference_mode():
        img_init = logger.drr(logger.init_pose)
        img_final = logger.drr(logger.final_pose)

    gt_pil = _to_pil(logger.gt)
    init_pil = _to_pil(img_init)
    fin_pil = _to_pil(img_final)

    frame1 = Image.fromarray(np.hstack([np.array(gt_pil), np.array(gt_pil)]))
    frame2 = Image.fromarray(np.hstack([np.array(init_pil), np.array(fin_pil)]))

    buf = BytesIO()
    frame1.save(buf, format="GIF", save_all=True, append_images=[frame2], loop=0, duration=duration)
    buf.seek(0)
    return RegistrationOutput(buf.read(), fmt="gif", width=width)


def plot(
    logger: OptimizationLogger,
    method: VizMethod = "edges",
    histeq_strength: float = 0.5,
    width: int = 900,
    **kwargs,
) -> RegistrationOutput:
    """Visualise registration results (GT | GT vs initial pred | GT vs final pred).

    Args:
        logger: OptimizationLogger with .gt, .init_pose, .final_pose, and .drr.
        method: Comparison method, one of "overlay", "checkerboard", "edges".
        histeq_strength: Histogram equalisation strength (0.0 = off, 1.0 = full).
        width: Display width in pixels when rendered in Jupyter.
        **kwargs: Forwarded to the chosen method function. "checkerboard" accepts
            n_patches (int, default 8); "edges" accepts sigma (float, default 2.0).

    Returns:
        RegistrationOutput that renders inline in Jupyter and can be saved to disk.
    """
    if method not in {"overlay", "checkerboard", "edges"}:
        raise ValueError(f"method must be 'overlay', 'checkerboard', or 'edges'; got {method!r}")

    with torch.inference_mode():
        img_init = logger.drr(logger.init_pose)
        img_final = logger.drr(logger.final_pose)

    gt_pil, init_pil, fin_pil = _histeq_joint(
        [_to_pil(t) for t in [logger.gt, img_init, img_final]],
        strength=histeq_strength,
    )

    viz_fn = {"overlay": _overlay, "checkerboard": _checkerboard, "edges": _edges}[method]

    _dpi = 100  # internal only — cancels out in figsize, controls rendering sharpness
    h, w = logger.gt.squeeze().shape[-2:]
    fig, axs = plt.subplots(ncols=3, figsize=(3 * w / _dpi, h / _dpi), dpi=_dpi)

    plot_drr(logger.gt, axs=axs[0], ticks=False)
    axs[0].set_aspect("auto")

    for ax, pred_pil in zip(axs[1:], [init_pil, fin_pil]):
        ax.imshow(viz_fn(pred_pil, gt_pil, **kwargs), aspect="auto")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.subplots_adjust(wspace=0, hspace=0, top=1, bottom=0, left=0, right=1)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=_dpi)
    plt.close(fig)
    buf.seek(0)
    return RegistrationOutput(buf.read(), fmt="png", width=width)


def _to_pil(t: torch.Tensor) -> Image.Image:
    """Normalise a tensor to [0, 255] and return a PIL RGB image."""
    a = t.squeeze().cpu().float()
    a = (a - a.min()) / (a.max() + 1e-8)
    return Image.fromarray((a.numpy() * 255).astype(np.uint8)).convert("RGB")


def _histeq_joint(imgs: list[Image.Image], strength: float) -> list[Image.Image]:
    """Equalise images using a single shared CDF.

    Pooling all pixels into one histogram keeps relative brightness consistent
    across GT and prediction, preventing either channel from dominating overlays.
    """
    arrays = [np.array(img.convert("L")) for img in imgs]

    hist, _ = np.histogram(np.concatenate([a.flatten() for a in arrays]), bins=256, range=[0, 256])
    cdf = hist.cumsum()
    cdf_norm = ((cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())).astype(np.uint8)

    result = []
    for arr in arrays:
        eq_arr = cdf_norm[arr]
        blended = ((1 - strength) * arr + strength * eq_arr).clip(0, 255).astype(np.uint8)
        result.append(Image.fromarray(blended).convert("RGB"))
    return result


def _overlay(pred: Image.Image, gt: Image.Image) -> np.ndarray:
    """pred → red channel, gt → green channel; overlap appears yellow."""
    r, _, _ = pred.split()
    _, g, _ = gt.split()
    return np.array(Image.merge("RGB", (r, g, Image.new("L", pred.size))))


def _checkerboard(pred: Image.Image, gt: Image.Image, n_patches: int = 8) -> np.ndarray:
    """Alternate n×n tiles between gt (even indices) and pred (odd indices)."""
    w, h = gt.size
    gt_arr, pred_arr = np.array(gt), np.array(pred)
    out = np.empty_like(gt_arr)
    ph, pw = h // n_patches, w // n_patches

    for i in range(n_patches):
        for j in range(n_patches):
            y0, y1 = i * ph, (i + 1) * ph if i < n_patches - 1 else h
            x0, x1 = j * pw, (j + 1) * pw if j < n_patches - 1 else w
            src = gt_arr if (i + j) % 2 == 0 else pred_arr
            out[y0:y1, x0:x1] = src[y0:y1, x0:x1]

    return out


def _edges(pred: Image.Image, gt: Image.Image, sigma: float = 2.0) -> np.ndarray:
    """Gradient-magnitude edges of pred overlaid in red on grayscale gt."""
    blurred = pred.convert("L").filter(ImageFilter.GaussianBlur(radius=sigma))
    arr = np.array(blurred).astype(np.float32)
    gy, gx = np.gradient(arr)
    edges = np.hypot(gx, gy)
    edges /= edges.max() + 1e-8

    gt_arr = np.array(gt.convert("L")).astype(np.float32)
    rgb = np.zeros((*gt_arr.shape, 3), dtype=np.uint8)
    rgb[:, :, 0] = np.clip(gt_arr + edges * 255, 0, 255).astype(np.uint8)
    rgb[:, :, 1] = np.clip(gt_arr * (1 - edges), 0, 255).astype(np.uint8)
    rgb[:, :, 2] = np.clip(gt_arr * (1 - edges), 0, 255).astype(np.uint8)
    return rgb
