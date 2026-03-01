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
