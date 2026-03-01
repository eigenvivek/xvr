import torch
from nanodrr.data import Subject
from torchio import LabelMap, ScalarImage


def load_subject(
    imagepath: str,
    labelpath: str | None = None,
    labels: list[int] | None = None,
) -> Subject:
    """Load a volume, optionally masked by a label map.

    If no labelpath is provided, the image is loaded directly as a Subject.
    If a labelpath is given, the image is masked so that only voxels whose
    corresponding label values are in `labels` are retained; all other voxels
    are set to the image minimum. The masked image is then wrapped in a Subject.

    Args:
        imagepath: Path to the scalar image file.
        labelpath: Path to the label map file. If None, the image is loaded
            without any masking.
        labels: List of integer label values to retain. If None and a labelpath
            is given, all non-zero labels (1 to max) are retained. Raises a
            ValueError if provided without a labelpath.

    Returns:
        A Subject containing the (optionally masked) scalar image.

    Raises:
        ValueError: If `labels` is provided but `labelpath` is None.
    """
    if labelpath is None:
        if labels is not None:
            raise ValueError("Labels provided but no labelpath given.")
        return Subject.from_filepath(imagepath)

    image = ScalarImage(imagepath)
    label = LabelMap(labelpath)

    if labels is None:
        labels_tensor = torch.arange(1, int(label.data.max()) + 1)
    else:
        labels_tensor = torch.tensor(labels)

    tensor = torch.where(
        torch.isin(label.data, labels_tensor),
        image.data,
        image.data.min(),
    )

    image = ScalarImage(tensor=tensor, affine=image.affine)

    return Subject.from_images(image)
