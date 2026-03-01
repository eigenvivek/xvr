import torch
from nanodrr.data import Subject
from torchio import LabelMap, ScalarImage


def load_subject(
    imagepath: str,
    labelpath: str | None = None,
    labels: list[int] | None = None,
) -> Subject:
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
