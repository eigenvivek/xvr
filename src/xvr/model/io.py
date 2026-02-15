import random

from torchio import LabelMap, ScalarImage, Subject, SubjectsLoader, Transform

from nanodrr.data.io import Subject as NanoSubject
from nanodrr.data.preprocess import hu_to_mu


class RandomHUToMu(Transform):
    def __init__(
        self,
        mu_water: float = 0.0192,
        mu_bone_range: tuple[float, float] = (0.0, 0.2),
        hu_bone: float = 1000.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mu_water = mu_water
        self.mu_bone_range = mu_bone_range
        self.hu_bone = hu_bone

    def apply_transform(self, subject: Subject) -> Subject:
        mu_bone = random.uniform(*self.mu_bone_range)
        for image in subject.get_images(intensity_only=True):
            image.set_data(hu_to_mu(image.data, self.mu_water, mu_bone, self.hu_bone))
        return subject


class SubjectIterator:
    """Wraps a SubjectsLoader to yield Subject objects instead of dicts."""

    def __init__(self, loader: SubjectsLoader):
        self.loader = loader

    def _to_subject(self, data: dict) -> NanoSubject:
        image = ScalarImage(
            tensor=data["volume"]["data"][0],
            affine=data["volume"]["affine"][0],
        )
        mask_data = data.get("mask")
        label = (
            LabelMap(tensor=mask_data["data"][0], affine=mask_data["affine"][0])
            if mask_data is not None
            else None
        )
        return NanoSubject.from_images(image, label, convert_to_mu=False).cuda()

    def __iter__(self):
        for data in self.loader:
            yield self._to_subject(data)
