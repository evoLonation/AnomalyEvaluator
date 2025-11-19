import torch
from data.base import DatasetOverrideGetItem
from data.cached_dataset import CachedDataset
from data.cached_impl import RealIAD
from data.detection_dataset import (
    Dataset,
    DetectionDataset,
    MetaDataset,
    TensorSample,
)
from data.utils import ImageSize, Transform
from jaxtyping import Shaped


def rotate_transform(
    image: Shaped[torch.Tensor, "*C H W"], k: int
) -> Shaped[torch.Tensor, "*C H W"]:
    if image.shape[-1] != image.shape[-2]:
        from data.utils import pad_to_square

        image = pad_to_square(image, pad_value=0)
    image = torch.rot90(image, k=k, dims=[-2, -1])
    return image


class RandomRotatedDetectionDataset(DetectionDataset):
    def __init__(
        self,
        base_dataset: DetectionDataset,
        seed: int,
    ):
        self.base_dataset = base_dataset
        self.seed = seed
        categories = base_dataset.get_categories()
        name = base_dataset.get_name() + "(rotated)"
        super().__init__(name=name, categories=categories)
    
    def get_labels(self, category: str):
        return self.base_dataset.get_labels(category)

    def get_tensor(self, category: str, transform: Transform) -> Dataset[TensorSample]:
        tensor_dataset = self.base_dataset.get_tensor(
            category, Transform(transform.resize)
        )
        generator = torch.Generator().manual_seed(self.seed)
        angles = torch.randperm(len(tensor_dataset), generator=generator)
        angles = angles % 4
        origin_getitem = tensor_dataset.__getitem__

        def getitem_override(index: int) -> TensorSample:
            sample = origin_getitem(index)
            k = int(angles[index].item())
            image = rotate_transform(sample.image, k)
            mask = rotate_transform(sample.mask, k)
            image = transform.image_transform(image)
            mask = transform.mask_transform(mask)
            sample.image = image
            sample.mask = mask
            return sample

        return DatasetOverrideGetItem(tensor_dataset, getitem_override)

if __name__ == "__main__":
    from .summary import generate_summary_view
    generate_summary_view(RandomRotatedDetectionDataset(RealIAD(), seed=42))