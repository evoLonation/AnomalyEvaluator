import torch
from data.cached_dataset import CachedDataset
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


class RandomRotatedDataset(Dataset[TensorSample]):
    def __init__(
        self,
        base_dataset: Dataset[TensorSample],
        seed: int,
        transform: Transform,
    ):
        self.base_dataset = base_dataset
        self.transform = transform
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(base_dataset), generator=generator)
        self.angles = indices % 4

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> TensorSample:
        sample = self.base_dataset[index]
        k = int(self.angles[index].item())
        image = rotate_transform(sample.image, k)
        mask = rotate_transform(sample.mask, k)
        image = self.transform.image_transform(image)
        mask = self.transform.mask_transform(mask)
        sample.image = image
        sample.mask = mask
        return sample


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

    def get_tensor(self, category: str, transform: Transform) -> Dataset[TensorSample]:
        resize_transform = Transform(resize=transform.resize)
        tensor_dataset = self.base_dataset.get_tensor(category, resize_transform)
        after_transform = Transform(
            image_transform=transform.image_transform,
            mask_transform=transform.mask_transform,
        )
        return RandomRotatedDataset(
            base_dataset=tensor_dataset,
            seed=self.seed,
            transform=after_transform,
        )
