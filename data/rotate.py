import torch
from data.detection_dataset import (
    CategoryTensorDataset,
    DetectionDataset,
    MetaDataset,
    TensorDataset,
    TensorSample,
)
from data.utils import ImageSize


class RandomRotateCategoryDataset(CategoryTensorDataset):
    def __init__(
        self,
        base_dataset: CategoryTensorDataset,
        seed: int,
    ):
        self.base_dataset = base_dataset
        self.generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(base_dataset), generator=self.generator)
        self.angles = indices % 4
        self.image_size = base_dataset.get_imagesize()
        if self.image_size.h != self.image_size.w:
            max_side = max(self.image_size.h, self.image_size.w)
            self.image_size = ImageSize.square(max_side)
            self.need_pad = True
        else:
            self.need_pad = False

    def get_item(self, idx: int) -> TensorSample:
        sample = self.base_dataset.get_item(idx)
        angle = self.angles[idx].item()
        image = torch.rot90(sample.image, k=int(angle), dims=[1, 2])
        mask = torch.rot90(sample.mask, k=int(angle), dims=[0, 1])
        if self.need_pad:
            from data.utils import pad_to_square

            image = pad_to_square(image, pad_value=0)
            mask = pad_to_square(mask, pad_value=0)

        assert (
            image.shape[1] == self.image_size.h
        ), f"{image.shape} vs {self.image_size}"
        assert (
            image.shape[2] == self.image_size.w
        ), f"{image.shape} vs {self.image_size}"
        assert mask.shape[0] == self.image_size.h, f"{mask.shape} vs {self.image_size}"
        assert mask.shape[1] == self.image_size.w, f"{mask.shape} vs {self.image_size}"
        return TensorSample(
            image=image,
            mask=mask,
            label=sample.label,
        )

    def __len__(self) -> int:
        return len(self.base_dataset)

    def get_labels(self):
        return self.get_labels()

    def get_imagesize(self) -> ImageSize:
        return self.image_size


class RandomRotatedDataset(TensorDataset):
    def __init__(
        self,
        base_dataset: TensorDataset,
        seed: int,
    ):
        name = base_dataset.name + "(rotated)"
        category_datas: dict[str, CategoryTensorDataset] = {
            k: RandomRotateCategoryDataset(v, seed=seed)
            for k, v in base_dataset.category_datas.items()
        }
        super().__init__(name=name, category_datas=category_datas)


class RandomRotatedDetectionDataset(DetectionDataset):
    """
    只会对get_tensor_dataset_impl产生的TensorDataset进行旋转扩增, 不会影响get_meta_dataset
    """

    def __init__(
        self,
        base_dataset: DetectionDataset,
        seed: int,
    ):
        self.base_dataset = base_dataset
        self.seed = seed
        self.name = base_dataset.name + "(rotated)"

    def get_meta_dataset(self) -> MetaDataset:
        return self.base_dataset.get_meta_dataset()

    def get_tensor_dataset_impl(self, resize: ImageSize | int | None) -> TensorDataset:
        return RandomRotatedDataset(
            self.base_dataset.get_tensor_dataset_impl(resize), seed=self.seed
        )
