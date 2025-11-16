from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, final, overload
import torch
from torch.utils.data import Dataset as TorchDataset
import pandas as pd
from itertools import dropwhile
from jaxtyping import Float, Bool

from data.base import Dataset, ListDataset

from .utils import (
    ImageResize,
    ImageSize,
    ImageTransform,
    MaskTransform,
    Transform,
    generate_empty_mask,
    generate_image,
    generate_mask,
    normalize_image,
)


@dataclass
class MetaSample:
    image_path: str
    mask_path: str | None
    label: bool


@dataclass
class TensorSample:
    image: Float[torch.Tensor, "C=3 H W"]
    mask: Bool[torch.Tensor, "H W"]
    label: bool

    @staticmethod
    def collate_fn(batch: list["TensorSample"]) -> "TensorSampleBatch":
        images = torch.stack([b.image for b in batch])
        masks = torch.stack([b.mask for b in batch])
        labels = torch.tensor([b.label for b in batch])
        return TensorSampleBatch(
            images=images,
            masks=masks,
            labels=labels,
        )


@dataclass
class TensorSampleBatch:
    images: Float[torch.Tensor, "N C=3 H W"]
    masks: Bool[torch.Tensor, "N H W"]
    labels: Bool[torch.Tensor, "N"]


@dataclass
class MetaInfo:
    data_dir: Path
    category_datas: dict[str, list[MetaSample]]

    def to_csv(self, save_path: Path):
        print(f"Saving meta data for dataset to {save_path}...")
        categories = list(self.category_datas.keys())
        data = {
            "image_path": ([""] * len(categories))
            + [
                Path(sample.image_path).relative_to(self.data_dir).as_posix()
                for c in categories
                for sample in self.category_datas[c]
            ],
            "mask_path": ([""] * len(categories))
            + [
                (
                    Path(sample.mask_path).relative_to(self.data_dir).as_posix()
                    if sample.mask_path is not None
                    else None
                )
                for c in categories
                for sample in self.category_datas[c]
            ],
            "category": categories
            + [c for c in categories for _ in self.category_datas[c]],
            "label": ([""] * len(categories))
            + [sample.label for c in categories for sample in self.category_datas[c]],
        }
        df = pd.DataFrame(data)
        df.to_csv(save_path, index=False)

    @classmethod
    def from_csv(cls, data_dir: Path, csv_path: Path) -> "MetaInfo":
        print(f"Loading meta data for dataset from {csv_path}...")
        df = pd.read_csv(csv_path)
        category_datas: dict[str, list[MetaSample]] = {}
        it = dropwhile(lambda x: pd.isna(x[1]["image_path"]), df.iterrows())
        for _, row in it:
            sample = MetaSample(
                image_path=str(data_dir / row["image_path"]),
                mask_path=(
                    str(data_dir / row["mask_path"])
                    if not pd.isna(row["mask_path"])
                    else None
                ),
                label=bool(row["label"]),
            )
            category = str(row["category"])
            category_datas.setdefault(category, []).append(sample)
        meta_info = MetaInfo(data_dir=data_dir, category_datas=category_datas)
        return meta_info


class DetectionDataset:
    """
    get_tensor 必须基于传入的category和transform返回对应的Dataset[TensorSample], 不能调用self.get_transform()
    get_transform 和 set_transform 用于辅助 __getitem__ 来方便使用
    """

    @abstractmethod
    def get_tensor(
        self, category: str, transform: Transform
    ) -> Dataset[TensorSample]: ...

    @overload
    def __init__(
        self,
        name: str,
        categories: list[str],
    ): ...
    @overload
    def __init__(
        self,
        name: str,
        meta_infos: MetaInfo,
    ): ...

    def __init__(self, name: str, *args: Any, **kwargs: Any):
        self._name = name
        if "categories" in kwargs:
            meta_or_categories = kwargs["categories"]
        elif "meta_infos" in kwargs:
            meta_or_categories = kwargs["meta_infos"]
        else:
            meta_or_categories = args[0]
        meta_or_categories: MetaInfo | list[str]
        if isinstance(meta_or_categories, MetaInfo):
            self._categories = list(meta_or_categories.category_datas.keys())
            self._meta_info = meta_or_categories
        else:
            self._categories = meta_or_categories
            self._meta_info = None
        self._transform = Transform()

    @final
    def __getitem__(self, category: str) -> Dataset[TensorSample]:
        return self.get_tensor(category, self.get_transform())

    @final
    def get_name(self) -> str:
        return self._name

    @final
    def get_categories(self) -> list[str]:
        return self._categories

    @final
    def has_meta(self) -> bool:
        return self._meta_info is not None

    @final
    def get_meta_info(self) -> MetaInfo:
        assert self._meta_info is not None
        return self._meta_info

    @final
    def get_data_dir(self) -> Path:
        assert self._meta_info is not None
        return self._meta_info.data_dir

    @final
    def get_meta(self, category: str) -> Dataset[MetaSample]:
        assert self._meta_info is not None
        return ListDataset(self._meta_info.category_datas[category])

    def get_labels(self, category: str) -> Dataset[bool]:
        assert self._meta_info is not None
        return ListDataset(
            [sample.label for sample in self._meta_info.category_datas[category]]
        )

    @final
    def set_transform(self, transform: Transform):
        self._transform = transform

    @final
    def set_resize(self, resize: ImageResize | None):
        self._transform.resize = resize

    @final
    def set_image_transform(self, image_transform: ImageTransform):
        self._transform.image_transform = image_transform

    @final
    def set_mask_transform(self, mask_transform: MaskTransform):
        self._transform.mask_transform = mask_transform

    @final
    def get_transform(self) -> Transform:
        return self._transform


type TensorFactory = Callable[[str, Transform], Dataset[TensorSample]]


class DetectionDatasetByFactory(DetectionDataset):
    def __init__(
        self,
        name: str,
        meta_or_categories: MetaInfo | list[str],
        tensor_factory: TensorFactory,
    ):
        super().__init__(name, meta_or_categories)
        self._tensor_factory = tensor_factory

    def get_tensor(self, category: str, transform: Transform) -> Dataset[TensorSample]:
        return self._tensor_factory(category, transform)


class DatasetByMeta(Dataset[TensorSample]):
    def __init__(
        self,
        meta_dataset: Dataset[MetaSample],
        transform: Transform,
    ):
        self._meta_dataset = meta_dataset
        self._transform = transform

    def __len__(self) -> int:
        return len(self._meta_dataset)

    def __getitem__(self, idx: int) -> TensorSample:
        meta_sample = self._meta_dataset[idx]
        image = generate_image(Path(meta_sample.image_path), self._transform.resize)
        if meta_sample.mask_path is not None:
            mask = generate_mask(Path(meta_sample.mask_path), self._transform.resize)
        else:
            mask_size = ImageSize.fromnumpy(image.shape)
            mask = generate_empty_mask(mask_size)
        image = normalize_image(image)
        image = torch.tensor(image)
        image = self._transform.image_transform(image)
        mask = torch.tensor(mask)
        mask = self._transform.mask_transform(mask)
        return TensorSample(
            image=image,
            mask=mask,
            label=meta_sample.label,
        )


class MetaDataset(DetectionDataset):
    def __init__(
        self,
        name: str,
        meta_info: MetaInfo,
    ):
        super().__init__(name, meta_info)

    def get_tensor(self, category: str, transform: Transform) -> Dataset[TensorSample]:
        return DatasetByMeta(self.get_meta(category), transform)
