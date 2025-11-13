from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self, final
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from itertools import dropwhile
from jaxtyping import Float, Bool
from torch.utils.data import Sampler, DataLoader

from .utils import ImageResize, ImageSize, ImageTransform, MaskTransform


@dataclass
class MetaSample:
    image_path: str
    mask_path: str | None
    label: bool


@dataclass
class CategoryMetaDataset(Dataset[MetaSample]):
    samples: list[MetaSample]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> MetaSample:
        return self.samples[idx]

    @staticmethod
    def collate_fn(batch: list[MetaSample]) -> list[MetaSample]:
        return batch


@dataclass
class MetaDataset:
    name: str
    category_datas: dict[str, CategoryMetaDataset]
    data_dir: Path

    @classmethod
    def get_meta_csv_path(cls, name: str, save_dir: Path) -> Path:
        return save_dir / f"{name}_meta.csv"

    def to_csv(self, data_dir: Path, save_dir: Path):
        save_path = self.get_meta_csv_path(self.name, save_dir)
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving meta data for dataset {self.name} to {save_path}...")
        categories = list(self.category_datas.keys())
        data = {
            "image_path": ([""] * len(categories))
            + [
                Path(sample.image_path).relative_to(data_dir).as_posix()
                for _, samples in self.category_datas.items()
                for sample in samples
            ],
            "mask_path": ([""] * len(categories))
            + [
                (
                    Path(sample.mask_path).relative_to(data_dir).as_posix()
                    if sample.mask_path is not None
                    else None
                )
                for _, samples in self.category_datas.items()
                for sample in samples
            ],
            "category": categories
            + [
                category
                for category, samples in self.category_datas.items()
                for _ in samples
            ],
            "label": ([""] * len(categories))
            + [
                sample.label
                for _, samples in self.category_datas.items()
                for sample in samples
            ],
        }
        df = pd.DataFrame(data)
        df.to_csv(save_path, index=False)

    @classmethod
    def from_csv(cls, name: str, data_dir: Path, save_dir: Path) -> Self:
        csv_path = cls.get_meta_csv_path(name, save_dir)
        print(f"Loading meta data for dataset {name} from {csv_path}...")
        df = pd.read_csv(csv_path)
        category_datas: dict[str, CategoryMetaDataset] = {}
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
            category_datas.setdefault(category, CategoryMetaDataset([])).samples.append(
                sample
            )
        return cls(name, category_datas, data_dir)

    @classmethod
    def get_categories(cls, name: str, save_dir: Path) -> list[str]:
        csv_path = cls.get_meta_csv_path(name, save_dir)
        df = pd.read_csv(csv_path, chunksize=10)
        categories = []
        done = False
        for chunk in df:
            for _, row in chunk.iterrows():
                if row["image_path"] != "":
                    done = True
                    break
                categories.append(row["category"])
            if done:
                break
        return categories


@dataclass
class TensorSample:
    image: Float[torch.Tensor, "C=3 H W"]
    mask: Bool[torch.Tensor, "H W"]
    label: bool


@dataclass
class TensorSampleBatch:
    images: Float[torch.Tensor, "N C=3 H W"]
    masks: Bool[torch.Tensor, "N H W"]
    labels: Bool[torch.Tensor, "N"]


class CategoryTensorDataset(Dataset[TensorSample]):

    @abstractmethod
    def __len__(self) -> int: ...
    @abstractmethod
    def get_item(self, idx: int) -> TensorSample: ...
    @final
    def __getitem__(self, idx: int) -> TensorSample:
        sample = self.get_item(idx)
        if hasattr(self, "_image_transform") and self._image_transform is not None:
            sample.image = self._image_transform(sample.image)
        if hasattr(self, "_mask_transform") and self._mask_transform is not None:
            sample.mask = self._mask_transform(sample.mask)
        return sample

    @abstractmethod
    def get_labels(self) -> list[bool]: ...
    @staticmethod
    def collate_fn(batch: list[TensorSample]) -> TensorSampleBatch:
        images = torch.stack([b.image for b in batch])
        masks = torch.stack([b.mask for b in batch])
        labels = torch.tensor([b.label for b in batch], dtype=torch.bool)
        return TensorSampleBatch(
            images=images,
            masks=masks,
            labels=labels,
        )

    def set_transforms(
        self,
        image_transform: ImageTransform | None = None,
        mask_transform: MaskTransform | None = None,
    ):
        self._image_transform: ImageTransform | None = image_transform
        self._mask_transform: MaskTransform | None = mask_transform


@dataclass
class TensorDataset:
    name: str
    category_datas: dict[str, CategoryTensorDataset]


class DetectionDataset(ABC):
    name: str

    @abstractmethod
    def get_meta_dataset(self) -> MetaDataset: ...
    @final
    def get_tensor_dataset(
        self,
        resize: ImageResize | None,
        image_transform: ImageTransform | None = None,
        mask_transform: MaskTransform | None = None,
    ) -> TensorDataset:
        tensor_dataset = self.get_tensor_dataset_impl(resize)
        for category_data in tensor_dataset.category_datas.values():
            category_data.set_transforms(image_transform, mask_transform)
        return tensor_dataset

    @abstractmethod
    def get_tensor_dataset_impl(
        self,
        resize: ImageResize | None,
    ) -> TensorDataset: ...
