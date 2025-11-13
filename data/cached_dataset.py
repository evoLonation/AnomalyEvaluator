from pathlib import Path
from abc import abstractmethod
import json
from typing import Self, override

import h5py
import numpy as np
from tqdm import tqdm

from .utils import ImageSize, generate_image, generate_mask
from .detection_dataset import (
    CategoryTensorDataset,
    DetectionDataset,
    MetaDataset,
    MetaSample,
    CategoryMetaDataset,
    TensorDataset,
    TensorSample,
)


class TensorH5Dataset(TensorDataset):
    class CategoryDataset(CategoryTensorDataset):
        def __init__(
            self,
            category: str,
            h5_file: Path,
        ):
            self.h5_file = h5_file
            self.category = category
            with h5py.File(self.h5_file, "r") as h5f:
                self.length = len(h5f[category]["images"])  # type: ignore

        @override
        def __len__(self) -> int:
            return self.length

        @override
        def __getitem__(self, idx: int) -> TensorSample:
            # 每次单独打开句柄是考虑到了线程安全性
            with h5py.File(self.h5_file, "r") as h5f:
                image = h5f[self.category]["images"][idx]  # type: ignore
                image = normalize_image(image)  # type: ignore
                mask_index = h5f[self.category]["mask_indices"][idx]  # type: ignore
                if mask_index == -1:
                    mask = generate_empty_mask((image.shape[2], image.shape[1]))  # type: ignore
                else:
                    mask = h5f[self.category]["masks"][mask_index]  # type: ignore
                label = h5f[self.category]["labels"][idx].item()  # type: ignore
                return TensorSample(image=image, mask=mask, label=label)  # type: ignore

        @override
        def get_labels(self) -> list[bool]:
            with h5py.File(self.h5_file, "r") as h5f:
                labels = h5f[self.category]["labels"][:]  # type: ignore
                return list(labels)  # type: ignore

    @classmethod
    def get_h5_path(
        cls, name: str, save_dir: Path, image_size: ImageSize | int | None = None
    ) -> Path:
        if image_size is None:
            return save_dir / f"{name}_default.h5"
        if isinstance(image_size, int):
            return save_dir / f"{name}_{image_size}.h5"
        return save_dir / f"{name}_{image_size[0]}x{image_size[1]}.h5"

    @classmethod
    def get_all_h5_paths(cls, name: str, save_dir: Path) -> list[Path]:
        pattern = save_dir / f"{name}_*.h5"
        return list(pattern.parent.glob(pattern.name))

    @classmethod
    def get_categories(cls, name: str, save_dir: Path) -> list[str]:
        h5_path = cls.get_all_h5_paths(name, save_dir)[0]
        categories = []
        with h5py.File(h5_path, "r") as h5f:
            categories = list(h5f.keys())
        return categories

    @classmethod
    def to_h5(
        cls,
        dataset: MetaDataset,
        save_dir: Path,
        image_size: ImageSize | int | None = None,
    ):
        save_path = cls.get_h5_path(dataset.name, save_dir, image_size)
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving tensor dataset {dataset.name} to {save_path}...")

        try:
            with h5py.File(save_path, "w") as h5f:
                for category, samples in tqdm(
                    dataset.category_datas.items(), desc="Saving to H5"
                ):
                    grp = h5f.create_group(category)
                    images = []
                    masks = []
                    mask_indices = []
                    for sample in samples:
                        img = generate_image(Path(sample.image_path), image_size)
                        images.append(img)
                        if sample.mask_path is None:
                            mask_indices.append(-1)
                        else:
                            mask_indices.append(len(masks))
                            mask = generate_mask(Path(sample.mask_path), image_size)
                            masks.append(mask)
                    images = np.stack(images)
                    masks = np.stack(masks)
                    mask_indices = np.array(mask_indices)
                    labels = np.array([s.label for s in samples], dtype=np.bool_)
                    grp.create_dataset("images", data=images, chunks=True)
                    grp.create_dataset("masks", data=masks, chunks=True)
                    grp.create_dataset("mask_indices", data=mask_indices)
                    grp.create_dataset("labels", data=labels)
        except BaseException:
            if save_path.exists():
                save_path.unlink()
            raise

    @classmethod
    def from_h5(
        cls,
        name: str,
        save_dir: Path,
        image_size: ImageSize | int | None = None,
    ) -> Self:
        h5_path = cls.get_h5_path(name, save_dir, image_size)
        print(f"Loading tensor dataset {name} from {h5_path}...")
        category_datas: dict[str, CategoryTensorDataset] = {}
        categories = cls.get_categories(name, save_dir)
        for category in categories:
            category_datas[category] = cls.CategoryDataset(category, h5_path)
        return cls(name, category_datas)


class CachedDataset(DetectionDataset):
    default_meta_save_dir = Path("data_cache/meta")
    default_tensor_save_dir = Path("data_cache/tensor")

    def __init__(
        self,
        name: str,
        data_dir: Path,
        meta_save_dir: Path | None = None,
        tensor_save_dir: Path | None = None,
    ) -> None:
        if meta_save_dir is None:
            meta_save_dir = self.default_meta_save_dir
        if tensor_save_dir is None:
            tensor_save_dir = self.default_tensor_save_dir
        meta_dataset = None
        if not MetaDataset.get_meta_csv_path(name, meta_save_dir).exists():
            category_datas = self.load_from_data_dir(data_dir)
            category_datas = {
                cat: CategoryMetaDataset(datas) for cat, datas in category_datas.items()
            }
            meta_dataset = MetaDataset(name, category_datas, data_dir)
            meta_dataset.to_csv(data_dir, meta_save_dir)

        self.name = name
        self.data_dir = data_dir
        self.tensor_save_dir = tensor_save_dir
        self.meta_save_dir = meta_save_dir
        self.meta_dataset = meta_dataset

    @override
    def get_meta_dataset(self) -> MetaDataset:
        if self.meta_dataset is None:
            self.meta_dataset = MetaDataset.from_csv(
                self.name, self.data_dir, self.meta_save_dir
            )
        return self.meta_dataset

    @override
    def get_tensor_dataset(self, image_size: ImageSize | int | None) -> TensorDataset:
        if not TensorH5Dataset.get_h5_path(
            self.name, self.tensor_save_dir, image_size
        ).exists():
            TensorH5Dataset.to_h5(
                self.get_meta_dataset(), self.tensor_save_dir, image_size
            )
        tensor_dataset = TensorH5Dataset.from_h5(
            self.name, self.tensor_save_dir, image_size
        )
        return tensor_dataset

    @classmethod
    @abstractmethod
    def load_from_data_dir(cls, data_dir: Path) -> dict[str, list[MetaSample]]: ...
