from pathlib import Path
from abc import abstractmethod
from typing import Self, override

import h5py
from h5py import Group, Dataset as H5Dataset
import numpy as np
import torch
from tqdm import tqdm

from data.base import ListDataset

from .utils import (
    ImageResize,
    ImageSize,
    Transform,
    generate_image,
    generate_mask,
    normalize_image,
    generate_empty_mask,
)
from .detection_dataset import (
    Dataset,
    MetaInfo,
    MetaDataset,
    MetaSample,
    TensorSample,
)


class DatasetByH5(Dataset[TensorSample]):
    def __init__(
        self,
        category: str,
        h5_path: Path,
        transform: Transform,
    ):
        self._category = category
        self._h5_path = h5_path
        self._transform = transform
        with h5py.File(self._h5_path, "r") as h5f:
            images: H5Dataset = h5f[self._category]["images"]  # type: ignore
            self._length = len(images)

    @override
    def __len__(self) -> int:
        return self._length

    @override
    def __getitem__(self, idx: int) -> TensorSample:
        # 每次单独打开句柄是考虑到了线程安全性
        with h5py.File(self._h5_path, "r") as h5f:
            category_data: Group = h5f[self._category]  # type: ignore
            images: H5Dataset = category_data["images"]  # type: ignore
            masks: H5Dataset = category_data["masks"]  # type: ignore
            mask_indices: H5Dataset = category_data["mask_indices"]  # type: ignore
            labels: H5Dataset = category_data["labels"]  # type: ignore
            image = images[idx]
            image = normalize_image(image)
            mask_index: int = mask_indices[idx].item()
            if mask_index == -1:
                mask = generate_empty_mask(ImageSize.fromnumpy(image.shape))
            else:
                mask = masks[mask_index]
            image = self._transform.image_transform(torch.tensor(image))
            mask = self._transform.mask_transform(torch.tensor(mask))
            label: bool = labels[idx].item()
            return TensorSample(image=image, mask=mask, label=label)

    @classmethod
    def to_h5(
        cls,
        meta_info: MetaInfo,
        save_path: Path,
        resize: ImageResize | None = None,
    ):
        print(f"Saving tensor dataset to {save_path}...")

        try:
            with h5py.File(save_path, "w") as h5f:
                for category, samples in tqdm(
                    meta_info.category_datas.items(), desc="Saving to H5"
                ):
                    grp = h5f.create_group(category)
                    images = []
                    masks = []
                    mask_indices = []
                    for sample in samples:
                        img = generate_image(Path(sample.image_path), resize)
                        images.append(img)
                        if sample.mask_path is None:
                            mask_indices.append(-1)
                        else:
                            mask_indices.append(len(masks))
                            mask = generate_mask(Path(sample.mask_path), resize)
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


class CachedDataset(MetaDataset):
    default_meta_save_dir = Path("data_cache/meta")
    default_tensor_save_dir = Path("data_cache/tensor")

    def __init__(
        self,
        name: str,
        data_dir: Path,
        meta_save_dir: Path = default_meta_save_dir,
        tensor_save_dir: Path = default_tensor_save_dir,
        meta_split_category: bool = False,
        is_zip_file: bool = False,
    ) -> None:
        csv_path = self.get_meta_csv_path(name, meta_save_dir, meta_split_category)
        if not csv_path.exists():
            category_datas = self.load_from_data_dir(data_dir)
            category_datas_: dict[str, Dataset[MetaSample]] = {
                k: ListDataset(v) for k, v in category_datas.items()
            }
            meta_info = MetaInfo(data_dir, category_datas_)
            meta_info.to_csv(csv_path, split_category=meta_split_category)
        else:
            meta_info = MetaInfo.from_csv(
                data_dir, csv_path, split_category=meta_split_category
            )
        self._tensor_save_dir = tensor_save_dir
        super().__init__(name, meta_info, is_zip_file=is_zip_file)

    def get_tensor(self, category: str, transform: Transform) -> Dataset[TensorSample]:
        h5_path = self.get_h5_path(
            self.get_name(), self._tensor_save_dir, transform.resize
        )
        if h5_path.exists():
            return DatasetByH5(category=category, h5_path=h5_path, transform=transform)
        return super().get_tensor(category, transform)

    def cache(self):
        h5_path = self.get_h5_path(
            self.get_name(), self._tensor_save_dir, self.get_transform().resize
        )
        if not h5_path.exists():
            self._tensor_save_dir.mkdir(parents=True, exist_ok=True)
            DatasetByH5.to_h5(
                self.get_meta_info(), h5_path, self.get_transform().resize
            )

    @classmethod
    def get_meta_csv_path(cls, name: str, save_dir: Path, split_category: bool) -> Path:
        path = save_dir / f"{name}_meta"
        if not split_category:
            path = path.with_suffix(".csv")
        return path

    @classmethod
    def get_h5_path(
        cls, name: str, save_dir: Path, resize: ImageResize | None = None
    ) -> Path:
        if resize is None:
            return save_dir / f"{name}_default.h5"
        if isinstance(resize, int):
            return save_dir / f"{name}_{resize}.h5"
        return save_dir / f"{name}_{resize.w}x{resize.h}.h5"

    @classmethod
    @abstractmethod
    def load_from_data_dir(cls, data_dir: Path) -> dict[str, list[MetaSample]]: ...
