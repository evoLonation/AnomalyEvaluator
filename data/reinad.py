from pathlib import Path
import h5py
import numpy as np
from typing import cast, override
from jaxtyping import UInt8

import torch

from .detection_dataset import (
    Dataset,
    DetectionDataset,
    ListDataset,
    TensorSample,
)
from .utils import (
    Transform,
    generate_empty_mask,
    normalize_image,
    resize_image,
    resize_mask,
    ImageSize,
    to_tensor_image,
)


class ReinAD(DetectionDataset):
    class CategoryDataset(Dataset[TensorSample]):
        def __init__(
            self,
            category: str,
            h5_path: Path,
            transform: Transform,
        ):
            self._h5_path = h5_path
            self._category = category
            self._transform = transform

            with h5py.File(self._h5_path, "r") as h5f:
                # 统计总图像数量
                images_group = cast(h5py.Group, h5f["Images"])
                self.length = 0
                self.chunk_info = (
                    []
                )  # 存储 (chunk_name, start_idx, end_idx, is_anomaly)

                # 先处理 Anomaly chunks，然后处理 Normal chunks
                # 分别对 Anomaly 和 Normal 的 key 进行排序
                anomaly_keys = sorted(
                    [k for k in images_group.keys() if k.startswith("Anomaly_")]
                )
                normal_keys = sorted(
                    [k for k in images_group.keys() if k.startswith("Normal_")]
                )

                # 遍历所有 Anomaly chunks
                for key in anomaly_keys:
                    chunk_data = cast(h5py.Dataset, images_group[key])
                    chunk_size = chunk_data.shape[0]
                    self.chunk_info.append(
                        (key, self.length, self.length + chunk_size, True)
                    )
                    self.length += chunk_size

                # 遍历所有 Normal chunks
                for key in normal_keys:
                    chunk_data = cast(h5py.Dataset, images_group[key])
                    chunk_size = chunk_data.shape[0]
                    self.chunk_info.append(
                        (key, self.length, self.length + chunk_size, False)
                    )
                    self.length += chunk_size

        @override
        def __len__(self) -> int:
            return self.length

        @override
        def __getitem__(self, idx: int) -> TensorSample:
            if idx < 0 or idx >= self.length:
                raise IndexError(f"Index {idx} out of range [0, {self.length})")

            # 找到对应的 chunk
            chunk_name = None
            chunk_idx = 0
            is_anomaly = False

            for name, start, end, anomaly in self.chunk_info:
                if start <= idx < end:
                    chunk_name = name
                    chunk_idx = idx - start
                    is_anomaly = anomaly
                    break

            assert (
                chunk_name is not None
            ), f"Invalid index {idx}, chunk_info: {self.chunk_info}"

            with h5py.File(self._h5_path, "r") as h5f:
                # 读取图像数据 [H, W, C]
                image = cast(h5py.Dataset, cast(h5py.Group, h5f["Images"])[chunk_name])[
                    chunk_idx
                ]
                image = to_tensor_image(image)

                # 读取掩码数据（如果是异常样本）
                if is_anomaly:
                    mask = cast(
                        h5py.Dataset, cast(h5py.Group, h5f["Masks"])[chunk_name]
                    )[chunk_idx]
                    mask = mask.astype(bool)
                    mask = torch.tensor(mask)
                else:
                    # 正常样本生成空掩码
                    mask = generate_empty_mask(ImageSize.fromtensor(image))

                if self._transform.resize is not None:
                    image = resize_image(image, self._transform.resize)
                    mask = resize_mask(mask, self._transform.resize)
                image = normalize_image(image)
                image = self._transform.image_transform(torch.tensor(image))
                mask = self._transform.mask_transform(torch.tensor(mask))

                return TensorSample(image=image, mask=mask, label=is_anomaly)

        def get_labels(self) -> list[bool]:
            labels = []
            with h5py.File(self._h5_path, "r") as h5f:
                for name, start, end, is_anomaly in self.chunk_info:
                    chunk_size = end - start
                    labels.extend([is_anomaly] * chunk_size)
            return labels

    def __init__(
        self,
        path: Path = Path("~/hdd/ReinAD").expanduser(),
    ):
        self.name = "ReinAD"
        self.path = path
        self.test_dir = path / "test"
        assert self.test_dir.exists(), f"Test directory {self.test_dir} does not exist"
        categories = []
        for h5_file in sorted(self.test_dir.glob("*.h5")):
            # 从文件名提取 category
            category = h5_file.stem
            categories.append(category)
        super().__init__("ReinAD", categories)

    def get_tensor(self, category: str, transform: Transform) -> CategoryDataset:
        h5_path = self.test_dir / f"{category}.h5"
        return self.CategoryDataset(
            category=category, h5_path=h5_path, transform=transform
        )

    @override
    def get_labels(self, category: str) -> Dataset[bool]:
        return ListDataset(self.get_tensor(category, self.get_transform()).get_labels())
