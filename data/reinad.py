from pathlib import Path
import h5py
import numpy as np
from typing import cast, override

import torch

from .detection_dataset import (
    DetectionDataset,
    MetaDataset,
    TensorDataset,
    CategoryTensorDataset,
    TensorSample,
)
from .utils import ImageResize, resize_image, resize_mask, ImageSize


class ReinAD(DetectionDataset):
    class CategoryDataset(CategoryTensorDataset):
        def __init__(
            self,
            category: str,
            h5_file: Path,
            resize: ImageResize | None = None,
        ):
            self.h5_file = h5_file
            self.category = category
            self.resize = resize

            with h5py.File(self.h5_file, "r") as h5f:
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
        def get_item(self, idx: int) -> TensorSample:
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

            with h5py.File(self.h5_file, "r") as h5f:
                # 读取图像数据 [H, W, C]
                image = cast(h5py.Dataset, cast(h5py.Group, h5f["Images"])[chunk_name])[
                    chunk_idx
                ]
                # 转换为 [C, H, W] 并归一化
                image = np.transpose(image, (2, 0, 1)).astype(np.float32) / 255.0

                # 读取掩码数据（如果是异常样本）
                if is_anomaly:
                    mask = cast(
                        h5py.Dataset, cast(h5py.Group, h5f["Masks"])[chunk_name]
                    )[chunk_idx]
                    mask = mask.astype(bool)
                else:
                    # 正常样本生成空掩码
                    mask = np.zeros((image.shape[1], image.shape[2]), dtype=bool)

                # 如果需要 resize
                if self.resize is not None:
                    image = resize_image(image, self.resize)
                    mask = resize_mask(mask, self.resize)

                return TensorSample(
                    image=torch.tensor(image), mask=torch.tensor(mask), label=is_anomaly
                )

        @override
        def get_labels(self) -> list[bool]:
            labels = []
            with h5py.File(self.h5_file, "r") as h5f:
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

    def get_meta_dataset(self) -> MetaDataset:
        raise NotImplementedError(
            "ReinAD dataset is already in HDF5 format, use get_tensor_dataset directly"
        )

    @override
    def get_tensor_dataset_impl(self, resize: ImageResize | None) -> TensorDataset:
        category_datas: dict[str, CategoryTensorDataset] = {}

        # 遍历 test 目录下的所有 .h5 文件
        for h5_file in sorted(self.test_dir.glob("*.h5")):
            # 从文件名提取 category
            category = h5_file.stem
            category_datas[category] = self.CategoryDataset(category, h5_file, resize)

        return TensorDataset(name="ReinAD", category_datas=category_datas)
