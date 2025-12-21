from bisect import bisect_left
from dataclasses import dataclass
from typing import Iterator
import torch
from torch.utils.data import Sampler, RandomSampler

from data.base import Dataset
from data.detection_dataset import DetectionDataset, TensorSample, TensorSampleBatch
from data.utils import Transform


@dataclass
class MixedSampleBatch(TensorSampleBatch):
    categories: list[str]


@dataclass
class MixedSample(TensorSample):
    category: str

    @staticmethod
    def collate_fn(batch: list["MixedSample"]) -> "MixedSampleBatch":  # type: ignore
        # assert all([x.category == batch[0].category for x in batch])
        batch_tensor = TensorSample.collate_fn(batch)  # type: ignore
        return MixedSampleBatch(
            images=batch_tensor.images,
            masks=batch_tensor.masks,
            labels=batch_tensor.labels,
            categories=[x.category for x in batch],
        )


class MixedDataset(Dataset[MixedSample]):
    def __init__(self, dataset: DetectionDataset, transform: Transform):
        self.dataset = dataset
        self.transform = transform
        self.category_list = dataset.get_categories()
        self.category_to_idx: dict[str, int] = {
            category: i for i, category in enumerate(self.category_list)
        }
        self.start_idx_list: list[int] = []
        current_idx = 0
        for category in self.category_list:
            self.start_idx_list.append(current_idx)
            current_idx += dataset.get_sample_count(category)
        self.total_count = current_idx
        self.tensor_data_list: list[Dataset[TensorSample]] = [
            dataset.get_tensor(category, transform) for category in self.category_list
        ]

    def __len__(self) -> int:
        return self.total_count

    def __getitem__(self, index: int) -> MixedSample:
        # bisect_left: 找到第一个大于 index 的 start_idx，然后减一得到类别索引
        category_i = bisect_left(self.start_idx_list, index + 1) - 1
        category = self.category_list[category_i]
        category_start_idx = self.start_idx_list[category_i]
        sample_idx = index - category_start_idx
        sample = self.tensor_data_list[category_i][sample_idx]
        return MixedSample(
            image=sample.image,
            mask=sample.mask,
            label=sample.label,
            category=category,
        )

    def get_category_start_idx(self, category: str) -> int:
        category_i = self.category_to_idx[category]
        return self.start_idx_list[category_i]

    def get_origin(self) -> DetectionDataset:
        return self.dataset


class MixedBatchSampler(Sampler):
    """
    batch 内部的类别是相同的，但不同 batch 之间类别可以不同，通过 category_random 控制是否随机打乱类别顺序
    如果 normal 为 True，则每个类别内只采样正常样本，且允许重复采样
    """

    def __init__(
        self,
        mixed_dataset: MixedDataset,
        seed: int,
        batch_size: int,
        category_random: bool,
        normal: bool,
    ):
        self.mixed_dataset = mixed_dataset
        self.dataset = mixed_dataset.get_origin()
        self.seed = seed
        self.batch_size = batch_size
        self.normal = normal
        self.category_list = self.dataset.get_categories()
        self.category_num = len(self.category_list)
        self.batch_num_list: list[int] = [
            (self.dataset.get_sample_count(category) + batch_size - 1) // batch_size
            for category in self.category_list
        ]
        # 一系列类别索引，每个类别索引重复对应的批次数
        self.batches: list[int] = []
        for cat_i, batch_num in enumerate(self.batch_num_list):
            self.batches.extend([cat_i] * batch_num)
        if category_random:
            self.sampler = RandomSampler(
                self.batches,
                replacement=False,
                generator=torch.Generator().manual_seed(self.seed),
            )
        else:
            self.sampler = torch.utils.data.SequentialSampler(self.batches)

        if not normal:
            self.samplers: list[RandomSampler] = [
                RandomSampler(
                    self.dataset.get_labels(category),
                    replacement=False,
                    generator=torch.Generator().manual_seed(self.seed),
                )
                for category in self.category_list
            ]
        else:
            self.normal_indices_list: list[list[int]] = [
                [i for i, x in enumerate(self.dataset.get_labels(category)) if x == 0]
                for category in self.category_list
            ]
            self.normal_samplers: list[RandomSampler] = [
                RandomSampler(
                    self.normal_indices_list[i],
                    replacement=True,
                    num_samples=self.dataset.get_sample_count(self.category_list[i]),
                    generator=torch.Generator().manual_seed(self.seed),
                )
                for i in range(self.category_num)
            ]

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self) -> Iterator[list[int]]:
        sampler_iters: list[Iterator[int]] = [
            iter(sampler)
            for sampler in (self.samplers if not self.normal else self.normal_samplers)
        ]
        for category_i in [self.batches[i] for i in iter(self.sampler)]:
            category = self.category_list[category_i]
            batch_indices: list[int] = []
            for _ in range(self.batch_size):
                try:
                    sample_idx = next(sampler_iters[category_i])
                    if self.normal:
                        sample_idx = self.normal_indices_list[category_i][sample_idx]
                    sample_idx = (
                        self.mixed_dataset.get_category_start_idx(category) + sample_idx
                    )
                    batch_indices.append(sample_idx)
                except StopIteration:
                    break
            assert len(batch_indices) > 0
            yield batch_indices


class MixedInBatchSampler(Sampler):
    """
    完全随机的，batch 内部的类别可以不同
    """

    def __init__(
        self,
        mixed_dataset: MixedDataset,
        seed: int,
        batch_size: int,
        normal: bool,
    ):
        self._base = MixedBatchSampler(
            mixed_dataset, seed, 1, category_random=True, normal=normal
        )
        self.batch_size = batch_size

    def __len__(self) -> int:
        return (len(self._base) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[list[int]]:
        batch_indices: list[int] = []
        for sample_idx_list in iter(self._base):
            batch_indices.extend(sample_idx_list)
            if len(batch_indices) >= self.batch_size:
                assert len(batch_indices) == self.batch_size
                yield batch_indices
                batch_indices = []
        if len(batch_indices) > 0:
            yield batch_indices