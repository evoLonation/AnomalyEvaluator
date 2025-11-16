from abc import abstractmethod
from torch.utils.data import Dataset as TorchDataset
from typing import Callable


class Dataset[T](TorchDataset[T]):
    @abstractmethod
    def __len__(self) -> int: ...
    @abstractmethod
    def __getitem__(self, index: int) -> T: ...

    @staticmethod
    def bypt(dataset: TorchDataset[T]) -> "Dataset[T]":
        return DatasetByTorch(dataset)


class DatasetByTorch[T](Dataset[T]):
    def __init__(
        self,
        torch_dataset: TorchDataset[T],
    ):
        self._torch_dataset = torch_dataset
        assert hasattr(self._torch_dataset, "__len__")

    def __len__(self) -> int:
        return len(self._torch_dataset)  # type: ignore

    def __getitem__(self, index: int) -> T:
        return self._torch_dataset[index]


class ListDataset[T](Dataset[T]):
    def __init__(self, samples: list[T]):
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> T:
        return self._samples[index]


def tuple_collate_fn[T1, T2, B1, B2](
    batch: list[tuple[T1, T2]],
    t1_collate_fn: Callable[[list[T1]], B1] = lambda x: x,
    t2_collate_fn: Callable[[list[T2]], B2] = lambda x: x,
) -> tuple[B1, B2]:
    items1 = [b[0] for b in batch]
    items2 = [b[1] for b in batch]
    if t1_collate_fn is not None:
        items1 = t1_collate_fn(items1)
    if t2_collate_fn is not None:
        items2 = t2_collate_fn(items2)
    return (items1, items2)


class ZipedDataset[T1, T2](Dataset[tuple[T1, T2]]):
    def __init__(self, dataset1: Dataset[T1], dataset2: Dataset[T2]):
        assert len(dataset1) == len(dataset2)
        self._dataset1 = dataset1
        self._dataset2 = dataset2
        assert len(self._dataset1) == len(self._dataset2)

    def __len__(self) -> int:
        return len(self._dataset1)

    def __getitem__(self, index: int) -> tuple[T1, T2]:
        return (self._dataset1[index], self._dataset2[index])


class DatasetWithIndex[T](Dataset[tuple[int, T]]):
    def __init__(self, base_dataset: Dataset[T]):
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> tuple[int, T]:
        item = self.base_dataset[index]
        return index, item

class DatasetOverrideGetItem[T](Dataset[T]):
    def __init__(self, base_dataset: Dataset[T], getitem_override: Callable[[int], T]):
        self.base_dataset = base_dataset
        self.getitem_override = getitem_override

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> T:
        return self.getitem_override(index)