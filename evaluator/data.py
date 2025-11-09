from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import dropwhile
from typing import Self, override
from PIL import Image
import numpy as np
from pathlib import Path
import pandas as pd
import json
from jaxtyping import Bool, Float
import torch
from torch.utils.data import Dataset
import h5py
from tqdm import tqdm


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


type ImageSize = tuple[int, int]  # (width, height)


def generate_image(
    image_path: Path, image_size: ImageSize | None = None
) -> Float[np.ndarray, "C=3 H W"]:
    img = Image.open(image_path).convert("RGB")
    if image_size is not None and img.size != image_size:
        img = img.resize(image_size, Image.Resampling.LANCZOS)
    img = np.array(img).astype(np.float32) / 255.0  # 归一化到0-1
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    return img


def resize_image(
    image: Float[np.ndarray, "C=3 H W"], image_size: ImageSize
) -> Float[np.ndarray, "C=3 H2 W2"]:
    if image.shape[1:] != (image_size[1], image_size[0]):
        img = np.transpose(image, (1, 2, 0))  # CHW to HWC
        img = Image.fromarray((img * 255).astype(np.uint8))
        img = img.resize(image_size, Image.Resampling.LANCZOS)
        img = np.array(img).astype(np.float32) / 255.0  # 归一化到0-1
        img = np.transpose(img, (2, 0, 1))  # HWC to CHW
        return img
    return image


def generate_mask(
    mask_path: Path, image_size: ImageSize | None = None
) -> Bool[np.ndarray, "H W"]:
    img_mask = Image.open(mask_path).convert("L")
    img_mask = (np.array(img_mask) > 0).astype(
        np.uint8
    ) * 255  # 将图片中的掩码部分变为255，非掩码部分为0
    img_mask = Image.fromarray(img_mask)
    # size: (W, H)
    if image_size is not None and img_mask.size != image_size:
        img_mask = img_mask.resize(image_size, Image.Resampling.BILINEAR)
    img_mask = np.array(img_mask)
    img_mask = img_mask > 127  # 二值化
    return img_mask.astype(bool)


def resize_mask(
    mask: Bool[np.ndarray, "H W"], image_size: ImageSize
) -> Bool[np.ndarray, "H2 W2"]:
    if mask.shape != (image_size[1], image_size[0]):
        img_mask = Image.fromarray((mask.astype(np.uint8)) * 255)
        img_mask = img_mask.resize(image_size, Image.Resampling.BILINEAR)
        img_mask = np.array(img_mask)
        img_mask = img_mask > 127  # 二值化
        return img_mask.astype(bool)
    return mask


def generate_empty_mask(image_size: ImageSize) -> Bool[np.ndarray, "H W"]:
    return np.zeros((image_size[1], image_size[0]), dtype=bool)


@dataclass
class TensorSample:
    image: Float[np.ndarray, "C=3 H W"]
    mask: Bool[np.ndarray, "H W"]
    label: bool


@dataclass
class TensorSampleBatch:
    images: Float[torch.Tensor, "N C=3 H W"]
    masks: Bool[torch.Tensor, "N H W"]
    labels: Bool[torch.Tensor, "N"]


@dataclass
class TensorDataset:
    class CategoryDataset(Dataset[TensorSample]):
        @abstractmethod
        def __len__(self) -> int: ...
        @abstractmethod
        def __getitem__(self, idx: int) -> TensorSample: ...
        @abstractmethod
        def get_labels(self) -> list[bool]: ...
        @staticmethod
        def collate_fn(batch: list[TensorSample]) -> TensorSampleBatch:
            images = np.stack([b.image for b in batch])
            masks = np.stack([b.mask for b in batch])
            labels = np.array([b.label for b in batch], dtype=bool)
            return TensorSampleBatch(
                images=torch.tensor(images),
                masks=torch.tensor(masks),
                labels=torch.tensor(labels),
            )

    name: str
    category_datas: dict[str, CategoryDataset]


class TensorH5Dataset(TensorDataset):
    class CategoryDataset(TensorDataset.CategoryDataset):
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
        cls, name: str, save_dir: Path, image_size: ImageSize | None = None
    ) -> Path:
        if image_size is None:
            return save_dir / f"{name}_default.h5"
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
        image_size: ImageSize | None = None,
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
        image_size: ImageSize | None = None,
    ) -> Self:
        h5_path = cls.get_h5_path(name, save_dir, image_size)
        print(f"Loading tensor dataset {name} from {h5_path}...")
        category_datas: dict[str, CategoryTensorDataset] = {}
        categories = cls.get_categories(name, save_dir)
        for category in categories:
            category_datas[category] = cls.CategoryDataset(category, h5_path)
        return cls(name, category_datas)


CategoryTensorDataset = TensorDataset.CategoryDataset


class DetectionDataset(ABC):
    name: str

    @abstractmethod
    def get_meta_dataset(self) -> MetaDataset: ...
    @abstractmethod
    def get_tensor_dataset(self, image_size: ImageSize | None) -> TensorDataset: ...


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
    def get_tensor_dataset(self, image_size: ImageSize | None) -> TensorDataset:
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


def generate_summary_view(
    dataset: MetaDataset | TensorDataset,
    save_dir: Path = Path("summary_views"),
    max_samples_per_type: int = 5,
    image_size: ImageSize = (224, 224),
):
    """
    为数据集的每个类别生成概览图，展示正常和异常样本

    Args:
        dataset: 检测数据集
        save_dir: 保存目录
        max_samples_per_type: 每种类型（正常/异常）最多抽样的图片数量
        image_size: 每张图片resize后的大小
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # 创建保存目录
    save_dir = save_dir / dataset.name
    save_dir.mkdir(parents=True, exist_ok=True)

    for category, samples in dataset.category_datas.items():
        # 分离正常和异常样本
        if isinstance(samples, CategoryMetaDataset):
            normal_indices = [i for i, s in enumerate(samples.samples) if not s.label]
            anomaly_indices = [i for i, s in enumerate(samples.samples) if s.label]
        else:
            normal_indices = [i for i, s in enumerate(samples.get_labels()) if not s]
            anomaly_indices = [i for i, s in enumerate(samples.get_labels()) if s]

        # 抽样
        normal_count = min(max_samples_per_type, len(normal_indices))
        anomaly_count = min(max_samples_per_type, len(anomaly_indices))

        if normal_count > 0:
            normal_indices = np.random.choice(
                len(normal_indices), size=normal_count, replace=False
            )
            if isinstance(samples, CategoryMetaDataset):
                selected_normal = [samples.samples[i] for i in normal_indices]
            else:
                selected_normal = [samples[i.item()] for i in normal_indices]
        else:
            selected_normal = []

        if anomaly_count > 0:
            anomaly_indices = np.random.choice(
                len(anomaly_indices), size=anomaly_count, replace=False
            )
            if isinstance(samples, CategoryMetaDataset):
                selected_anomaly = [samples.samples[i] for i in anomaly_indices]
            else:
                selected_anomaly = [samples[i.item()] for i in anomaly_indices]
        else:
            selected_anomaly = []

        # 计算网格布局
        total_images = normal_count + anomaly_count
        if total_images == 0:
            print(f"Warning: No samples found for category {category}")
            continue

        # 每行显示的图片数量
        cols = min(5, total_images)
        rows = (total_images + cols - 1) // cols

        # 创建图形
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)

        # 填充图片
        idx = 0

        # 先显示正常样本
        for sample in selected_normal:
            row, col = idx // cols, idx % cols
            if isinstance(sample, MetaSample):
                img = Image.open(sample.image_path).convert("RGB")
                img = img.resize(image_size, Image.Resampling.LANCZOS)
            else:
                img = sample.image
                img = np.transpose(img, (1, 2, 0))  # CHW to HWC
                img = (img * 255).astype(np.uint8)
                img = Image.fromarray(img)
                img = img.resize(image_size, Image.Resampling.LANCZOS)
            axes[row, col].imshow(img)
            axes[row, col].set_title("Normal", fontsize=10, color="green")
            axes[row, col].axis("off")
            idx += 1

        # 再显示异常样本
        for sample in selected_anomaly:
            row, col = idx // cols, idx % cols
            if isinstance(sample, MetaSample):
                img = Image.open(sample.image_path).convert("RGB")
                img = img.resize(image_size, Image.Resampling.LANCZOS)
            else:
                img = sample.image
                img = np.transpose(img, (1, 2, 0))  # CHW to HWC
                img = (img * 255).astype(np.uint8)
                img = Image.fromarray(img)
                img = img.resize(image_size, Image.Resampling.LANCZOS)
            axes[row, col].imshow(img)
            axes[row, col].set_title("Anomaly", fontsize=10, color="red")
            axes[row, col].axis("off")
            idx += 1

        # 隐藏多余的子图
        for i in range(idx, rows * cols):
            row, col = i // cols, i % cols
            axes[row, col].axis("off")

        # 添加整体标题和图例
        fig.suptitle(
            f"{dataset.name} - {category}\n(Normal: {len(normal_indices)}, Anomaly: {len(anomaly_indices)})",
            fontsize=14,
            fontweight="bold",
        )

        # 添加图例
        normal_patch = mpatches.Patch(
            color="green", label=f"Normal ({normal_count} shown)"
        )
        anomaly_patch = mpatches.Patch(
            color="red", label=f"Anomaly ({anomaly_count} shown)"
        )
        fig.legend(
            handles=[normal_patch, anomaly_patch], loc="upper right", fontsize=10
        )

        plt.tight_layout()

        # 保存图片
        save_path = save_dir / f"{category}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved summary view for category '{category}' to {save_path}")

    print(f"\nAll summary views saved to {save_dir.absolute()}")


class MVTecLike(CachedDataset):
    def __init__(
        self,
        name: str,
        path: Path,
    ):
        super().__init__(name, path)

    good_category = "good"

    @classmethod
    def load_from_data_dir(cls, data_dir: Path) -> dict[str, list[MetaSample]]:
        categories = sorted(data_dir.iterdir())
        categories = [d.name for d in categories if d.is_dir()]

        image_suffixes = [".png", ".jpg", ".jpeg", ".JPG", ".JPEG", ".bmp"]

        category_datas: dict[str, list[MetaSample]] = {}
        for category in categories:
            category_dir = data_dir / category / "test"
            if not category_dir.exists():
                raise ValueError(f"Category path {category_dir} does not exist.")

            samples: list[MetaSample] = []

            # 加载正常样本 (good文件夹)
            good_dir = category_dir / cls.good_category
            for img_file in sorted(good_dir.iterdir()):
                assert (
                    img_file.suffix in image_suffixes
                ), f"Unsupported image format: {img_file}"
                samples.append(
                    MetaSample(
                        image_path=str(img_file),
                        mask_path=None,
                        label=False,
                    )
                )

            # 加载异常样本 (除good外的所有文件夹)
            for anomaly_dir in sorted(category_dir.iterdir()):
                if not anomaly_dir.is_dir() or anomaly_dir.name == cls.good_category:
                    continue
                anomaly_mask_dir = (
                    data_dir / category / "ground_truth" / anomaly_dir.name
                )
                for img_file, mask_file in zip(
                    sorted(anomaly_dir.iterdir()), sorted(anomaly_mask_dir.iterdir())
                ):
                    assert mask_file.stem.startswith(
                        img_file.stem
                    ), f"Image and mask file names do not match: {img_file} vs {mask_file}"
                    samples.append(
                        MetaSample(
                            image_path=str(img_file),
                            mask_path=str(mask_file),
                            label=True,
                        )
                    )

            category_datas[category] = samples

        return category_datas


class MVTecAD(MVTecLike):
    def __init__(
        self,
        path: Path = Path("~/hdd/mvtec_anomaly_detection").expanduser(),
    ):
        super().__init__("MVTecAD", path)


class VisA(MVTecLike):
    def __init__(
        self,
        path: Path = Path("~/hdd/VisA_pytorch/1cls").expanduser(),
    ):
        super().__init__("VisA", path)


class RealIAD(CachedDataset):
    def __init__(self, path: Path = Path("~/hdd/Real-IAD").expanduser()):
        super().__init__("RealIAD", path)

    @classmethod
    def load_from_data_dir(cls, data_dir: Path) -> dict[str, list[MetaSample]]:
        json_dir = data_dir / "realiad_jsons"
        image_dir = data_dir / "realiad_1024"
        assert json_dir.exists() and image_dir.exists()

        category_datas: dict[str, list[MetaSample]] = {}
        for json_file in json_dir.glob("*.json"):
            print(f"Loading dataset from {json_file}...")
            with open(json_file, "r") as f:
                data = json.load(f)
            normal_class = data["meta"]["normal_class"]
            prefix: str = data["meta"]["prefix"]
            category: str = json_file.stem

            samples: list[MetaSample] = []

            for item in data["test"]:
                anomaly_class = item["anomaly_class"]
                correct_label = anomaly_class != normal_class
                image_path = image_dir / prefix / item["image_path"]
                image_path = str(image_path)
                mask_path = (
                    image_dir / prefix / item["mask_path"] if correct_label else None
                )
                mask_path = str(mask_path) if mask_path is not None else None
                samples.append(
                    MetaSample(
                        image_path=image_path,
                        mask_path=mask_path,
                        label=correct_label,
                    )
                )

            category_datas[category] = samples

        return category_datas


class RealIADDevidedByAngle(CachedDataset):
    def __init__(self, path: Path = Path("~/hdd/Real-IAD").expanduser()):
        super().__init__("RealIAD(angle)", path)

    @classmethod
    def load_from_data_dir(cls, data_dir: Path) -> dict[str, list[MetaSample]]:
        category_datas = RealIAD.load_from_data_dir(data_dir)
        divided_category_datas: dict[str, list[MetaSample]] = {}
        for category, samples in category_datas.items():
            angle_category_datas: dict[str, list[MetaSample]] = {}
            for angle_i in range(1, 6):
                angle_substr = f"C{angle_i}"
                angle_indices = [
                    i
                    for i, sample in enumerate(samples)
                    if angle_substr in sample.image_path
                ]
                angle_category_datas[f"{category}_{angle_substr}"] = [
                    samples[i] for i in angle_indices
                ]
            assert len(samples) == sum(
                len(datas) for datas in angle_category_datas.values()
            ), (
                f"Data size mismatch when dividing by angle for category {category}:"
                f" {len(samples)} vs {sum(len(datas) for datas in angle_category_datas.values())}"
            )
            divided_category_datas.update(angle_category_datas)

        return divided_category_datas


class MVTecLOCO(CachedDataset):
    def __init__(
        self,
        path: Path = Path("~/hdd/mvtec_loco_anomaly_detection").expanduser(),
    ):
        super().__init__("mvtec_loco", path)

    @classmethod
    def load_from_data_dir(cls, data_dir: Path) -> dict[str, list[MetaSample]]:
        meta_file = data_dir / "meta.json"
        with open(meta_file, "r") as f:
            data = json.load(f)

        category_datas: dict[str, list[MetaSample]] = {}
        # 只使用 test 数据进行评估
        for category, samples_data in data["test"].items():
            samples: list[MetaSample] = []

            for sample in samples_data:
                img_path = data_dir / sample["img_path"]
                is_anomaly = sample["anomaly"] == 1

                if is_anomaly and sample["mask_path"]:
                    mask_path = data_dir / sample["mask_path"]
                    mask_path_str = str(mask_path)
                else:
                    mask_path_str = None

                samples.append(
                    MetaSample(
                        image_path=str(img_path),
                        mask_path=mask_path_str,
                        label=is_anomaly,
                    )
                )

            category_datas[category] = samples

        return category_datas


class MPDD(MVTecLike):
    def __init__(
        self,
        path: Path = Path("~/hdd/MPDD").expanduser(),
    ):
        super().__init__("MPDD", path)


class BTech(MVTecLike):
    good_category = "ok"

    def __init__(
        self,
        path: Path = Path("~/hdd/BTech_Dataset_transformed").expanduser(),
    ):
        super().__init__("BTech", path)


class _3CAD(MVTecLike):
    def __init__(
        self,
        path: Path = Path("~/hdd/3CAD").expanduser(),
    ):
        super().__init__("3CAD", path)


class ReinAD(DetectionDataset):
    class CategoryDataset(TensorDataset.CategoryDataset):
        def __init__(
            self,
            category: str,
            h5_file: Path,
            image_size: ImageSize | None = None,
        ):
            self.h5_file = h5_file
            self.category = category
            self.image_size = image_size

            with h5py.File(self.h5_file, "r") as h5f:
                # 统计总图像数量
                images_group = h5f["Images"]
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
                    chunk_data = images_group[key]
                    chunk_size = chunk_data.shape[0]
                    self.chunk_info.append(
                        (key, self.length, self.length + chunk_size, True)
                    )
                    self.length += chunk_size

                # 遍历所有 Normal chunks
                for key in normal_keys:
                    chunk_data = images_group[key]
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

            with h5py.File(self.h5_file, "r") as h5f:
                # 读取图像数据 [H, W, C]
                image = h5f["Images"][chunk_name][chunk_idx]
                # 转换为 [C, H, W] 并归一化
                image = np.transpose(image, (2, 0, 1)).astype(np.float32) / 255.0

                # 读取掩码数据（如果是异常样本）
                if is_anomaly:
                    mask = h5f["Masks"][chunk_name][chunk_idx]
                    mask = mask.astype(bool)
                else:
                    # 正常样本生成空掩码
                    mask = np.zeros((image.shape[1], image.shape[2]), dtype=bool)

                # 如果需要 resize
                if self.image_size is not None:
                    image = resize_image(image, self.image_size)
                    mask = resize_mask(mask, self.image_size)

                return TensorSample(image=image, mask=mask, label=is_anomaly)

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
    def get_tensor_dataset(self, image_size: ImageSize | None) -> TensorDataset:
        category_datas: dict[str, CategoryTensorDataset] = {}

        # 遍历 test 目录下的所有 .h5 文件
        for h5_file in sorted(self.test_dir.glob("*.h5")):
            # 从文件名提取 category
            category = h5_file.stem
            category_datas[category] = self.CategoryDataset(
                category, h5_file, image_size
            )

        return TensorDataset(name="ReinAD", category_datas=category_datas)


if __name__ == "__main__":
    for dataset in list[CachedDataset](
        [
            MVTecAD(),
            VisA(),
            RealIAD(),
            RealIADDevidedByAngle(),
            MVTecLOCO(),
            MPDD(),
            BTech(),
            _3CAD(),
        ]
    ):
        generate_summary_view(dataset.get_meta_dataset())
        dataset.get_tensor_dataset((336, 336))
        dataset.get_tensor_dataset((518, 518))

    dataset = ReinAD()
    generate_summary_view(dataset.get_tensor_dataset(None))
