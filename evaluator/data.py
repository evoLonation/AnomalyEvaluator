from abc import abstractmethod
from dataclasses import dataclass
from itertools import dropwhile
from typing import Iterable
from PIL import Image
import numpy as np
from pathlib import Path
import pandas as pd
import json
from jaxtyping import Bool, Float
import torch
from torch.utils.data import Dataset, DataLoader
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


@dataclass
class MetaDataset:
    name: str
    category_datas: dict[str, CategoryMetaDataset]

    @staticmethod
    def get_meta_csv_path(name: str, save_dir: Path) -> Path:
        return save_dir / f"{name}_meta.csv"

    def to_csv(self, data_dir: Path, save_dir: Path):
        save_path = MetaDataset.get_meta_csv_path(self.name, save_dir)
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

    @staticmethod
    def from_csv(name: str, data_dir: Path, save_dir: Path) -> "MetaDataset":
        csv_path = MetaDataset.get_meta_csv_path(name, save_dir)
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
            category = row["category"]
            category_datas.setdefault(category, CategoryMetaDataset([])).samples.append(
                sample
            )
        return MetaDataset(name, category_datas)

    @staticmethod
    def get_categories(name: str, save_dir: Path) -> list[str]:
        csv_path = MetaDataset.get_meta_csv_path(name, save_dir)
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


def generate_masks(
    datas: list[MetaSample], image_size: ImageSize
) -> Bool[np.ndarray, "N H W"]:
    mask_paths = [x.mask_path for x in datas]
    masks = []
    for mask_path in mask_paths:
        mask = (
            generate_mask(Path(mask_path), image_size)
            if mask_path is not None
            else generate_empty_mask(image_size)
        )
        masks.append(mask)
    return np.array(masks, dtype=bool)


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
    class CategoryTensorDataset(Dataset[TensorSample]):
        def __init__(
            self,
            category: str,
            h5_file: Path,
        ):
            self.h5_file = h5_file
            self.category = category
            with h5py.File(self.h5_file, "r") as h5f:
                self.length = len(h5f[category]["images"])  # type: ignore

        def __len__(self):
            return self.length

        def __getitem__(self, idx: int) -> TensorSample:
            # 每次单独打开句柄是考虑到了线程安全性
            with h5py.File(self.h5_file, "r") as h5f:
                image = h5f[self.category]["images"][idx]  # type: ignore
                mask = h5f[self.category]["masks"][idx]  # type: ignore
                label = h5f[self.category]["labels"][idx]  # type: ignore
                return TensorSample(image=image, mask=mask, label=label)  # type: ignore

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
    category_datas: dict[str, CategoryTensorDataset]

    @staticmethod
    def get_h5_path(
        name: str, save_dir: Path, image_size: ImageSize | None = None
    ) -> Path:
        if image_size is None:
            return save_dir / f"{name}_default.h5"
        return save_dir / f"{name}_{image_size[0]}x{image_size[1]}.h5"

    @staticmethod
    def get_categories(name: str, save_dir: Path) -> list[str]:
        h5_path = TensorDataset.get_h5_path(name, save_dir)
        categories = []
        with h5py.File(h5_path, "r") as h5f:
            categories = list(h5f.keys())
        return categories

    @staticmethod
    def to_h5_default(
        dataset: MetaDataset,
        save_dir: Path,
    ):
        save_path = TensorDataset.get_h5_path(dataset.name, save_dir)
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving tensor dataset {dataset.name} to {save_path}...")

        with h5py.File(save_path, "w") as h5f:
            for category, samples in tqdm(
                dataset.category_datas.items(), desc="Saving to H5"
            ):
                grp = h5f.create_group(category)
                images = []
                masks = []
                for sample in samples:
                    img = generate_image(Path(sample.image_path))
                    images.append(img)
                    mask = (
                        generate_mask(Path(sample.mask_path))
                        if sample.mask_path is not None
                        else generate_empty_mask((img.shape[2], img.shape[1]))
                    )
                    masks.append(mask)
                images = np.stack(images)
                masks = np.stack(masks)
                grp.create_dataset("images", data=images, chunks=True, compression="gzip")
                grp.create_dataset("masks", data=masks, chunks=True, compression="gzip")
                grp.create_dataset(
                    "labels",
                    data=np.array([s.label for s in samples], dtype=np.bool),
                    chunks=True,
                )

    @staticmethod
    def to_h5(name: str, save_dir: Path, image_size: ImageSize):
        default_h5_path = TensorDataset.get_h5_path(name, save_dir, None)
        target_h5_path = TensorDataset.get_h5_path(name, save_dir, image_size)
        print(
            f"Resizing tensor dataset {name} from {default_h5_path} to {target_h5_path}..."
        )

        with (
            h5py.File(default_h5_path, "r") as h5f_in,
            h5py.File(target_h5_path, "w") as h5f_out,
        ):
            for category in tqdm(h5f_in.keys()):
                grp_in = h5f_in[category]
                grp_out = h5f_out.create_group(category)
                images = []
                masks = []
                for idx in range(len(grp_in["images"])):  # type: ignore
                    img = grp_in["images"][idx]  # pyright: ignore[reportIndexIssue]
                    img_resized = resize_image(img, image_size)  # type: ignore
                    images.append(img_resized)
                    mask = grp_in["masks"][idx]  # pyright: ignore[reportIndexIssue]
                    mask_resized = resize_mask(mask, image_size)  # type: ignore
                    masks.append(mask_resized)
                images = np.stack(images)
                masks = np.stack(masks)
                grp_out.create_dataset("images", data=images, chunks=True)
                grp_out.create_dataset("masks", data=masks, chunks=True)
                grp_out.create_dataset(
                    "labels",
                    data=grp_in["labels"][:],  # pyright: ignore[reportIndexIssue]
                    chunks=True,
                )

    @staticmethod
    def from_h5(
        name: str,
        save_dir: Path,
        image_size: ImageSize | None = None,
    ) -> "TensorDataset":
        h5_path = TensorDataset.get_h5_path(name, save_dir, image_size)
        print(f"Loading tensor dataset {name} from {h5_path}...")
        category_datas: dict[str, TensorDataset.CategoryTensorDataset] = {}
        categories = TensorDataset.get_categories(name, save_dir)
        for category in categories:
            category_datas[category] = TensorDataset.CategoryTensorDataset(
                category, h5_path
            )
        return TensorDataset(name, category_datas)


class CachedDataset:
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
            meta_dataset = MetaDataset(name, category_datas)
            meta_dataset.to_csv(data_dir, meta_save_dir)

        if not TensorDataset.get_h5_path(name, tensor_save_dir).exists():
            if meta_dataset is None:
                meta_dataset = MetaDataset.from_csv(name, data_dir, meta_save_dir)
            TensorDataset.to_h5_default(meta_dataset, tensor_save_dir)

        self.name = name
        self.data_dir = data_dir
        self.tensor_save_dir = tensor_save_dir
        self.meta_save_dir = meta_save_dir
        self.meta_dataset = meta_dataset

    def get_meta_dataset(self) -> MetaDataset:
        if self.meta_dataset is None:
            self.meta_dataset = MetaDataset.from_csv(
                self.name, self.data_dir, self.meta_save_dir
            )
        return self.meta_dataset

    def get_tensor_dataset(self, image_size: ImageSize | None = None) -> TensorDataset:
        if image_size is None:
            return TensorDataset.from_h5(self.name, self.tensor_save_dir, image_size)
        if not TensorDataset.get_h5_path(
            self.name, self.tensor_save_dir, image_size
        ).exists():
            TensorDataset.to_h5(self.name, self.tensor_save_dir, image_size)
        tensor_dataset = TensorDataset.from_h5(
            self.name, self.tensor_save_dir, image_size
        )
        return tensor_dataset

    @classmethod
    @abstractmethod
    def load_from_data_dir(cls, data_dir: Path) -> dict[str, list[MetaSample]]: ...


def generate_summary_view(
    dataset: MetaDataset,
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
        normal_samples = [s for s in samples if not s.label]
        anomaly_samples = [s for s in samples if s.label]

        # 抽样
        normal_count = min(max_samples_per_type, len(normal_samples))
        anomaly_count = min(max_samples_per_type, len(anomaly_samples))

        if normal_count > 0:
            normal_indices = np.random.choice(
                len(normal_samples), size=normal_count, replace=False
            )
            selected_normal = [normal_samples[i] for i in normal_indices]
        else:
            selected_normal = []

        if anomaly_count > 0:
            anomaly_indices = np.random.choice(
                len(anomaly_samples), size=anomaly_count, replace=False
            )
            selected_anomaly = [anomaly_samples[i] for i in anomaly_indices]
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
            img = Image.open(sample.image_path).convert("RGB")
            img = img.resize(image_size, Image.Resampling.LANCZOS)
            axes[row, col].imshow(img)
            axes[row, col].set_title("Normal", fontsize=10, color="green")
            axes[row, col].axis("off")
            idx += 1

        # 再显示异常样本
        for sample in selected_anomaly:
            row, col = idx // cols, idx % cols
            img = Image.open(sample.image_path).convert("RGB")
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
            f"{dataset.name} - {category}\n(Normal: {len(normal_samples)}, Anomaly: {len(anomaly_samples)})",
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


@dataclass
class BatchJointDataset:
    name: str
    category_datas: dict[str, CategoryMetaDataset]
    batch_size: int  # -1 means use all samples in one category as a batch


def generate_random_batch_dataset(
    base_dataset: MetaDataset, batch_size: int, seed: int = 42
) -> BatchJointDataset:
    category_datas: dict[str, CategoryMetaDataset] = {}
    rng = np.random.default_rng(seed)
    for category, samples in base_dataset.category_datas.items():
        indices = rng.choice(len(samples), size=len(samples), replace=False)
        category_datas[category] = CategoryMetaDataset([samples[i] for i in indices])

    return BatchJointDataset(
        name=f"{base_dataset.name}(b_random_{batch_size}_{seed})",
        category_datas=category_datas,
        batch_size=batch_size,
    )


def generate_all_samples_batch_dataset(
    base_dataset: MetaDataset,
) -> BatchJointDataset:
    return BatchJointDataset(
        name=f"{base_dataset.name}(b_all)",
        category_datas=base_dataset.category_datas,
        batch_size=-1,
    )


if __name__ == "__main__":
    generate_summary_view(MVTecAD().get_meta_dataset())
    generate_summary_view(VisA().get_meta_dataset())
    generate_summary_view(RealIAD().get_meta_dataset())
    generate_summary_view(RealIADDevidedByAngle().get_meta_dataset())
    generate_summary_view(MVTecLOCO().get_meta_dataset())
    generate_summary_view(MPDD().get_meta_dataset())
    generate_summary_view(BTech().get_meta_dataset())
    generate_summary_view(_3CAD().get_meta_dataset())
