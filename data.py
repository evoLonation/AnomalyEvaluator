from abc import abstractmethod
from ast import Str
from dataclasses import dataclass
from typing import overload
from PIL import Image
from jaxtyping import Bool, jaxtyped
import numpy as np
from typeguard import typechecked as typechecker
from pathlib import Path
import pandas as pd
from jaxtyping import jaxtyped
import json


@dataclass
class Sample:
    image_path: str
    mask_path: str | None
    category: str
    label: bool


@dataclass
class DetectionDataset:
    name: str
    data_dir: Path
    category_datas: dict[str, list[Sample]]

    # static member
    default_meta_save_dir = Path("meta")

    @staticmethod
    def get_meta_csv_path(name: str, save_dir: Path | None = None) -> Path:
        if save_dir is None:
            save_dir = DetectionDataset.default_meta_save_dir
        return save_dir / f"{name}_meta.csv"

    def to_csv(self, save_dir: Path | None = None):
        save_path = DetectionDataset.get_meta_csv_path(self.name, save_dir)
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "image_path": [
                Path(sample.image_path).relative_to(self.data_dir).as_posix()
                for _, samples in self.category_datas.items()
                for sample in samples
            ],
            "mask_path": [
                (
                    Path(sample.mask_path).relative_to(self.data_dir).as_posix()
                    if sample.mask_path is not None
                    else None
                )
                for _, samples in self.category_datas.items()
                for sample in samples
            ],
            "category": [
                sample.category
                for _, samples in self.category_datas.items()
                for sample in samples
            ],
            "label": [
                sample.label
                for _, samples in self.category_datas.items()
                for sample in samples
            ],
        }
        df = pd.DataFrame(data)
        df.to_csv(save_path, index=False)

    @staticmethod
    def from_csv(
        name: str, data_dir: Path, save_dir: Path | None = None
    ) -> dict[str, list[Sample]]:
        csv_path = DetectionDataset.get_meta_csv_path(name, save_dir)
        df = pd.read_csv(csv_path)
        category_datas: dict[str, list[Sample]] = {}
        for _, row in df.iterrows():
            sample = Sample(
                image_path=str(data_dir / row["image_path"]),
                mask_path=(
                    str(data_dir / row["mask_path"])
                    if not pd.isna(row["mask_path"])
                    else None
                ),
                category=str(row["category"]),
                label=bool(row["label"]),
            )
            if sample.category not in category_datas:
                category_datas[sample.category] = []
            category_datas[sample.category].append(sample)
        return category_datas


@jaxtyped(typechecker=typechecker)
def generate_masks(
    datas: list[Sample],
    image_shape: tuple[int, int],
) -> Bool[np.ndarray, "N H W"]:
    mask_paths = [x.mask_path for x in datas]
    masks = []
    for mask_path in mask_paths:
        if mask_path is None:
            masks.append(np.zeros(image_shape, dtype=bool))
            continue
        img_mask = Image.open(mask_path).convert("L")
        img_mask = (np.array(img_mask) > 0).astype(
            np.uint8
        ) * 255  # 将图片中的掩码部分变为255，非掩码部分变为0
        img_mask = Image.fromarray(img_mask, mode="L")
        # size: (W, H)
        if img_mask.size != (image_shape[1], image_shape[0]):
            # 对correct_masks进行resize, 类似下面的处理方式
            img_mask = img_mask.resize(
                (image_shape[1], image_shape[0]), Image.Resampling.BILINEAR
            )
        img_mask = np.array(img_mask)
        img_mask = img_mask > 127  # 二值化
        masks.append(img_mask)
    return np.array(masks, dtype=bool)


class CachedMetaDataset(DetectionDataset):

    def __init__(
        self,
        name: str,
        data_dir: Path,
        meta_save_dir: Path | None = None,
        sample_limit: int = -1,
    ):
        if DetectionDataset.get_meta_csv_path(name, meta_save_dir).exists():
            print(f"Loading cached meta data for dataset {name} from CSV...")
            category_datas = DetectionDataset.from_csv(
                name, data_dir, save_dir=meta_save_dir
            )
            super().__init__(name, data_dir, category_datas)
        else:
            print(f"Generating meta data for dataset {name} from data directory...")
            category_datas = self.load_from_data_dir(data_dir)
            for cat, datas in category_datas.items():
                for sample in datas:
                    assert sample.category == cat
            super().__init__(name, data_dir, category_datas)
            self.to_csv(meta_save_dir)

        if sample_limit != -1:
            sampled_category_datas: dict[str, list[Sample]] = {}
            for category, datas in self.category_datas.items():
                if len(datas) <= sample_limit:
                    sampled_category_datas[category] = datas
                    continue
                indices = np.random.choice(len(datas), size=sample_limit, replace=False)
                sampled_category_datas[category] = [datas[i] for i in indices]
            self.category_datas = sampled_category_datas

    @classmethod
    @abstractmethod
    def load_from_data_dir(cls, data_dir: Path) -> dict[str, list[Sample]]:
        pass


def generate_summary_view(
    dataset: DetectionDataset,
    save_dir: Path = Path("summary_views"),
    max_samples_per_type: int = 5,
    image_size: tuple[int, int] = (224, 224),
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


class MVTecLike(CachedMetaDataset):
    def __init__(
        self,
        name: str,
        path: Path,
        sample_limit: int = -1,
    ):
        super().__init__(name, path, sample_limit=sample_limit)

    good_category = "good"

    @classmethod
    def load_from_data_dir(cls, data_dir: Path) -> dict[str, list[Sample]]:
        categories = sorted(data_dir.iterdir())
        categories = [d.name for d in categories if d.is_dir()]

        image_suffixes = [".png", ".jpg", ".jpeg", ".JPG", ".JPEG", ".bmp"]

        category_datas: dict[str, list[Sample]] = {}
        for category in categories:
            category_dir = data_dir / category / "test"
            if not category_dir.exists():
                raise ValueError(f"Category path {category_dir} does not exist.")

            samples: list[Sample] = []

            # 加载正常样本 (good文件夹)
            good_dir = category_dir / cls.good_category
            for img_file in sorted(good_dir.iterdir()):
                assert (
                    img_file.suffix in image_suffixes
                ), f"Unsupported image format: {img_file}"
                samples.append(
                    Sample(
                        image_path=str(img_file),
                        mask_path=None,
                        category=category,
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
                        Sample(
                            image_path=str(img_file),
                            mask_path=str(mask_file),
                            category=category,
                            label=True,
                        )
                    )

            category_datas[category] = samples

        return category_datas


class MVTecAD(MVTecLike):
    def __init__(
        self,
        path: Path = Path("~/hdd/mvtec_anomaly_detection").expanduser(),
        sample_limit: int = -1,
    ):
        super().__init__("MVTecAD", path, sample_limit=sample_limit)


class VisA(MVTecLike):
    def __init__(
        self,
        path: Path = Path("~/hdd/VisA_pytorch/1cls").expanduser(),
        sample_limit: int = -1,
    ):
        super().__init__("VisA", path, sample_limit=sample_limit)


class RealIAD(CachedMetaDataset):
    def __init__(
        self, path: Path = Path("~/hdd/Real-IAD").expanduser(), sample_limit: int = -1
    ):
        super().__init__("RealIAD", path, sample_limit=sample_limit)

    @classmethod
    def load_from_data_dir(cls, data_dir: Path) -> dict[str, list[Sample]]:
        json_dir = data_dir / "realiad_jsons"
        image_dir = data_dir / "realiad_1024"
        assert json_dir.exists() and image_dir.exists()

        category_datas: dict[str, list[Sample]] = {}
        for json_file in json_dir.glob("*.json"):
            print(f"Loading dataset from {json_file}...")
            with open(json_file, "r") as f:
                data = json.load(f)
            normal_class = data["meta"]["normal_class"]
            prefix: str = data["meta"]["prefix"]
            category: str = json_file.stem

            samples: list[Sample] = []

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
                    Sample(
                        image_path=image_path,
                        mask_path=mask_path,
                        category=category,
                        label=correct_label,
                    )
                )

            category_datas[category] = samples

        return category_datas


class RealIADDevidedByAngle(CachedMetaDataset):
    def __init__(
        self, path: Path = Path("~/hdd/Real-IAD").expanduser(), sample_limit: int = -1
    ):
        super().__init__("RealIAD(angle)", path, sample_limit=sample_limit)

    @classmethod
    def load_from_data_dir(cls, data_dir: Path) -> dict[str, list[Sample]]:
        category_datas = RealIAD.load_from_data_dir(data_dir)
        divided_category_datas: dict[str, list[Sample]] = {}
        for category, samples in category_datas.items():
            angle_category_datas: dict[str, list[Sample]] = {}
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
                for sample in angle_category_datas[f"{category}_{angle_substr}"]:
                    sample.category = f"{category}_{angle_substr}"
            assert len(samples) == sum(
                len(datas) for datas in angle_category_datas.values()
            ), (
                f"Data size mismatch when dividing by angle for category {category}:"
                f" {len(samples)} vs {sum(len(datas) for datas in angle_category_datas.values())}"
            )
            divided_category_datas.update(angle_category_datas)

        return divided_category_datas


class MVTecLOCO(CachedMetaDataset):
    def __init__(
        self,
        path: Path = Path("~/hdd/mvtec_loco_anomaly_detection").expanduser(),
        sample_limit: int = -1,
    ):
        super().__init__("mvtec_loco", path, sample_limit=sample_limit)

    @classmethod
    def load_from_data_dir(cls, data_dir: Path) -> dict[str, list[Sample]]:
        meta_file = data_dir / "meta.json"
        with open(meta_file, "r") as f:
            data = json.load(f)

        category_datas: dict[str, list[Sample]] = {}
        # 只使用 test 数据进行评估
        for category, samples_data in data["test"].items():
            samples: list[Sample] = []

            for sample in samples_data:
                img_path = data_dir / sample["img_path"]
                is_anomaly = sample["anomaly"] == 1

                if is_anomaly and sample["mask_path"]:
                    mask_path = data_dir / sample["mask_path"]
                    mask_path_str = str(mask_path)
                else:
                    mask_path_str = None

                samples.append(
                    Sample(
                        image_path=str(img_path),
                        mask_path=mask_path_str,
                        category=category,
                        label=is_anomaly,
                    )
                )

            category_datas[category] = samples

        return category_datas


class MPDD(MVTecLike):
    def __init__(
        self,
        path: Path = Path("~/hdd/MPDD").expanduser(),
        sample_limit: int = -1,
    ):
        super().__init__("MPDD", path, sample_limit=sample_limit)


class BTech(MVTecLike):
    good_category = "ok"

    def __init__(
        self,
        path: Path = Path("~/hdd/BTech_Dataset_transformed").expanduser(),
        sample_limit: int = -1,
    ):
        super().__init__("BTech", path, sample_limit=sample_limit)


class _3CAD(MVTecLike):
    def __init__(
        self,
        path: Path = Path("~/hdd/3CAD").expanduser(),
        sample_limit: int = -1,
    ):
        super().__init__("3CAD", path, sample_limit=sample_limit)


@dataclass
class BatchJointDataset:
    name: str
    data_dir: Path
    category_datas: dict[str, list[Sample]]
    batch_size: int  # -1 means use all samples in one category as a batch


def generate_random_batch_dataset(
    base_dataset: DetectionDataset, batch_size: int, seed: int = 42
) -> BatchJointDataset:
    category_datas: dict[str, list[Sample]] = {}
    rng = np.random.default_rng(seed)
    for category, samples in base_dataset.category_datas.items():
        indices = rng.choice(len(samples), size=len(samples), replace=False)
        category_datas[category] = [samples[i] for i in indices]

    return BatchJointDataset(
        name=f"{base_dataset.name}(b_random_{batch_size}_{seed})",
        data_dir=base_dataset.data_dir,
        category_datas=category_datas,
        batch_size=batch_size,
    )


def generate_all_samples_batch_dataset(
    base_dataset: DetectionDataset,
) -> BatchJointDataset:
    return BatchJointDataset(
        name=f"{base_dataset.name}(b_all)",
        data_dir=base_dataset.data_dir,
        category_datas=base_dataset.category_datas,
        batch_size=-1,
    )
