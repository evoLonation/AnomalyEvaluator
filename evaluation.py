import json
from pathlib import Path
from PIL import Image
from dataclasses import dataclass
import numpy as np
from jaxtyping import Bool, jaxtyped
import pandas as pd
from tqdm import tqdm
from typeguard import typechecked as typechecker
from detector import DetectionGroundTruth, Detector
from metrics import MetricsCalculator, DetectionMetrics


@dataclass
class CategoryData:
    category: str
    image_paths: list[str]
    correct_labels: list[bool]
    mask_paths: list[str | None]

    def __post_init__(self):
        assert len(self.image_paths) == len(self.correct_labels) == len(self.mask_paths)


class DetectionDataset:
    def __init__(
        self,
        name: str,
        category_datas: list[CategoryData],
    ):
        self.name = name
        self.category_datas = category_datas
        self.category2data = {data.category: data for data in category_datas}
        assert len(self.category2data) == len(
            category_datas
        ), "Duplicate category names found."

    @jaxtyped(typechecker=typechecker)
    def generate_masks(
        self, category: str, image_shape: tuple[int, int], start: int = 0, end: int = -1
    ) -> Bool[np.ndarray, "N H W"]:
        mask_paths = self.category2data[category].mask_paths
        if end == -1:
            end = len(mask_paths)
        masks = []
        for mask_path in mask_paths[start:end]:
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


class MVTecAD(DetectionDataset):
    def __init__(self, path: Path, sample_limit: int = -1):
        # MVTec数据集类别列表
        categories = [
            "bottle",
            "cable",
            "capsule",
            "carpet",
            "grid",
            "hazelnut",
            "leather",
            "metal_nut",
            "pill",
            "screw",
            "tile",
            "toothbrush",
            "transistor",
            "wood",
            "zipper",
        ]

        category_datas = []
        for category in categories:
            category_path = path / category / "test"
            if not category_path.exists():
                raise ValueError(f"Category path {category_path} does not exist.")
            mask_path = path / category / "ground_truth"

            image_paths: list[str] = []
            correct_labels: list[bool] = []
            mask_paths: list[str | None] = []

            # 加载正常样本 (good文件夹)
            good_path = category_path / "good"
            for img_file in good_path.glob("*.png"):
                image_paths.append(str(img_file))
                correct_labels.append(False)
                mask_paths.append(None)

            # 加载异常样本 (除good外的所有文件夹)
            for anomaly_dir in category_path.iterdir():
                if anomaly_dir.is_dir() and anomaly_dir.name != "good":
                    for img_file in anomaly_dir.glob("*.png"):
                        prefix, suffix = img_file.name.split(".", 2)
                        mask_file = (
                            mask_path / anomaly_dir.name / (f"{prefix}_mask.{suffix}")
                        )
                        image_paths.append(str(img_file))
                        correct_labels.append(True)
                        mask_paths.append(str(mask_file))

            if 0 < sample_limit <= len(image_paths):
                indices = np.random.choice(
                    len(image_paths), size=sample_limit, replace=False
                )
                image_paths = [image_paths[i] for i in indices]
                correct_labels = [correct_labels[i] for i in indices]
                mask_paths = [mask_paths[i] for i in indices]

            category_datas.append(
                CategoryData(
                    category=category,
                    image_paths=image_paths,
                    correct_labels=correct_labels,
                    mask_paths=mask_paths,
                )
            )

        super().__init__("MVTecAD", category_datas)


class RealIAD(DetectionDataset):
    def __init__(self, path: Path, sample_limit: int = -1):
        json_dir = path / "realiad_jsons"
        image_dir = path / "realiad_1024"
        assert json_dir.exists() and image_dir.exists()

        category_datas = []
        for json_file in json_dir.glob("*.json"):
            print(f"Loading dataset from {json_file}...")
            with open(json_file, "r") as f:
                data = json.load(f)
            normal_class = data["meta"]["normal_class"]
            prefix: str = data["meta"]["prefix"]
            category: str = json_file.stem

            image_paths: list[str] = []
            correct_labels: list[bool] = []
            mask_paths: list[str | None] = []

            for item in data["test"]:
                anomaly_class = item["anomaly_class"]
                correct_label = anomaly_class != normal_class
                image_path = image_dir / prefix / item["image_path"]
                image_path = str(image_path)
                mask_path = (
                    image_dir / prefix / item["mask_path"] if correct_label else None
                )
                mask_path = str(mask_path) if mask_path is not None else None
                image_paths.append(image_path)
                correct_labels.append(correct_label)
                mask_paths.append(mask_path)

            if 0 < sample_limit <= len(image_paths):
                indices = np.random.choice(
                    len(image_paths), size=sample_limit, replace=False
                )
                image_paths = [image_paths[i] for i in indices]
                correct_labels = [correct_labels[i] for i in indices]
                mask_paths = [mask_paths[i] for i in indices]

            category_datas.append(
                CategoryData(
                    category=category,
                    image_paths=image_paths,
                    correct_labels=correct_labels,
                    mask_paths=mask_paths,
                )
            )

        super().__init__("RealIAD", category_datas)


class VisA(DetectionDataset):
    def __init__(self, path: Path, sample_limit: int = -1):
        # 类似 MVTecAD 的目录结构：
        # <path>/<category>/
        #   ├── train/good/*
        #   ├── test/good/*
        #   ├── test/bad/*.{jpg,png,...}
        #   └── ground_truth/bad/*.png  (与 bad 图像同名，扩展名为 .png)

        # 自动收集所有类别目录
        categories = [d.name for d in path.iterdir() if d.is_dir()]

        category_datas = []
        for category in categories:
            category_path = path / category / "test"
            if not category_path.exists():
                raise ValueError(f"Category path {category_path} does not exist.")

            gt_bad_path = path / category / "ground_truth" / "bad"

            image_paths: list[str] = []
            correct_labels: list[bool] = []
            mask_paths: list[str | None] = []

            # 加载正常样本 (good 文件夹)，无掩码
            good_path = category_path / "good"
            for pattern in ["*.png", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]:
                for img_file in good_path.glob(pattern):
                    image_paths.append(str(img_file))
                    correct_labels.append(False)
                    mask_paths.append(None)

            # 加载异常样本 (bad 文件夹)，掩码与图像同名且为 .png
            bad_path = category_path / "bad"
            for pattern in ["*.png", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]:
                for img_file in bad_path.glob(pattern):
                    prefix = img_file.stem  # 例如 000 (来自 000.JPG)
                    mask_file = gt_bad_path / f"{prefix}.png"
                    image_paths.append(str(img_file))
                    correct_labels.append(True)
                    mask_paths.append(str(mask_file))

            if 0 < sample_limit <= len(image_paths):
                indices = np.random.choice(
                    len(image_paths), size=sample_limit, replace=False
                )
                image_paths = [image_paths[i] for i in indices]
                correct_labels = [correct_labels[i] for i in indices]
                mask_paths = [mask_paths[i] for i in indices]

            category_datas.append(
                CategoryData(
                    category=category,
                    image_paths=image_paths,
                    correct_labels=correct_labels,
                    mask_paths=mask_paths,
                )
            )

        super().__init__("VisA", category_datas)


class MVTecLOCO(DetectionDataset):
    def __init__(self, path: Path, sample_limit: int = -1):
        meta_file = path / "meta.json"
        with open(meta_file, "r") as f:
            data = json.load(f)

        category_datas = []
        # 只使用 test 数据进行评估
        for category, samples in data["test"].items():
            image_paths: list[str] = []
            correct_labels: list[bool] = []
            mask_paths: list[str | None] = []

            for sample in samples:
                img_path = path / sample["img_path"]
                image_paths.append(str(img_path))

                is_anomaly = sample["anomaly"] == 1
                correct_labels.append(is_anomaly)

                if is_anomaly and sample["mask_path"]:
                    mask_path = path / sample["mask_path"]
                    mask_paths.append(str(mask_path))
                else:
                    mask_paths.append(None)

            if 0 < sample_limit <= len(image_paths):
                indices = np.random.choice(
                    len(image_paths), size=sample_limit, replace=False
                )
                image_paths = [image_paths[i] for i in indices]
                correct_labels = [correct_labels[i] for i in indices]
                mask_paths = [mask_paths[i] for i in indices]

            category_datas.append(
                CategoryData(
                    category=category,
                    image_paths=image_paths,
                    correct_labels=correct_labels,
                    mask_paths=mask_paths,
                )
            )

        super().__init__("MVTecLOCO", category_datas)


def evaluation_detection(
    path: Path, detector: Detector, dataset: DetectionDataset, batch_size: int = 1
):
    print(f"Evaluating detector {detector.name} on dataset {dataset.name}...")

    total_samples = sum(len(data.image_paths) for data in dataset.category_datas)
    print(
        f"Dataset has {total_samples} samples across {len(dataset.category_datas)} categories."
    )

    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    metrics_output_path = path / f"{detector.name}_{dataset.name}_metrics.csv"
    category_metrics: list[tuple[str, DetectionMetrics]] = []
    if metrics_output_path.exists():
        print(f"Metrics for {detector.name} on {dataset.name} already exist.")
        table = pd.read_csv(metrics_output_path, index_col=0)
        if "Average" in table.index:
            print(f"All metrics already computed. Skipping evaluation.")
            return
        else:
            existing_categories = set(table.index)
            print(f"Existing categories: {existing_categories}")
    else:
        existing_categories = set()
        table = pd.DataFrame(
            columns=[x for x in DetectionMetrics.__dataclass_fields__.keys()]
        )

    for data in dataset.category_datas:
        if data.category in existing_categories:
            print(f"Metrics for category {data.category} already exist. Skipping.")
            continue
        print(f"Evaluating category: {data.category}")
        metrics_calculator = MetricsCalculator(type(detector))

        for i in tqdm(
            range(0, len(data.image_paths), batch_size),
            desc=f"Processing {data.category}",
        ):
            batch_image_paths = data.image_paths[i : i + batch_size]
            batch_correct_labels = data.correct_labels[i : i + batch_size]
            results = detector(batch_image_paths, data.category)
            ground_truth = DetectionGroundTruth(
                true_labels=np.array(batch_correct_labels, dtype=bool),
                true_masks=dataset.generate_masks(
                    category=data.category,
                    image_shape=results.anomaly_maps.shape[1:],
                    start=i,
                    end=i + batch_size,
                ),
            )
            metrics_calculator.update(results, ground_truth)

        metrics = metrics_calculator.compute()
        category_metrics.append((data.category, metrics))
        table.loc[data.category] = [getattr(metrics, col) for col in table.columns]
        table.to_csv(metrics_output_path)  # 每计算完一个类别就保存一次，防止意外中断
        print(f"Category {data.category} metrics saved: {metrics}")

    # 计算平均指标
    table.loc["Average"] = [table[col].mean() for col in table.columns]
    table.to_csv(metrics_output_path)
    print(f"Average metrics saved: {table.loc['Average']}")
    print(f"Evaluation of {detector.name} on {dataset.name} completed.")
