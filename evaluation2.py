from abc import abstractmethod
import json
from pathlib import Path
from PIL import Image
from dataclasses import dataclass

import numpy as np
from jaxtyping import Float, Bool, jaxtyped, Int
import pandas as pd
from sklearn.metrics import auc
import torch
from tqdm import tqdm
from typeguard import typechecked as typechecker
from numpy.typing import NDArray
from torchmetrics.classification import (
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    BinaryAUROC,
    BinaryAveragePrecision,
)


@jaxtyped(typechecker=typechecker)
@dataclass
class DetectionResult:
    pred_scores: Float[np.ndarray, "N"]
    anomaly_maps: Float[np.ndarray, "N H W"]


@jaxtyped(typechecker=typechecker)
@dataclass
class DetectionGroundTruth:
    true_labels: Bool[np.ndarray, "N"]
    true_masks: Bool[np.ndarray, "N H W"]


@dataclass
class DetectionMetrics:
    precision: float
    recall: float
    f1_score: float
    auroc: float
    ap: float
    pixel_auroc: float
    pixel_aupro: float

    def __str__(self):
        return (
            f"Precision: {self.precision:.4f}, Recall: {self.recall:.4f}, F1-Score: {self.f1_score:.4f}, "
            f"Image-AUROC: {self.auroc:.4f}, Image-AP: {self.ap:.4f}, "
            f"Pixel-AUROC: {self.pixel_auroc:.4f}, Pixel-AUPro: {self.pixel_aupro:.4f}"
        )


# 摘自 AnomalyCLIP/metrics.py
# expect_fpr: 期望的假正率，只取低于这个阈值的部分来计算AUC
# PRO = 正确检测的像素数 / 真实异常区域的像素数
@jaxtyped(typechecker=typechecker)
def cal_pro_score(
    masks: Bool[np.ndarray, "N H W"],
    amaps: Float[np.ndarray, "N H W"],
    max_step=200,
    expect_fpr=0.3,
):
    from skimage import measure

    # ref: https://github.com/gudovskiy/cflow-ad/blob/master/train.py
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    min_th, max_th = amaps.min(), amaps.max()
    delta = (max_th - min_th) / max_step
    pros, fprs, ths = [], [], []

    @dataclass
    class RegionProps:
        area: int
        coord_xs: Int[np.ndarray, "N"]
        coord_ys: Int[np.ndarray, "N"]

        def __init__(self, region):
            self.area = region.area
            coords = region.coords
            self.coord_xs, self.coord_ys = coords.T

    mask_regions = [
        [RegionProps(x) for x in measure.regionprops(measure.label(mask))]
        for mask in masks
    ]
    inverse_masks = 1 - masks
    for th in tqdm(np.arange(min_th, max_th, delta)):
        binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
        pro = []
        for binary_amap, regions in zip(binary_amaps, mask_regions):
            for region in regions:
                tp_pixels = binary_amap[region.coord_xs, region.coord_ys].sum()
                pro.append(tp_pixels / region.area)
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()
        pros.append(np.array(pro).mean())
        fprs.append(fpr)
        ths.append(th)
    pros, fprs, ths = np.array(pros), np.array(fprs), np.array(ths)
    idxes = fprs < expect_fpr
    fprs = fprs[idxes]
    fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
    pro_auc = auc(fprs, pros[idxes])
    assert isinstance(pro_auc, float)
    return pro_auc


class MetricsCalculator:
    def __init__(self):
        self.precision_metric = BinaryPrecision()
        self.recall_metric = BinaryRecall()
        self.f1_metric = BinaryF1Score()
        self.auroc_metric = BinaryAUROC()
        self.ap_metric = BinaryAveragePrecision()
        self.pixel_auroc_metric = BinaryAUROC()
        # todo: 改成渐进式
        self.anomaly_maps: Float[np.ndarray, "N H W"] = np.array([])
        self.true_masks: Bool[np.ndarray, "N H W"] = np.array([])

    @jaxtyped(typechecker=typechecker)
    def update(self, preds: DetectionResult, gts: DetectionGroundTruth):
        pred_score = torch.tensor(preds.pred_scores)
        true_label = torch.tensor(gts.true_labels)
        self.precision_metric.update(pred_score, true_label)
        self.recall_metric.update(pred_score, true_label)
        self.ap_metric.update(pred_score, true_label)
        self.f1_metric.update(pred_score, true_label)
        self.auroc_metric.update(pred_score, true_label)
        self.ap_metric.update(pred_score, true_label)

        pred_score_pixel = torch.tensor(preds.anomaly_maps).flatten()
        true_mask_pixel = torch.tensor(gts.true_masks).flatten()
        self.pixel_auroc_metric.update(pred_score_pixel, true_mask_pixel)
        try:
            self.anomaly_maps = (
                np.concatenate((self.anomaly_maps, preds.anomaly_maps), axis=0)
                if self.anomaly_maps.size
                else preds.anomaly_maps
            )
            self.true_masks = (
                np.concatenate((self.true_masks, gts.true_masks), axis=0)
                if self.true_masks.size
                else gts.true_masks
            )
        except Exception as e:
            self.anomaly_maps = np.array([])
            self.true_masks = np.array([])
            print(f"Warning: Failed to concatenate arrays: {e}")

    def compute(self) -> DetectionMetrics:
        precision = self.precision_metric.compute().item()
        recall = self.recall_metric.compute().item()
        f1_score = self.f1_metric.compute().item()
        auroc = self.auroc_metric.compute().item()
        ap = self.ap_metric.compute().item()
        pixel_auroc = self.pixel_auroc_metric.compute().item()
        try:
            pixel_aupro = cal_pro_score(self.true_masks, self.anomaly_maps)
        except Exception as e:
            print(f"Warning: Failed to compute pixel_aupro: {e}")
            pixel_aupro = 0.0
        return DetectionMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            auroc=auroc,
            ap=ap,
            pixel_auroc=pixel_auroc,
            pixel_aupro=pixel_aupro,
        )


class Detector:
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def __call__(self, image_paths: list[str]) -> DetectionResult:
        pass


class DetectionDataset:
    def __init__(
        self,
        name: str,
        image_paths: list[str],
        correct_labels: list[bool],
        mask_paths: list[str | None],
    ):
        assert len(image_paths) == len(correct_labels) == len(mask_paths)
        self.name = name
        self.image_paths = image_paths
        self.correct_labels = correct_labels
        self.mask_paths = mask_paths

    @jaxtyped(typechecker=typechecker)
    def generate_masks(
        self, image_shape: tuple[int, int], start: int = 0, end: int = -1
    ) -> Bool[np.ndarray, "N H W"]:
        if end == -1:
            end = len(self.mask_paths)
        masks = []
        for mask_path in self.mask_paths[start:end]:
            if mask_path is None:
                masks.append(np.zeros(image_shape, dtype=bool))
                continue
            img_mask = Image.open(mask_path).convert("1")
            # size: (W, H)
            if img_mask.size != (image_shape[1], image_shape[0]):
                # 对correct_masks进行resize, 类似下面的处理方式
                img_mask = img_mask.resize(
                    (image_shape[1], image_shape[0]), Image.NEAREST
                )
            img_mask = np.array(img_mask)
            masks.append(img_mask)
        return np.array(masks, dtype=bool)


class MVTecAD(DetectionDataset):
    def __init__(self, path: Path, sample_limit: int = -1):
        image_paths: list[str] = []
        correct_labels: list[bool] = []
        mask_paths: list[str | None] = []

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

        for category in categories:
            category_path = path / category / "test"
            if not category_path.exists():
                raise ValueError(f"Category path {category_path} does not exist.")
            mask_path = path / category / "ground_truth"

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

        super().__init__("MVTecAD", image_paths, correct_labels, mask_paths)


class RealIAD(DetectionDataset):
    def __init__(self, path: Path, sample_limit: int = -1):
        image_paths: list[str] = []
        correct_labels: list[bool] = []
        mask_paths: list[str | None] = []

        json_dir = path / "realiad_jsons"
        image_dir = path / "realiad_1024"
        assert json_dir.exists() and image_dir.exists()
        for json_file in json_dir.glob("*.json"):
            print(f"Loading dataset from {json_file}...")
            with open(json_file, "r") as f:
                data = json.load(f)
            normal_class = data["meta"]["normal_class"]
            prefix: str = data["meta"]["prefix"]
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

        super().__init__("RealIAD", image_paths, correct_labels, mask_paths)


class VisA(DetectionDataset):
    def __init__(self, path: Path, sample_limit: int = -1):
        image_paths: list[str] = []
        correct_labels: list[bool] = []
        mask_paths: list[str | None] = []

        meta_file = path / "meta.json"
        with open(meta_file, "r") as f:
            data = json.load(f)

        # 只使用test数据进行评估
        for category, samples in data["test"].items():
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

        super().__init__("VisA", image_paths, correct_labels, mask_paths)


def evaluation_detection(
    path: Path, detector: Detector, dataset: DetectionDataset, batch_size: int = 1
):
    print(f"Evaluating detector {detector.name} on dataset {dataset.name}...")
    print(f"Dataset has {len(dataset.image_paths)} samples.")

    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    columns = [x for x in DetectionMetrics.__dataclass_fields__.keys()]
    metrics_output_path = path / f"{detector.name}_{dataset.name}_metrics.csv"
    if metrics_output_path.exists():
        print(
            f"Metrics for {detector.name} on {dataset.name} already exist. Skipping..."
        )
        return

    metrics_calculator = MetricsCalculator()
    for i in tqdm(range(0, len(dataset.image_paths), batch_size)):
        batch_image_paths = dataset.image_paths[i : i + batch_size]
        batch_correct_labels = dataset.correct_labels[i : i + batch_size]
        results = detector(batch_image_paths)
        ground_truth = DetectionGroundTruth(
            true_labels=np.array(batch_correct_labels, dtype=bool),
            true_masks=dataset.generate_masks(
                image_shape=results.anomaly_maps.shape[1:], start=i, end=i + batch_size
            ),
        )
        metrics_calculator.update(results, ground_truth)

    metrics = metrics_calculator.compute()
    print(f"Metrics: {metrics}")
    table = pd.DataFrame(
        data=[[getattr(metrics, col) for col in columns]],
        columns=columns,
        index=[f"{detector.name}_{dataset.name}"],
    )
    table.to_csv(metrics_output_path)
