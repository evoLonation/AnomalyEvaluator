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
    # precision: float
    # recall: float
    # f1_score: float
    auroc: float
    ap: float
    pixel_auroc: float
    pixel_aupro: float

    def __str__(self):
        return (
            # f"Precision: {self.precision:.4f}, Recall: {self.recall:.4f}, F1-Score: {self.f1_score:.4f}, "
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
    inverse_masks_sum = inverse_masks.sum()
    for th in tqdm(np.arange(min_th, max_th, delta)):
        binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
        pro = []
        for binary_amap, regions in zip(binary_amaps, mask_regions):
            for region in regions:
                tp_pixels = binary_amap[region.coord_xs, region.coord_ys].sum()
                pro.append(tp_pixels / region.area)
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks_sum
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

@jaxtyped(typechecker=typechecker)
def cal_pro_score_gpu(
    masks: Bool[np.ndarray, "N H W"],
    amaps: Float[np.ndarray, "N H W"],
    max_step: int = 200,
    expect_fpr: float = 0.3,
) -> float:
    assert torch.cuda.is_available()
    dev = torch.device("cuda")

    masks_t = torch.from_numpy(masks).to(dev)
    amaps_t = torch.from_numpy(amaps).to(dev)

    min_th, max_th = amaps_t.min(), amaps_t.max()
    ths = torch.linspace(min_th, max_th, max_step, device=dev)
    inverse_masks_t = ~masks_t
    inverse_masks_sum = inverse_masks_t.sum()

    from skimage import measure
    @dataclass
    class RegionProps:
        area: int
        coord_xs: Int[NDArray[np.int_], "N"]
        coord_ys: Int[NDArray[np.int_], "N"]

        def __init__(self, region):
            self.area = region.area
            coords = region.coords
            self.coord_xs, self.coord_ys = coords.T
    mask_regions = [
        [RegionProps(x) for x in measure.regionprops(measure.label(mask))]
        for mask in masks
    ]

    # 用于存储每个块的计算结果
    pros_chunks = []
    fprs_chunks = []

    total_size = 2 ** 30  # 1GB
    total_size = 4 * total_size  # 4GB
    th_chunk_size = total_size // (masks_t.shape[0] * masks_t.shape[1] * masks_t.shape[2] * 4)  # 每个float32占4字节
    if th_chunk_size == 0:
        th_chunk_size = 1
    print(f"Threshold chunk size: {th_chunk_size}")
    # 核心优化：按块（chunk）循环处理阈值，避免一次性载入全部
    for th_chunk in tqdm(torch.split(ths, th_chunk_size)):
        # th_chunk 是一个小的阈值张量，例如包含 50 个阈值

        # 1. 对当前块进行并行阈值化
        #    中间张量 binary_amaps_t_chunk 的形状为 [N, len(th_chunk), H, W]
        #    其大小远小于之前的 [N, max_step, H, W]，从而节省了显存
        binary_amaps_t_chunk = amaps_t.unsqueeze(1) > th_chunk.view(1, -1, 1, 1)

        # 2. 对当前块并行计算 FPR
        fp_pixels_chunk = torch.logical_and(inverse_masks_t.unsqueeze(1), binary_amaps_t_chunk).sum(dim=(0, 2, 3))
        fprs_chunk = fp_pixels_chunk / inverse_masks_sum
        fprs_chunks.append(fprs_chunk)

        # 3. 对当前块并行计算 PRO
        all_region_pros_chunk = []
        for i, regions in enumerate(mask_regions):
            if not regions:
                continue
            
            binary_amap_i_chunk = binary_amaps_t_chunk[i]
            for region in regions:
                coord_xs_t = torch.from_numpy(region.coord_xs).long().to(dev)
                coord_ys_t = torch.from_numpy(region.coord_ys).long().to(dev)

                tp_pixels_per_th_chunk = binary_amap_i_chunk[:, coord_xs_t, coord_ys_t].sum(dim=1)
                region_pro_chunk = tp_pixels_per_th_chunk / region.area
                all_region_pros_chunk.append(region_pro_chunk.unsqueeze(0))
        
        if not all_region_pros_chunk:
            pros_chunk = torch.zeros_like(th_chunk)
        else:
            pros_chunk = torch.cat(all_region_pros_chunk, dim=0).mean(dim=0)
        
        pros_chunks.append(pros_chunk)

    pros = torch.cat(pros_chunks)
    fprs = torch.cat(fprs_chunks)
    
    # 后处理部分与之前完全相同
    pros, fprs, ths = pros.cpu().numpy(), fprs.cpu().numpy(), ths.cpu().numpy()
    
    idxes = fprs < expect_fpr
    if not np.any(idxes):
        return 0.0

    fprs = fprs[idxes]
    pros = pros[idxes]
    
    if fprs.max() == fprs.min():
        return 0.0
        
    fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
    pro_auc = auc(fprs, pros)
    assert isinstance(pro_auc, float)
    return pro_auc

class MetricsCalculator:
    def __init__(self):
        # self.precision_metric = BinaryPrecision()
        # self.recall_metric = BinaryRecall()
        # self.f1_metric = BinaryF1Score()
        self.auroc_metric = BinaryAUROC()
        self.ap_metric = BinaryAveragePrecision()
        self.pixel_auroc_metric = BinaryAUROC()
        # todo: 改成渐进式
        self.anomaly_maps: list[Float[np.ndarray, "N H W"]] = []
        self.true_masks: list[Bool[np.ndarray, "N H W"]] = []

    @jaxtyped(typechecker=typechecker)
    def update(self, preds: DetectionResult, gts: DetectionGroundTruth):
        pred_score = torch.tensor(preds.pred_scores)
        true_label = torch.tensor(gts.true_labels)
        # self.precision_metric.update(pred_score, true_label)
        # self.recall_metric.update(pred_score, true_label)
        # self.f1_metric.update(pred_score, true_label)
        self.ap_metric.update(pred_score, true_label)
        self.auroc_metric.update(pred_score, true_label)
        self.ap_metric.update(pred_score, true_label)

        pred_score_pixel = torch.tensor(preds.anomaly_maps).flatten()
        true_mask_pixel = torch.tensor(gts.true_masks).flatten()
        self.pixel_auroc_metric.update(pred_score_pixel, true_mask_pixel)
        self.anomaly_maps.append(preds.anomaly_maps)
        self.true_masks.append(gts.true_masks)

    def compute(self) -> DetectionMetrics:
        # precision = self.precision_metric.compute().item()
        # recall = self.recall_metric.compute().item()
        # f1_score = self.f1_metric.compute().item()
        auroc = self.auroc_metric.compute().item()
        ap = self.ap_metric.compute().item()
        pixel_auroc = self.pixel_auroc_metric.compute().item()
        anomaly_maps = np.concatenate(self.anomaly_maps, axis=0)
        true_masks = np.concatenate(self.true_masks, axis=0)
        pixel_aupro = cal_pro_score_gpu(true_masks, anomaly_maps)
        return DetectionMetrics(
            # precision=precision,
            # recall=recall,
            # f1_score=f1_score,
            auroc=auroc,
            ap=ap,
            pixel_auroc=pixel_auroc,
            pixel_aupro=pixel_aupro,
        )


class Detector:
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def __call__(self, image_paths: list[str], class_name: str) -> DetectionResult:
        pass


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
            img_mask = (np.array(img_mask) > 0).astype(np.uint8) * 255 # 将图片中的掩码部分变为255，非掩码部分变为0
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
        meta_file = path / "meta.json"
        with open(meta_file, "r") as f:
            data = json.load(f)

        category_datas = []
        # 只使用test数据进行评估
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

        super().__init__("VisA", category_datas)


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
        table = pd.DataFrame(columns=[x for x in DetectionMetrics.__dataclass_fields__.keys()])

    for data in dataset.category_datas:
        if data.category in existing_categories:
            print(f"Metrics for category {data.category} already exist. Skipping.")
            continue
        print(f"Evaluating category: {data.category}")
        metrics_calculator = MetricsCalculator()

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
