from dataclasses import dataclass
import os
from pathlib import Path
import shutil
from typing import Literal, cast
from sklearn.metrics import precision_recall_curve, roc_curve
from tqdm import tqdm
import numpy as np
from jaxtyping import Int, Float, Bool
import pandas as pd

from data.detection_dataset import (
    Dataset,
    DetectionDataset,
    DetectionDatasetByFactory,
    MetaInfo,
    MetaSample,
)
from data import MVTecAD, RealIAD, RealIADDevidedByAngle, VisA
from torch.utils.data import Subset

from data.utils import ImageSize


def find_optimal_threshold(
    pred_scores: Float[np.ndarray, "N"],
    true_labels: Bool[np.ndarray, "N"],
    method: Literal["youden", "f1", "precision", "recall"] = "youden",
) -> float:
    """
    找到最优阈值
    Args:
        method:
            - "youden": 最大化 Youden's J statistic (Sensitivity + Specificity - 1)
            - "f1": 最大化 F1-Score
            - "precision": 满足指定精确率的阈值; 该方法能够保证高精确率（如95%），但可能牺牲召回率
            - "recall": 满足指定召回率的阈值; 该方法能够保证高召回率（如95%），但可能牺牲精确率
    """
    if method == "youden":
        # 基于ROC曲线找最优点
        fpr, tpr, thresholds = roc_curve(true_labels, pred_scores)
        j_scores = tpr - fpr  # Youden's J statistic
        optimal_idx = np.argmax(j_scores)
        return thresholds[optimal_idx]

    elif method == "f1":
        # 基于PR曲线找最优F1
        precision, recall, thresholds = precision_recall_curve(true_labels, pred_scores)
        # 注意：thresholds比precision/recall少一个元素
        f1_scores = (
            2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
        )
        optimal_idx = np.argmax(f1_scores)
        return thresholds[optimal_idx]

    elif method == "recall":
        # 找到满足高召回率(如95%)的最低阈值
        precision, recall, thresholds = precision_recall_curve(true_labels, pred_scores)
        target_recall = 0.95
        idx = np.where(recall[:-1] >= target_recall)[0]
        if len(idx) == 0:
            return pred_scores.min()
        return thresholds[idx[-1]].item()

    elif method == "precision":
        # 找到满足高精确率(如95%)的最高阈值
        precision, recall, thresholds = precision_recall_curve(true_labels, pred_scores)
        target_precision = 0.95
        idx = np.where(precision[:-1] >= target_precision)[0]
        if len(idx) == 0:
            return pred_scores.max()
        return thresholds[idx[0]].item()

    raise ValueError(f"Unknown method: {method}")


def find_misclassified_samples(
    pred_scores: Float[np.ndarray, "N"],
    true_labels: Bool[np.ndarray, "N"],
    threshold: float = 0.5,
) -> tuple[Int[np.ndarray, "FN"], Int[np.ndarray, "FP"]]:
    """
    Returns:
        false_negatives: 漏检样本的索引 (真实为异常，预测为正常)
        false_positives: 误检样本的索引 (真实为正常，预测为异常)
    """
    pred_labels = pred_scores >= threshold
    # 漏检：真实异常但预测正常
    false_negatives = np.where((true_labels == 1) & (pred_labels == 0))[0]
    # 误检：真实正常但预测异常
    false_positives = np.where((true_labels == 0) & (pred_labels == 1))[0]
    return false_negatives, false_positives


@dataclass
class MisclassifiedSamplesAnalysisResult:
    threshold: float
    fn_indices: Int[np.ndarray, "M"]  # 漏检样本索引
    fp_indices: Int[np.ndarray, "K"]  # 误检样本索引
    fn_scores: Float[np.ndarray, "M"]  # 漏检样本的异常分数
    fp_scores: Float[np.ndarray, "K"]  # 误检样本的异常分数


def analyze_errors(
    pred_scores: Float[np.ndarray, "N"],
    true_labels: Bool[np.ndarray, "N"],
    method: Literal["youden", "f1", "precision", "recall"] = "youden",
) -> MisclassifiedSamplesAnalysisResult:

    threshold = find_optimal_threshold(pred_scores, true_labels, method)
    fn_indices, fp_indices = find_misclassified_samples(
        pred_scores, true_labels, threshold
    )

    result = MisclassifiedSamplesAnalysisResult(
        threshold=threshold,
        fn_indices=fn_indices,
        fp_indices=fp_indices,
        fn_scores=pred_scores[fn_indices],
        fp_scores=pred_scores[fp_indices],
    )

    return result


def read_scores_csv(
    scores_csv: Path,
) -> tuple[list[int], list[str], list[float]]:
    """读取评分CSV文件"""
    df = pd.read_csv(scores_csv, header=None)
    csv_indices = df.iloc[:, 0].tolist()
    csv_image_paths = df.iloc[:, 1].tolist()
    csv_scores = df.iloc[:, 2].tolist()
    return csv_indices, csv_image_paths, csv_scores


def analyze_errors_by_csv(
    scores_csv: Path,
    dataset: Dataset[MetaSample],
) -> MisclassifiedSamplesAnalysisResult:

    labels = [sample.label for sample in dataset]
    csv_indices, csv_image_paths, csv_scores = read_scores_csv(scores_csv)
    # simple check
    assert dataset[csv_indices[0]].image_path == csv_image_paths[0]

    scores = [s for _, s in sorted(zip(csv_indices, csv_scores), key=lambda x: x[0])]
    result = analyze_errors(
        pred_scores=np.array(scores),
        true_labels=np.array(labels),
    )
    return result


def copy_images(save_dir: Path, image_paths: list[Path]):
    save_dir.mkdir(parents=True, exist_ok=False)
    copied_paths = []
    for img_path in image_paths:
        img_name = img_path.name
        dest_path = save_dir / img_name
        shutil.copy(img_path, dest_path)
        copied_paths.append(str(dest_path))
    return copied_paths


def get_all_error_images(
    scores_csv: Path, dataset: Dataset[MetaSample], save_dir: Path
):
    result = analyze_errors_by_csv(scores_csv, dataset)
    image_paths = [sample.image_path for sample in dataset]
    mask_paths = [sample.mask_path for sample in dataset]

    fn_image_paths = [image_paths[i] for i in result.fn_indices.tolist()]
    fn_mask_paths = [mask_paths[i] for i in result.fn_indices.tolist()]
    fp_image_paths = [image_paths[i] for i in result.fp_indices.tolist()]
    copy_images(
        save_dir / "false_negatives",
        [Path(p) for p in fn_image_paths] + [Path(cast(str, p)) for p in fn_mask_paths],
    )
    copy_images(save_dir / "false_positives", [Path(p) for p in fp_image_paths])


def get_error_dataset(
    dataset: DetectionDataset,
    scores_csvs: list[Path],
    categories: list[str],
) -> DetectionDataset:
    import pandas as pd

    meta_info = MetaInfo(
        data_dir=dataset.get_data_dir(),
        category_datas={},
    )
    category_indices = {}
    for scores_csv, category in zip(scores_csvs, categories):
        meta_dataset = dataset.get_meta(category)
        result = analyze_errors_by_csv(scores_csv, meta_dataset)
        indices = result.fn_indices.tolist() + result.fp_indices.tolist()
        meta_info.category_datas[category] = [meta_dataset[i] for i in indices]
        category_indices[category] = indices
    tensor_factory = lambda c, t: Dataset.bypt(
        Subset(dataset.get_tensor(c, t), category_indices[c])
    )
    return DetectionDatasetByFactory(
        dataset.get_name() + "(error)", meta_info, tensor_factory
    )


if __name__ == "__main__":

    def handle_realiad(category: str, angle: int):
        get_all_error_images(
            scores_csv=Path(
                f"detection_evaluation/AnomalyCLIP(mvtec)_RealIAD_angle_scores/{category}_C{angle}.csv"
            ),
            dataset=RealIADDevidedByAngle().get_meta(f"{category}_C{angle}"),
            save_dir=Path(f"results/error_images/RealIAD/{category}_C{angle}"),
        )
