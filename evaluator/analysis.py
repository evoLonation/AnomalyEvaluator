from dataclasses import dataclass
import os
from pathlib import Path
import shutil
from typing import Literal, cast
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve
from tqdm import tqdm
import numpy as np
from jaxtyping import Int, Float, Bool
import pandas as pd
from scipy.stats import rankdata

from data.base import ListDataset
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


# 定义返回结果的类型结构
@dataclass
class DiffResult:
    indices: Int[np.ndarray, "N"]  # 样本在原数组中的索引
    diff_values: Float[np.ndarray, "N"]  # 排名差异值（用于衡量差异的大小）
    scores1: Float[np.ndarray, "N"]  # 方法1的原分数
    scores2: Float[np.ndarray, "N"]  # 方法2的原分数


@dataclass
class AnalysisReport:
    # 方法1 表现更好的情况
    m1_better_on_anomaly: DiffResult  # 漏检挖掘 (M1检出, M2漏检)
    m1_better_on_normal: DiffResult  # 误报抑制 (M1正确, M2误报)

    # 方法2 表现更好的情况 (M1 退化)
    m2_better_on_anomaly: DiffResult  # M1漏检, M2检出
    m2_better_on_normal: DiffResult  # M1误报, M2正确


def compare_analyze(
    pred_scores1: Float[np.ndarray, "N"],
    pred_scores2: Float[np.ndarray, "N"],
    true_labels: Bool[np.ndarray, "N"],
    top_k: int = 10,
) -> AnalysisReport:
    """
    基于排名差异筛选关键样本。

    Args:
        pred_scores1: 方法1（通常是更好的那个）的异常分数
        pred_scores2: 方法2（通常是基准）的异常分数
        true_labels: 真实标签 (True/1 为异常, False/0 为正常)
        top_k: 每种情况筛选差异最大的前 K 个样本
    """
    # 将 nan 替换为0
    indices_nan = np.isnan(pred_scores1) | np.isnan(pred_scores2)
    pred_scores1 = np.nan_to_num(pred_scores1, nan=0.0)
    pred_scores2 = np.nan_to_num(pred_scores2, nan=0.0)

    # 1. 计算排名百分比 (0.0 ~ 1.0)
    # rankdata 返回 1 到 N，除以 N 归一化
    n = len(true_labels)
    rank1 = rankdata(pred_scores1) / n
    rank2 = rankdata(pred_scores2) / n

    # 2. 计算排名差异 (Rank1 - Rank2)
    # 如果 diff > 0: 方法1 认为它更异常
    # 如果 diff < 0: 方法2 认为它更异常
    rank_diff = rank1 - rank2

    def get_top_k(mask: np.ndarray, sort_descending: bool) -> DiffResult:
        """辅助函数：提取 Mask 内差异最大的 TopK"""
        valid_indices = np.where(mask)[0]
        valid_indices = valid_indices[~indices_nan[valid_indices]]
        if len(valid_indices) == 0:
            return DiffResult(
                indices=np.array([], dtype=int),
                diff_values=np.array([], dtype=float),
                scores1=np.array([]),
                scores2=np.array([]),
            )

        # 获取这些样本的差异值
        diffs = rank_diff[valid_indices]

        # 排序
        if sort_descending:
            # 找差异最大的正数 (M1 排名远高于 M2)
            sorted_idx_local = np.argsort(diffs)[::-1]
        else:
            # 找差异最小的负数 (M1 排名远低于 M2)
            sorted_idx_local = np.argsort(diffs)

        top_indices = valid_indices[sorted_idx_local[:top_k]]

        return DiffResult(
            indices=top_indices,
            diff_values=rank_diff[top_indices],
            scores1=pred_scores1[top_indices],
            scores2=pred_scores2[top_indices],
        )

    # Case 1: M1 更好 - 在异常样本上，M1 排名更高 (Hard Defect Discovery)
    # Label=True, Rank1 > Rank2
    mask_m1_anomaly = true_labels & (rank_diff > 0)
    # Case 2: M1 更好 - 在正常样本上，M1 排名更低 (False Positive Suppression)
    # Label=False, Rank1 < Rank2
    mask_m1_normal = (~true_labels) & (rank_diff < 0)
    # Case 3: M2 更好 - 在异常样本上，M2 排名更高 (M1 Regression)
    # Label=True, Rank1 < Rank2
    mask_m2_anomaly = true_labels & (rank_diff < 0)
    # Case 4: M2 更好 - 在正常样本上，M2 排名更低
    # Label=False, Rank1 > Rank2
    mask_m2_normal = (~true_labels) & (rank_diff > 0)

    return AnalysisReport(
        m1_better_on_anomaly=get_top_k(mask_m1_anomaly, sort_descending=True),
        m1_better_on_normal=get_top_k(mask_m1_normal, sort_descending=False),
        m2_better_on_anomaly=get_top_k(mask_m2_anomaly, sort_descending=False),
        m2_better_on_normal=get_top_k(mask_m2_normal, sort_descending=True),
    )


def compare_analyze_to_plot(
    pred_scores1: Float[np.ndarray, "N"],
    pred_scores2: Float[np.ndarray, "N"],
    true_labels: Bool[np.ndarray, "N"],
    save_path: Path,
    method1_name: str = "Ours (Method 1)",
    method2_name: str = "Baseline (Method 2)",
):
    """
    绘制对比散点图。
    X轴: Method 2 Scores (Normalized)
    Y轴: Method 1 Scores (Normalized)
    """
    # 将 nan 替换为0
    pred_scores1 = np.nan_to_num(pred_scores1, nan=0.0)
    pred_scores2 = np.nan_to_num(pred_scores2, nan=0.0)

    # 1. Min-Max 归一化到 [0, 1] 以便绘图
    def normalize(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    s1_norm = normalize(pred_scores1)
    s2_norm = normalize(pred_scores2)

    # 2. 准备绘图
    plt.figure(figsize=(10, 10), dpi=100)

    # 分离正常和异常样本
    normal_idx = ~true_labels.astype(bool)
    anomaly_idx = true_labels.astype(bool)

    # 绘制对角线 (y=x)
    plt.plot(
        [0, 1],
        [0, 1],
        color="gray",
        linestyle="--",
        alpha=0.5,
        label="Equal Performance",
    )

    # 绘制散点
    # 正常样本 (蓝色)
    plt.scatter(
        s2_norm[normal_idx],
        s1_norm[normal_idx],
        c="dodgerblue",
        alpha=0.4,
        s=20,
        label=f"Normal ({np.sum(normal_idx)})",
    )
    # 异常样本 (红色)
    plt.scatter(
        s2_norm[anomaly_idx],
        s1_norm[anomaly_idx],
        c="crimson",
        alpha=0.6,
        s=20,
        label=f"Anomaly ({np.sum(anomaly_idx)})",
    )

    # 3. 添加区域注解 (帮助理解)
    # 左上角: Method 1 High, Method 2 Low -> M1 Finds, M2 Misses
    plt.text(
        0.1,
        0.9,
        f"{method1_name}\nDetects More",
        fontsize=12,
        color="darkred",
        ha="left",
        va="top",
        fontweight="bold",
    )

    # 右下角: Method 1 Low, Method 2 High -> M1 Suppresses FPs
    plt.text(
        0.9,
        0.1,
        f"{method1_name}\nSuppresses Noise",
        fontsize=12,
        color="darkblue",
        ha="right",
        va="bottom",
        fontweight="bold",
    )

    # 4. 设置标签和标题
    plt.xlabel(f"{method2_name} Normalized Score", fontsize=12)
    plt.ylabel(f"{method1_name} Normalized Score", fontsize=12)
    plt.title(
        f"Score Distribution Comparison\n{method1_name} vs {method2_name}", fontsize=14
    )
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)

    # 5. 保存
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Comparison plot saved to {save_path}")


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
    assert dataset[csv_indices[0]].image_path == csv_image_paths[0], (dataset[csv_indices[0]].image_path, csv_image_paths[0])

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
        meta_info.category_datas[category] = ListDataset(
            [meta_dataset[i] for i in indices]
        )
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
