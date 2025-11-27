from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sized, cast
from matplotlib import category
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import cv2
import h5py
from torch.utils.data import Sampler
from jaxtyping import Int, Float, Bool

from .reproducibility import get_reproducible_dataloader
from .detector import DetectionGroundTruth, Detector, TensorDetector
from .metrics import BaseMetricsCalculator, MetricsCalculator, DetectionMetrics
from data.detection_dataset import (
    Dataset,
    DetectionDataset,
    MetaDataset,
    MetaSample,
    TensorSample,
    TensorSampleBatch,
)
from data.base import DatasetWithIndex, ZipedDataset, tuple_collate_fn
from data.utils import ImageSize, generate_mask, generate_empty_mask


def save_anomaly_maps_h5(
    indices: Int[torch.Tensor, "N"],
    anomaly_maps: Float[torch.Tensor, "N H W"],
    output_path: Path,
):
    """保存异常图到 H5 文件"""
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as h5f:
        h5f.create_dataset("indices", data=indices.cpu().numpy())
        h5f.create_dataset("anomaly_maps", data=anomaly_maps.cpu().numpy())


def load_anomaly_maps_h5(
    input_path: Path,
) -> tuple[Int[torch.Tensor, "N"], Float[torch.Tensor, "N H W"]]:
    """从 H5 文件加载异常图"""
    with h5py.File(input_path, "r") as h5f:
        indices_np = h5f["indices"][:]  # type: ignore
        anomaly_maps_np = h5f["anomaly_maps"][:]  # type: ignore
        indices = torch.tensor(indices_np, dtype=torch.long)
        anomaly_maps = torch.tensor(anomaly_maps_np, dtype=torch.float32)
    return indices, anomaly_maps


def visualize_anomaly_map_on_image(
    dataset: ZipedDataset[MetaSample, TensorSample],
    anomaly_maps: Float[torch.Tensor, "N H W"],
    output_dir: Path,
    data_dir: Path,
):
    """可视化异常图到原图上，对所有掩码图进行全局 min-max 归一化"""
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # 全局 min-max 归一化
    min_v = float(anomaly_maps.min())
    max_v = float(anomaly_maps.max())

    for idx, (meta_sample, tensor_sample) in tqdm(
        enumerate(cast(Iterable[tuple[MetaSample, TensorSample]], dataset)),
        desc="Visualizing anomaly maps",
    ):
        # 保存路径
        rel_path = (
            Path(meta_sample.image_path).resolve().relative_to(data_dir.resolve())
        )
        save_path = (output_dir / rel_path).with_suffix(".png")

        visualizer_image_map(
            tensor_sample.image, anomaly_maps[idx], save_path, min_v, max_v
        )


def visualizer_image_map(
    image: Float[torch.Tensor, "C H W"],
    anomaly_map: Float[torch.Tensor, "H W"],
    output_path: Path,
    min_v: float,
    max_v: float,
):
    """可视化单张图片的异常图"""
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    amap = anomaly_map.to(torch.float32)
    H, W = amap.shape[:2]

    # 使用图像 tensor 数据（已经过变换处理）
    img_np = image.cpu().numpy()  # [C, H', W']
    img_np = np.transpose(img_np, (1, 2, 0))  # [H', W', C]
    # 将图像数据转换为 uint8 范围
    img_np = (img_np * 255.0).clip(0, 255).astype(np.uint8)  # 假设归一化到 [0, 1]

    # 缩放到 anomaly_map 尺寸
    img_rgb = cv2.resize(img_np, (W, H))

    # 使用全局 min-max 归一化到 [0, 255]
    if max_v > min_v:
        mask8 = ((amap - min_v) / (max_v - min_v) * 255.0).to(torch.uint8)
    else:
        mask8 = torch.zeros_like(amap, dtype=torch.uint8)
    heatmap = cv2.applyColorMap(mask8.cpu().numpy(), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # 融合
    alpha = 0.5
    blended_rgb = (
        alpha * img_rgb.astype(np.float32) + (1.0 - alpha) * heatmap.astype(np.float32)
    ).astype(np.uint8)
    blended_bgr = cv2.cvtColor(blended_rgb, cv2.COLOR_RGB2BGR)

    cv2.imwrite(str(output_path), blended_bgr)


def get_metrics_average(metrics: dict[str, DetectionMetrics]) -> DetectionMetrics:
    avg_metrics = {}
    for datafield in DetectionMetrics.__dataclass_fields__.keys():
        avg_metrics[datafield] = np.mean(
            [getattr(m, datafield) for m in metrics.values()]
        ).item()
    avg_metrics = DetectionMetrics(**avg_metrics)
    return avg_metrics


def save_metrics_to_csv(metrics: dict[str, DetectionMetrics], output_path: Path):
    table = pd.DataFrame(
        columns=[x for x in DetectionMetrics.__dataclass_fields__.keys()]
    )
    for category, metric in metrics.items():
        table.loc[category] = [getattr(metric, col) for col in table.columns]
    table.loc["Average"] = [
        getattr(get_metrics_average(metrics), col) for col in table.columns
    ]
    formatted_to_csv(table, output_path)


def load_metrics_from_csv(input_path: Path) -> dict[str, DetectionMetrics]:
    table = formatted_read_csv(input_path)
    loaded_metrics: dict[str, DetectionMetrics] = {}
    for category in table.index:
        if category == "Average":
            continue
        row = table.loc[category]
        metric_values = {col: float(row[col]) for col in table.columns}
        loaded_metrics[category] = DetectionMetrics(**metric_values)
    return loaded_metrics


def save_multi_metrics_to_csv(
    all_metrics: list[dict[str, DetectionMetrics]], output_path: Path
):
    columns = [
        [f"{x}", f"{x}_std"] for x in DetectionMetrics.__dataclass_fields__.keys()
    ]
    columns = [item for sublist in columns for item in sublist]
    combined_table = pd.DataFrame(columns=columns)
    # 取所有指标字典中的类别的交集
    categories = [set(m.keys()) for m in all_metrics]
    common_categories = set.intersection(*categories)
    common_categories = sorted(common_categories)
    for metrics in all_metrics:
        metrics["Average"] = get_metrics_average(metrics)
    common_categories.append("Average")
    for category in common_categories:
        row_values = []
        for col in DetectionMetrics.__dataclass_fields__.keys():
            vals = [metrics[category].__dict__[col] for metrics in all_metrics]
            mean_val = np.mean(vals).item()
            std_val = np.std(vals).item()
            row_values.extend([mean_val, std_val])
        combined_table.loc[category] = row_values
    formatted_to_csv(combined_table, output_path)


def formatted_to_csv(df: pd.DataFrame, output_path: Path):
    df = df * 100
    # 所有浮点数转换为 xx.xx 的字符串形式
    for col in df.columns:
        df[col] = df[col].apply(lambda x: f"{x:05.2f}")
    # 对齐索引
    indexes: list[str] = df.index.tolist()
    longest_index_length = max(len(idx) for idx in indexes)
    aligned_indexes: list[str] = [
        idx + " " * (longest_index_length - len(idx)) for idx in indexes
    ]
    df.index = pd.Index(aligned_indexes)
    df.to_csv(output_path)


def formatted_read_csv(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path, index_col=0)
    df.index = df.index.str.rstrip()
    df = df.astype(float)  # 转换为浮点数
    # 如果所有数据大于1, 则除以100
    if (df.values >= 1).any():
        df = df / 100.0
    return df


# todo
@dataclass
class ResultSample:
    label: bool
    score: float
    image_path: str
    image: Float[torch.Tensor, "C H W"]
    mask: Bool[torch.Tensor, "H W"]
    anomaly_mask: Float[torch.Tensor, "H W"]


class ResultDataset(Dataset[ResultSample]):
    def __init__(self, samples: list[ResultSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> ResultSample:
        return self.samples[index]


def get_category_scores(
    score_dir: Path, map_dir: Path, dataset: DetectionDataset, category: str
) -> ResultDataset: ...


def evaluation_detection(
    path: Path,
    detector: Detector | TensorDetector,
    dataset: DetectionDataset,
    category: str | list[str] | None = None,
    save_anomaly_score: bool = False,
    save_anomaly_map: bool = False,
    visualize_anomaly_map: bool = False,
    batch_size: int = 1,
    sampler_getter: Callable[[str, Sized], Sampler] | None = None,
    namer: Callable[
        [Detector | TensorDetector, DetectionDataset], str
    ] = lambda det, dset: f"{det.name}_{dset.get_name()}",
    cpu_metrics: bool = False,
):
    """
    csv 异常分数保存格式：
    index(数据集原始索引),image_path,anomaly_score
    5,000001.png,0.1234
    3,000002.png,0.5678
    ...
    掩码图保存格式：
    H5 文件，包含两个数据集：
    - indices: shape (N,)，对应数据集中的样本索引
    - anomaly_maps: shape (N, H, W)，对应的异常图数据
    可视化掩码图仅当TensorDetector时支持，使用预处理后图像与异常图进行融合显示，保存为 PNG 格式，路径结构与数据集相同。
    """

    print(
        f"Evaluating detector {detector.name} on dataset "
        f"{dataset.get_name()} {'' if category is None else f'for category {category} '}..."
    )

    # total_samples = sum(len(datas) for datas in dataset.category_datas)
    # print(
    #     f"Dataset has {total_samples} samples across {len(dataset.category_datas)} categories."
    # )

    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    metrics_output_path = path / f"{namer(detector, dataset)}_metrics.csv"
    scores_output_dir = path / f"{namer(detector, dataset)}_scores"
    maps_output_dir = path / f"{namer(detector, dataset)}_maps"

    if metrics_output_path.exists():
        print(f"Metrics for {detector.name} on {dataset.get_name()} already exist.")
        category_metrics: dict[str, DetectionMetrics] = load_metrics_from_csv(
            metrics_output_path
        )
        existing_categories = set(category_metrics.keys())
        if len(existing_categories) > 0:
            print(f"Existing categories: {existing_categories}")
    else:
        existing_categories = set()
        category_metrics = {}

    # 如果提供了 category，则只评估对应类别
    all_categories = dataset.get_categories()
    if category is not None:
        categories = [category] if isinstance(category, str) else category
        assert set(categories).issubset(
            all_categories
        ), f"Some specified categories do not exist in the dataset: {set(categories) - set(all_categories)}"
    else:
        categories = list(all_categories)

    for category in categories:
        # 三类输出：metrics（按行写入 metrics_output_path）、scores（每类一个 csv 文件）、maps（按图片路径保存，类完成后生成 done 文件）
        metrics_needed = category not in existing_categories
        scores_file_path = scores_output_dir / f"{category}.csv"
        maps_done_flag = maps_output_dir / f"{category}_done"
        scores_needed = save_anomaly_score and not scores_file_path.exists()
        maps_needed = save_anomaly_map and not maps_done_flag.exists()

        if not (metrics_needed or scores_needed or maps_needed):
            print(f"Category {category} already has all requested outputs. Skipping.")
            continue

        print(f"Evaluating category: {category}")
        if metrics_needed:
            metrics_calculator = BaseMetricsCalculator(cpu=cpu_metrics)
        else:
            metrics_calculator = None

        # 为分数和掩码保存准备容器
        category_scores: list[tuple[int, str, float]] = []
        category_indices: list[int] = []
        category_anomaly_maps: list[Float[torch.Tensor, "N H W"]] = []

        if isinstance(detector, Detector):
            datas = dataset.get_meta(category)
            collate_fn = lambda b: b
            datas = DatasetWithIndex(datas)
        else:
            datas = dataset.get_tensor(category, detector.transform)
            if scores_needed or maps_needed:
                datas = ZipedDataset(dataset.get_meta(category), datas)
                collate_fn = lambda b: tuple_collate_fn(
                    b, t2_collate_fn=TensorSample.collate_fn
                )
                datas = DatasetWithIndex(datas)
            else:
                collate_fn = TensorSample.collate_fn
                datas = DatasetWithIndex(datas)

        indexed_collate_fn = lambda b: tuple_collate_fn(b, t2_collate_fn=collate_fn)

        dataloader = get_reproducible_dataloader(
            datas,
            batch_size=batch_size,
            num_workers=4,
            shuffle=False,
            collate_fn=indexed_collate_fn,
            sampler=sampler_getter(category, datas) if sampler_getter else None,
        )
        dataloader = cast(Iterable[tuple[list[int], Any]], dataloader)
        for i, batch in enumerate(tqdm(dataloader, desc=f"Processing {category}")):
            batch_indices, batch = batch
            batch = cast(
                TensorSampleBatch
                | list[MetaSample]
                | tuple[list[MetaSample], TensorSampleBatch],
                batch,
            )
            if isinstance(batch, list):
                batch_image_paths = [x.image_path for x in batch]
                batch_correct_labels = torch.tensor(
                    [x.label for x in batch], dtype=torch.bool
                )
                assert isinstance(detector, Detector)
                results = detector(batch_image_paths, category)
                batch_correct_masks = []
                for x in batch:
                    if x.mask_path:
                        mask = generate_mask(Path(x.mask_path))
                        if detector.mask_transform:
                            mask = detector.mask_transform(mask)
                    else:
                        image_size = ImageSize.fromtensor(results.anomaly_maps[0])
                        mask = generate_empty_mask(image_size)
                    batch_correct_masks.append(mask)
                batch_correct_masks = torch.stack(batch_correct_masks)
            elif isinstance(batch, TensorSampleBatch):
                batch_image_paths = None
                results = cast(TensorDetector, detector)(batch.images, category)
                batch_correct_labels = batch.labels
                batch_correct_masks = batch.masks
            else:
                meta_samples, tensor_batch = batch
                batch_image_paths = [x.image_path for x in meta_samples]
                results = cast(TensorDetector, detector)(tensor_batch.images, category)
                batch_correct_labels = tensor_batch.labels
                batch_correct_masks = tensor_batch.masks
            # 仅在需要指标时才计算 GT 和更新指标
            if metrics_needed:
                ground_truth = DetectionGroundTruth(
                    true_labels=batch_correct_labels,
                    true_masks=batch_correct_masks,
                )
                metrics_calculator = cast(MetricsCalculator, metrics_calculator)
                metrics_calculator.update(results, ground_truth)

            # 保存分数
            if scores_needed:
                assert batch_image_paths is not None
                for j, img_path in enumerate(batch_image_paths):
                    score = float(results.pred_scores[j])
                    category_scores.append((batch_indices[j], img_path, score))

            # 收集掩码图数据
            if maps_needed:
                category_indices.extend(batch_indices)
                category_anomaly_maps.append(results.anomaly_maps)

        # 写出该类别的各类结果（根据需要）
        if metrics_needed:
            metrics_calculator = cast(MetricsCalculator, metrics_calculator)
            metrics = metrics_calculator.compute()
            category_metrics[category] = metrics
            save_metrics_to_csv(category_metrics, metrics_output_path)
            print(f"Category {category} metrics saved: {metrics}")

        if scores_needed:
            scores_output_dir.mkdir(parents=True, exist_ok=True)
            with open(scores_file_path, "w") as f:
                for idx, img_path, score in category_scores:
                    f.write(f"{idx},{img_path},{score:.4f}\n")
            print(f"Category {category} scores saved: {scores_file_path}")

        if maps_needed:
            maps_output_dir.mkdir(parents=True, exist_ok=True)
            # 保存原始异常图到 h5
            h5_save_path = maps_output_dir / f"{category}.h5"
            indices_tensor = torch.tensor(category_indices, dtype=torch.long)
            anomaly_maps_tensor = torch.cat(category_anomaly_maps)
            save_anomaly_maps_h5(indices_tensor, anomaly_maps_tensor, h5_save_path)
            print(f"Category {category} anomaly maps saved to H5: {h5_save_path}")

            # 可视化所有掩码图（仅 TensorDetector 支持）
            if isinstance(detector, TensorDetector) and visualize_anomaly_map:
                # 获取原始的 MixedCategoryDataset（去掉 DatasetWithIndex 包装）
                visualize_anomaly_map_on_image(
                    cast(ZipedDataset[MetaSample, TensorSample], datas.base_dataset),
                    anomaly_maps_tensor,
                    maps_output_dir,
                    dataset.get_data_dir(),
                )

            # 标记完成
            maps_done_flag.touch()
            print(
                f"Category {category} maps visualization completed. Done flag: {maps_done_flag}"
            )

    print(f"Average metrics : {get_metrics_average(category_metrics)}")
    print(
        f"Evaluation of {detector.name} on {dataset.get_name()} "
        f"{'' if category is None else f'for category {category} '}completed."
    )


if __name__ == "__main__":
    save_multi_metrics_to_csv(
        [
            load_metrics_from_csv(
                Path("results/musc_11_19/MuSc2_RealIAD(angle)_metrics.csv")
            ),
            load_metrics_from_csv(
                Path("results/musc_11_19/MuSc2_RealIAD(angle)_seed43_metrics.csv")
            ),
        ],
        Path("results/musc_11_19/MuSc2_RealIAD(angle)_metrics_avg.csv"),
    )
