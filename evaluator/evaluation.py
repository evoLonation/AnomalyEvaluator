from pathlib import Path
from typing import Callable, Iterable, cast
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
from torch.utils.data import Dataset

from .reproducibility import get_reproducible_dataloader
from .detector import DetectionGroundTruth, Detector, TensorDetector
from .metrics import MetricsCalculator, DetectionMetrics
from .data import (
    CachedDataset,
    CategoryMetaDataset,
    CategoryTensorDataset,
    MetaSample,
    TensorSample,
    TensorSampleBatch,
    generate_empty_mask,
    generate_mask,
)

type MixedSample = tuple[MetaSample, TensorSample]
type MixedSampleBatch = tuple[list[MetaSample], TensorSampleBatch]


class MixedCategoryDataset(Dataset[MixedSample]):
    def __init__(
        self, tensor_dset: CategoryTensorDataset, meta_dset: CategoryMetaDataset
    ):
        self.tensor_dset = tensor_dset
        self.meta_dset = meta_dset
        assert len(tensor_dset) == len(
            meta_dset
        ), "Tensor and Meta datasets must have the same length."

    def __len__(self) -> int:
        return len(self.tensor_dset)

    def __getitem__(self, index: int) -> MixedSample:
        meta_sample = self.meta_dset[index]
        tensor_sample = self.tensor_dset[index]
        return meta_sample, tensor_sample

    def collate_fn(self, batch: list[MixedSample]) -> MixedSampleBatch:
        meta_samples = [x[0] for x in batch]
        tensor_samples = [x[1] for x in batch]
        tensor_batch = CategoryTensorDataset.collate_fn(tensor_samples)
        return meta_samples, tensor_batch


def evaluation_detection(
    path: Path,
    detector: Detector | TensorDetector,
    dataset: CachedDataset,
    category: str | list[str] | None = None,
    save_anomaly_score: bool = False,
    save_anomaly_map: bool = False,
    batch_size: int = 1,  # only used if not BatchJointDetector
    namer: Callable[
        [Detector | TensorDetector, CachedDataset], str
    ] = lambda det, dset: f"{det.name}_{dset.name}",
):

    print(
        f"Evaluating detector {detector.name} on dataset "
        f"{dataset.name} {'' if category is None else f'for category {category} '}..."
    )

    dset = (
        dataset.get_meta_dataset()
        if isinstance(detector, Detector)
        else dataset.get_tensor_dataset(detector.image_size)
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

    category_metrics: list[tuple[str, DetectionMetrics]] = []
    if metrics_output_path.exists():
        print(f"Metrics for {detector.name} on {dataset.name} already exist.")
        table = pd.read_csv(metrics_output_path, index_col=0)
        existing_categories = set(table.index)
        if len(existing_categories) > 0:
            print(f"Existing categories: {existing_categories}")
    else:
        existing_categories = set()
        table = pd.DataFrame(
            columns=[x for x in DetectionMetrics.__dataclass_fields__.keys()]
        )

    # 如果提供了 category，则只评估对应类别
    all_categories = set(x for x in dset.category_datas.keys())
    if category is not None:
        category_set = {category} if isinstance(category, str) else set(category)
        assert category_set.issubset(
            all_categories
        ), f"Some specified categories do not exist in the dataset: {category_set - all_categories}"
    else:
        category_set = all_categories

    for category, datas in dset.category_datas.items():
        if category not in category_set:
            continue
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
            metrics_calculator = MetricsCalculator(type(detector))
        else:
            metrics_calculator = None

        # 为分数保存准备容器
        category_scores: list[tuple[str, float]] = []

        if isinstance(datas, CategoryTensorDataset) and scores_needed and maps_needed:
            datas = MixedCategoryDataset(
                datas, dataset.get_meta_dataset().category_datas[category]
            )
        dataloader = get_reproducible_dataloader(
            datas,
            batch_size=batch_size,
            num_workers=4,
            shuffle=False,
            collate_fn=(
                None if isinstance(datas, CategoryMetaDataset) else datas.collate_fn
            ),
        )
        dataloader = cast(
            Iterable[list[MetaSample]]
            | Iterable[TensorSampleBatch]
            | Iterable[MixedSampleBatch],
            dataloader,
        )
        for i, batch in enumerate(tqdm(dataloader, desc=f"Processing {category}")):
            if isinstance(batch, list):
                batch_image_paths = [x.image_path for x in batch]
                batch_correct_labels = np.array([x.label for x in batch])
                results = cast(Detector, detector)(batch_image_paths, category)
                image_size = results.anomaly_maps.shape[1:]
                batch_correct_masks = []
                for x in batch:
                    if x.mask_path:
                        mask = generate_mask(Path(x.mask_path), image_size=image_size)
                    else:
                        mask = generate_empty_mask(image_size)
                    batch_correct_masks.append(mask)
                batch_correct_masks = np.array(batch_correct_masks)
            elif isinstance(batch, TensorSampleBatch):
                batch_image_paths = None
                results = cast(TensorDetector, detector)(batch.images, category)
                batch_correct_labels = batch.labels.numpy()
                batch_correct_masks = batch.masks.numpy()
            else:
                meta_samples, tensor_batch = batch
                batch_image_paths = [x.image_path for x in meta_samples]
                results = cast(TensorDetector, detector)(tensor_batch.images, category)
                batch_correct_labels = tensor_batch.labels.numpy()
                batch_correct_masks = tensor_batch.masks.numpy()
            # 仅在需要指标时才计算 GT 和更新指标
            if metrics_needed:
                ground_truth = DetectionGroundTruth(
                    true_labels=batch_correct_labels,
                    true_masks=batch_correct_masks,
                )
                metrics_calculator = cast(MetricsCalculator, metrics_calculator)
                metrics_calculator.update(results, ground_truth)

            # 保存分数
            if scores_needed and "batch_image_paths" in locals():
                for j, img_path in enumerate(cast(list[str], batch_image_paths)):
                    score = float(results.pred_scores[j])
                    category_scores.append((img_path, score))

            # 保存掩码图（叠加到原图的可视化样式）
            if maps_needed:
                # 目录：maps_output_dir / <relative to dataset.path> ，文件统一保存为 .png
                for j, img_path in enumerate(cast(list[str], batch_image_paths)):
                    rel_path = (
                        Path(img_path).resolve().relative_to(dataset.data_dir.resolve())
                    )
                    save_path = (maps_output_dir / rel_path).with_suffix(".png")
                    save_path.parent.mkdir(parents=True, exist_ok=True)

                    # 读取并缩放原图到 anomaly_map 尺寸
                    amap = results.anomaly_maps[j].astype(np.float32)
                    H, W = amap.shape[:2]
                    img_bgr = cv2.imread(img_path)
                    if img_bgr is None:
                        # 若读取失败则跳过该图（保持最小化处理，不引入额外异常）
                        continue
                    img_bgr = cv2.resize(img_bgr, (W, H))
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                    # 归一化到 [0, 255] 并着色
                    min_v = float(amap.min())
                    max_v = float(amap.max())
                    if max_v > min_v:
                        mask8 = ((amap - min_v) / (max_v - min_v) * 255.0).astype(
                            np.uint8
                        )
                    else:
                        mask8 = np.zeros_like(amap, dtype=np.uint8)
                    heatmap = cv2.applyColorMap(mask8, cv2.COLORMAP_JET)
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

                    # 融合
                    alpha = 0.5
                    blended_rgb = (
                        alpha * img_rgb.astype(np.float32)
                        + (1.0 - alpha) * heatmap.astype(np.float32)
                    ).astype(np.uint8)
                    blended_bgr = cv2.cvtColor(blended_rgb, cv2.COLOR_RGB2BGR)

                    cv2.imwrite(str(save_path), blended_bgr)

        # 写出该类别的各类结果（根据需要）
        if metrics_needed:
            metrics_calculator = cast(MetricsCalculator, metrics_calculator)
            metrics = metrics_calculator.compute()
            category_metrics.append((category, metrics))
            table.loc[category] = [getattr(metrics, col) for col in table.columns]
            table.to_csv(metrics_output_path)  # 每计算完一个类别就保存一次
            print(f"Category {category} metrics saved: {metrics}")

        if scores_needed:
            scores_output_dir.mkdir(parents=True, exist_ok=True)
            with open(scores_file_path, "w") as f:
                for img_path, score in category_scores:
                    f.write(f"{img_path},{score}\n")
            print(f"Category {category} scores saved: {scores_file_path}")

        if maps_needed:
            maps_output_dir.mkdir(parents=True, exist_ok=True)
            # 以空文件标记类别完成
            maps_done_flag.touch()
            print(f"Category {category} maps saved. Done flag: {maps_done_flag}")

    # 计算平均指标
    if len(table) == len(dset.category_datas) and "Average" not in table.index:
        table.loc["Average"] = [table[col].mean() for col in table.columns]
        table.to_csv(metrics_output_path)
        print(f"Average metrics saved: {table.loc['Average']}")
    print(
        f"Evaluation of {detector.name} on {dataset.name} {'' if category is None else f'for category {category} '}completed."
    )
