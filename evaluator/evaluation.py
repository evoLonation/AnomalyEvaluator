from pathlib import Path
from typing import cast
import numpy as np
import pandas as pd
from tqdm import tqdm
from detector import BatchJointDetector, DetectionGroundTruth, Detector
from metrics import MetricsCalculator, DetectionMetrics
import cv2
from data import BatchJointDataset, DetectionDataset, generate_masks


def evaluation_detection(
    path: Path,
    detector: Detector | BatchJointDetector,
    dataset: DetectionDataset | BatchJointDataset,
    category: str | list[str] | None = None,
    save_anomaly_score: bool = False,
    save_anomaly_map: bool = False,
    batch_size: int = 1,  # only used if not BatchJointDetector
):
    assert isinstance(dataset, DetectionDataset) == isinstance(
        detector, Detector
    ), "Dataset and detector types do not match."

    print(
        f"Evaluating {'batch joint ' if isinstance(detector, BatchJointDetector) else ''}"
        f"detector {detector.name} on dataset {dataset.name} {'' if category is None else f'for category {category} '}..."
    )

    total_samples = sum(len(datas) for datas in dataset.category_datas)
    print(
        f"Dataset has {total_samples} samples across {len(dataset.category_datas)} categories."
    )

    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    metrics_output_path = path / f"{detector.name}_{dataset.name}_metrics.csv"
    scores_output_dir = path / f"{detector.name}_{dataset.name}_scores"
    maps_output_dir = path / f"{detector.name}_{dataset.name}_maps"

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
    if category is not None:
        category_set = {category} if isinstance(category, str) else set(category)
        assert category_set.issubset(
            set(x for x in dataset.category_datas.keys())
        ), f"Some specified categories do not exist in the dataset: {category_set - set(x for x in dataset.category_datas.keys())}"
    else:
        category_set = set(x for x in dataset.category_datas.keys())

    for category, datas in dataset.category_datas.items():
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

        if isinstance(dataset, BatchJointDataset):
            batch_size = dataset.batch_size
            if batch_size == -1:
                batch_size = len(datas)

        for i in tqdm(
            range(0, len(datas), batch_size),
            desc=f"Processing {category}",
        ):
            batch_image_paths = [x.image_path for x in datas[i : i + batch_size]]
            batch_correct_labels = [x.label for x in datas[i : i + batch_size]]
            results = detector(batch_image_paths, category)
            # 仅在需要指标时才计算 GT 和更新指标
            if metrics_needed:
                ground_truth = DetectionGroundTruth(
                    true_labels=np.array(batch_correct_labels, dtype=bool),
                    true_masks=generate_masks(
                        dataset.category_datas[category][i : i + batch_size],
                        image_shape=results.anomaly_maps.shape[1:],
                    ),
                )
                metrics_calculator = cast(MetricsCalculator, metrics_calculator)
                metrics_calculator.update(results, ground_truth)

            # 保存分数
            if scores_needed:
                for j, img_path in enumerate(batch_image_paths):
                    score = float(results.pred_scores[j])
                    category_scores.append((img_path, score))

            # 保存掩码图（叠加到原图的可视化样式）
            if maps_needed:
                # 目录：maps_output_dir / <relative to dataset.path> ，文件统一保存为 .png
                for j, img_path in enumerate(batch_image_paths):
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
    if len(table) == len(dataset.category_datas) and "Average" not in table.index:
        table.loc["Average"] = [table[col].mean() for col in table.columns]
        table.to_csv(metrics_output_path)
        print(f"Average metrics saved: {table.loc['Average']}")
    print(
        f"Evaluation of {detector.name} on {dataset.name} {'' if category is None else f'for category {category} '}completed."
    )
