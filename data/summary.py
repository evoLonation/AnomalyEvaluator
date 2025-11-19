from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .detection_dataset import (
    DetectionDataset,
    MetaDataset,
    MetaSample,
)
from .utils import ImageSize, Transform, to_pil_image


def generate_summary_view(
    dataset: DetectionDataset,
    save_dir: Path = Path("summary_views"),
    max_samples_per_type: int = 5,
    image_size: ImageSize = ImageSize(h=224, w=224),
):
    """
    为数据集的每个类别生成概览图，展示正常和异常样本

    Args:
        dataset: 检测数据集
        save_dir: 保存目录
        max_samples_per_type: 每种类型（正常/异常）最多抽样的图片数量
        image_size: 每张图片resize后的大小
    """

    # 创建保存目录
    save_dir = save_dir / dataset.get_name()
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset.set_transform(Transform())

    for category in dataset.get_categories():
        # 分离正常和异常样本
        normal_indices = [
            i for i, s in enumerate(dataset.get_labels(category)) if not s
        ]
        anomaly_indices = [i for i, s in enumerate(dataset.get_labels(category)) if s]

        # 抽样
        normal_count = min(max_samples_per_type, len(normal_indices))
        anomaly_count = min(max_samples_per_type, len(anomaly_indices))

        tensor_dataset = dataset[category]

        if normal_count > 0:
            normal_indices = np.random.choice(
                len(normal_indices), size=normal_count, replace=False
            )
            selected_normal = [tensor_dataset[i.item()] for i in normal_indices]
        else:
            selected_normal = []

        if anomaly_count > 0:
            anomaly_indices = np.random.choice(
                len(anomaly_indices), size=anomaly_count, replace=False
            )
            selected_anomaly = [tensor_dataset[i.item()] for i in anomaly_indices]
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
            img = sample.image.cpu().numpy()
            img = to_pil_image(img)
            img = img.resize(image_size.pil(), Image.Resampling.LANCZOS)
            axes[row, col].imshow(img)
            axes[row, col].set_title("Normal", fontsize=10, color="green")
            axes[row, col].axis("off")
            idx += 1

        # 再显示异常样本
        for sample in selected_anomaly:
            row, col = idx // cols, idx % cols
            img = sample.image.cpu().numpy()
            img = to_pil_image(img)
            img = img.resize(image_size.pil(), Image.Resampling.LANCZOS)
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
            f"{dataset.get_name()} - {category}\n(Normal: {len(normal_indices)}, Anomaly: {len(anomaly_indices)})",
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
