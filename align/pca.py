from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn.functional as F
import numpy as np
from jaxtyping import Float, Bool, jaxtyped
from torchvision.transforms import Compose, CenterCrop
from sklearn.decomposition import PCA
from PIL.Image import Resampling

from data.cached_impl import RealIADDevidedByAngle
from data.utils import ImageSize, Transform, to_pil_image
from evaluator.dinov2 import DINOv2VisionTransformer
from evaluator.dinov3 import DINOv3VisionTransformer
from evaluator.image_normalize import DINO_NORMALIZE
import evaluator.reproducibility as repro


def pca_background_mask(
    features: Float[torch.Tensor, "P D"],
    grid_size: ImageSize,
    threshold: float = 0.5,
) -> Bool[torch.Tensor, "P"]:
    """
    使用PCA进行背景检测，返回背景掩码
    Args:
        features: [P, D] 特征矩阵
    Returns:
        background_mask: [P] 布尔型背景掩码, True表示前景
    """
    # PCA降至1维
    features_centered = features - features.mean(dim=0, keepdim=True)
    U, S, V = torch.pca_lowrank(features_centered, q=1, niter=10)
    pca_features = torch.matmul(features_centered, V)  # [P, 1]

    # MinMax归一化
    norm_features = (pca_features - pca_features.min()) / (
        pca_features.max() - pca_features.min()
    )

    # threshold background
    background_mask = norm_features[:, 0] > threshold  # [P]

    # 如果边缘有 0.9 以上的为 True，则反转
    background_mask_hw = background_mask.view(*grid_size.hw())  # [H, W]
    edge_mask = torch.cat(
        [
            background_mask_hw[0, :],
            background_mask_hw[-1, :],
            background_mask_hw[:, 0],
            background_mask_hw[:, -1],
        ]
    )
    if edge_mask.float().mean() > 0.9:
        background_mask = ~background_mask
        print("  Inverted background mask based on edge analysis")

    return background_mask


def pca_visualize(
    features: Float[torch.Tensor, "P D"],
) -> Float[torch.Tensor, "P 3"]:
    """
    使用PCA进行前景特征可视化，返回3维特征
    Args:
        features: [P, D] 特征矩阵
    Returns:
        foreground_vis: [P, 3] 3维前景特征可视化
    """
    # PCA降至3维
    features_centered = features - features.mean(dim=0, keepdim=True)
    U, S, V = torch.pca_lowrank(features_centered, q=3)
    features_foreground = torch.matmul(features_centered, V)  # [P, 3]

    # MinMax归一化
    norm_features_foreground = (features_foreground - features_foreground.min()) / (
        features_foreground.max() - features_foreground.min()
    )

    return norm_features_foreground


if __name__ == "__main__":
    repro.init(42)
    dataset = RealIADDevidedByAngle()

    if True:
        dino = DINOv3VisionTransformer()
        shortest_side = 518
        image_size = ImageSize.square(512)
    else:
        dino = DINOv2VisionTransformer()
        shortest_side = 518
        image_size = ImageSize.square(518)

    grid_size = ImageSize(
        h=image_size.h // dino.get_patch_size(),
        w=image_size.w // dino.get_patch_size(),
    )
    patch_size = dino.get_patch_size()
    root_dir = Path("results_analysis/pac_background_masks_dinov3_torch2")
    for category in dataset.get_categories():
        print(f"Processing category: {category}")
        category_dir = root_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        transform = Transform(
            resize=shortest_side,
            image_transform=Compose([CenterCrop(image_size.hw()), DINO_NORMALIZE]),
        )
        origin_tensor_data = dataset.get_tensor(
            category,
            Transform(
                resize=shortest_side, image_transform=CenterCrop(image_size.hw())
            ),
        )
        tensor_data = dataset.get_tensor(category, transform)
        # 随机抽两个
        random_indices = torch.randperm(len(tensor_data))[:2]
        for idx in random_indices:
            idx = int(idx)
            image = tensor_data[idx].image
            origin_image = origin_tensor_data[idx].image
            features = dino(image.unsqueeze(0)).squeeze(0)  # [P, D]

            # 第一阶段：背景检测 - GPU版本
            background_mask = pca_background_mask(
                features, grid_size=grid_size, threshold=0.7
            )

            # 第二阶段：背景抑制
            fg_features = features.clone()
            fg_features[~background_mask] = 0

            # 第三阶段：前景特征可视化 - GPU版本
            foreground_vis = pca_visualize(fg_features)

            background_mask = background_mask.reshape(*grid_size.hw())
            foreground_vis = foreground_vis.reshape(*grid_size.hw(), 3)

            foreground_vis_upsampled = foreground_vis.permute(2, 0, 1).unsqueeze(0)
            foreground_vis_upsampled = F.interpolate(
                foreground_vis_upsampled,
                size=image_size.hw(),
                mode="bilinear",
                align_corners=False,
            ).squeeze()

            # 保存图像
            # 1. 原图
            origin_pil = to_pil_image(origin_image)
            origin_pil.save(category_dir / f"{idx}_origin.png")

            # 2. 背景掩码（二值图）
            background_pil = to_pil_image(background_mask).resize(
                image_size.pil(), resample=Resampling.NEAREST
            )
            background_pil.save(category_dir / f"{idx}_background_mask.png")

            # 3. 前景特征可视化（RGB）
            foreground_pil = to_pil_image(foreground_vis_upsampled).resize(
                image_size.pil(), resample=Resampling.BILINEAR
            )
            foreground_pil.save(category_dir / f"{idx}_foreground_vis.png")

            print(f"  Saved images for index {idx}")
