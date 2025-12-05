from pathlib import Path
import cv2
from torch import Tensor, adaptive_avg_pool1d
import torch
import torch.nn.functional as F
import numpy as np
from jaxtyping import Float, Bool, jaxtyped, Shaped
from torchvision.transforms import Compose, CenterCrop
from PIL.Image import Resampling

from align.pca import pca_background_mask
from analysis.similarity import draw_rectangle
from analysis.utils import get_mask_anomaly_indices
from data.cached_impl import RealIADDevidedByAngle
from data.utils import (
    ImageSize,
    Transform,
    denormalize_image,
    from_cv2_image,
    to_cv2_image,
    to_pil_image,
)
from evaluator.dinov2 import DINOv2VisionTransformer
from evaluator.dinov3 import DINOv3VisionTransformer
from evaluator.image_normalize import DINO_NORMALIZE
import evaluator.reproducibility as repro


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
    transform = Transform(
        resize=shortest_side,
        image_transform=Compose([CenterCrop(image_size.hw()), DINO_NORMALIZE]),
        mask_transform=CenterCrop(image_size.hw()),
    )
    origin_transform = Transform(
        resize=shortest_side,
        image_transform=CenterCrop(image_size.hw()),
        mask_transform=CenterCrop(image_size.hw()),
    )
    root_dir = Path("results_analysis/anomaly_patch_similarity_dinov3")
    for category in dataset.get_categories():
        print(f"Processing category: {category}")
        category_dir = root_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        origin_tensor_data = dataset.get_tensor(category, origin_transform)
        tensor_data = dataset.get_tensor(category, transform)
        labels = dataset.get_labels(category)
        # 在异常里随机抽两个
        anomaly_indices = [i for i, label in enumerate(labels) if label]
        random_indices = torch.tensor(anomaly_indices)[
            torch.randperm(len(anomaly_indices))[:2]
        ]
        sample1 = tensor_data[int(random_indices[0])]
        sample2 = tensor_data[int(random_indices[1])]
        img1 = sample1.image
        mask1 = sample1.mask
        img2 = sample2.image
        origin_img1 = origin_tensor_data[int(random_indices[0])].image
        origin_img2 = origin_tensor_data[int(random_indices[1])].image
        features1 = dino(img1.unsqueeze(0)).squeeze(0)  # [P, D]
        features2 = dino(img2.unsqueeze(0)).squeeze(0)  # [P, D]
        features1 = features1 / features1.norm(dim=-1, keepdim=True)
        features2 = features2 / features2.norm(dim=-1, keepdim=True)
        background_mask = pca_background_mask(
            features1, grid_size=grid_size, threshold=0.7
        )
        # 对 异常区域 取 patch
        fg_indices = get_mask_anomaly_indices(mask1, patch_size=patch_size)[:10]
        for idx in fg_indices:
            idx = int(idx)
            patch_feature = features1[idx : idx + 1]  # [1, D]
            # 计算与features2中所有patch的相似度
            similarities: Float[Tensor, "P"] = F.cosine_similarity(
                patch_feature, features2, dim=-1
            )  # [P]
            similarities_2d = similarities.reshape(*grid_size.hw())  # [H, W]
            # 归一化到0-1
            # similarities_2d = (similarities_2d - similarities_2d.min()) / (
            #     similarities_2d.max() - similarities_2d.min() + 1e-8
            # )
            similarities_2d[similarities_2d < 0] = 0
            similarities_image = (
                to_pil_image(similarities_2d)
                .convert("RGB")
                .resize((image_size.w, image_size.h), resample=Resampling.BILINEAR)
            )
            similarities_image = from_cv2_image(
                cv2.applyColorMap(to_cv2_image(similarities_2d), cv2.COLORMAP_JET)
            )
            similarities_image = F.interpolate(
                similarities_image.unsqueeze(0),
                size=image_size.hw(),
                mode="bilinear",
                align_corners=False,
            ).squeeze()
            origin_img1_with_rect = draw_rectangle(
                origin_img1, idx, patch_size, color=(0, 255, 0), thickness=2
            )
            _, sim_topk_indices = torch.topk(similarities_2d.flatten(), k=3)
            colors = [(0, 0, 255), (0, 255, 255), (0, 255, 0)]
            for i, sim_idx in enumerate(sim_topk_indices):
                sim_idx = int(sim_idx)
                similarities_image = draw_rectangle(
                    similarities_image, sim_idx, patch_size, color=colors[i], thickness=2
                )
            # 将原图、目标图像、相似度图横向堆叠保存
            total_image = torch.cat(
                [
                    origin_img1_with_rect,
                    denormalize_image(origin_img2),
                    similarities_image,
                ],
                dim=-1,
            )
            to_pil_image(total_image).save(category_dir / f"patch_sim_{idx}_max{similarities_2d.max().item():.4f}.png")
            # to_pil_image(origin_img1_with_rect).save(category_dir / f"{idx}_img1.png")
            # to_pil_image(origin_img2).save(category_dir / f"{idx}_img2.png")
            # to_pil_image(similarities_image).save(
            #     category_dir / f"{idx}_similarity.png"
            # )
