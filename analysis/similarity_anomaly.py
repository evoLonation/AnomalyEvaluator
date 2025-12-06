from pathlib import Path
from typing import Callable, Literal
import cv2
from torch import Generator, Tensor, adaptive_avg_pool1d
import torch
import torch.nn.functional as F
import numpy as np
from jaxtyping import Float, Bool, jaxtyped, Shaped
from torchvision.transforms import Compose, CenterCrop
from PIL.Image import Resampling, Image

from align.pca import pca_background_mask
from analysis.similarity import draw_rectangle
from analysis.utils import get_mask_anomaly_indices
from data.cached_impl import MVTecAD, RealIADDevidedByAngle
from data.detection_dataset import DetectionDataset
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
from evaluator.train2 import get_trained_model


def visualize_patch_similarity_single(
    origin_img: Float[Tensor, "3 H W"],
    target_img: Float[Tensor, "3 H W"],
    patch_feat: Float[Tensor, "D"],
    features: Float[Tensor, "P D"],
    patch_idx: int,
    patch_size: int,
) -> tuple[Image, float]:
    """可视化某个patch与目标图像所有patch的相似度"""
    H, W = origin_img.shape[-2:]
    grid_h = H // patch_size
    grid_w = W // patch_size

    # 计算与features中所有patch的相似度
    patch_feature = patch_feat.unsqueeze(0)  # [1, D]
    similarities: Float[Tensor, "P"] = F.cosine_similarity(
        patch_feature, features, dim=-1
    )  # [P]
    max_sim = float(similarities.max().item())
    similarities_2d = similarities.reshape(grid_h, grid_w)  # [H, W]

    # 归一化到0-1
    # similarities_2d = (similarities_2d - similarities_2d.min()) / (
    #     similarities_2d.max() - similarities_2d.min() + 1e-8
    # )
    similarities_2d[similarities_2d < 0] = 0

    # 将相似度图转换为热力图
    similarities_image = from_cv2_image(
        cv2.applyColorMap(to_cv2_image(similarities_2d), cv2.COLORMAP_JET)
    )
    similarities_image = F.interpolate(
        similarities_image.unsqueeze(0),
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    ).squeeze()

    # 在原图上标注选中的patch
    origin_img_with_rect = draw_rectangle(
        origin_img, patch_idx, patch_size, color=(0, 255, 0), thickness=2
    )

    # 在相似度图上标注top-k最相似的patch
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
            origin_img_with_rect,
            denormalize_image(target_img),
            similarities_image,
        ],
        dim=-1,
    )
    return to_pil_image(total_image), max_sim


def visualize_patch_similarity(
    name: str,
    dataset: DetectionDataset,
    transform: Transform,
    origin_transform: Transform,
    feature_extractor: Callable[[Float[Tensor, "N C H W"]], Float[Tensor, "N P D"]],
    patch_size: int,
    mode: Literal["anomaly", "random"] = "anomaly",
):
    """可视化某个patch与目标图像所有patch的相似度"""
    root_dir = Path(f"results_analysis/anomaly_patch_similarity/{name}/{dataset.get_name()}")
    root_dir.mkdir(parents=True, exist_ok=True)

    for category in dataset.get_categories():
        print(f"Processing category: {category}")
        category_dir = root_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        tensor_data = dataset.get_tensor(category, transform)
        origin_tensor_data = dataset.get_tensor(category, origin_transform)
        labels = dataset.get_labels(category)
        # 在异常里随机抽两个
        anomaly_indices = [i for i, label in enumerate(labels) if label]
        random_indices = torch.tensor(anomaly_indices)[
            torch.randperm(
                len(anomaly_indices),
                generator=Generator().manual_seed(repro.get_global_seed()),
            )[:2]
        ]
        sample1 = tensor_data[int(random_indices[0])]
        sample2 = tensor_data[int(random_indices[1])]
        img1 = sample1.image
        mask1 = sample1.mask
        img2 = sample2.image
        origin_img1 = origin_tensor_data[int(random_indices[0])].image
        origin_img2 = origin_tensor_data[int(random_indices[1])].image
        with torch.no_grad():
            features1 = feature_extractor(img1.unsqueeze(0)).squeeze(0)  # [P, D]
            features2 = feature_extractor(img2.unsqueeze(0)).squeeze(0)  # [P, D]
        features1 = features1 / features1.norm(dim=-1, keepdim=True)
        features2 = features2 / features2.norm(dim=-1, keepdim=True)
        background_mask = pca_background_mask(
            features1,
            grid_size=ImageSize(
                h=img1.shape[1] // patch_size, w=img1.shape[2] // patch_size
            ),
            threshold=0.7,
        )
        if mode == "anomaly":
            # 对 异常区域 取 patch
            fg_indices = get_mask_anomaly_indices(mask1, patch_size=patch_size)[:10]
        elif mode == "random":
            # 在背景区域(background_mask为True)中均匀选取10个patch
            bg_indices = torch.nonzero(
                background_mask.flatten(), as_tuple=False
            ).squeeze(
                -1
            )  # [N]
            if len(bg_indices) == 0:
                print(f"  Warning: No background patches found, skipping...")
                continue
            # 均匀采样
            num_samples = min(10, len(bg_indices))
            step = len(bg_indices) / num_samples
            fg_indices = bg_indices[[int(i * step) for i in range(num_samples)]]
        else:
            assert False
        for idx in fg_indices:
            idx = int(idx)
            patch_feature = features1[idx]  # [D]
            # 计算相似度并保存可视化结果
            total_image, max_sim = visualize_patch_similarity_single(
                origin_img=origin_img1,
                target_img=origin_img2,
                patch_feat=patch_feature,
                features=features2,
                patch_idx=idx,
                patch_size=patch_size,
            )
            save_path = category_dir / f"patch_sim_{idx}_max{max_sim:.4f}.png"
            total_image.save(save_path)


if __name__ == "__main__":
    repro.init(42)
    # dataset = MVTecAD()
    dataset = RealIADDevidedByAngle()
    # shortest_side = 512
    shortest_side = 518
    image_size = ImageSize.square(512)
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

    epoch = 10
    # name = "dinov3"
    name = f"test2_e{epoch}"
    model = get_trained_model(name="test2", epoch=epoch)
    # model = DINOv3VisionTransformer()

    patch_size = model.vision.get_patch_size()
    # patch_size = model.get_patch_size()
    grid_size = ImageSize(
        h=image_size.h // patch_size,
        w=image_size.w // patch_size,
    )
    visualize_patch_similarity(
        name=name,
        dataset=dataset,
        transform=transform,
        origin_transform=origin_transform,
        feature_extractor=model.forward,
        patch_size=patch_size,
    )
