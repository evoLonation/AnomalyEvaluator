from pathlib import Path
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from sklearn.decomposition import PCA
from torchvision.transforms import Compose, CenterCrop, Normalize
from jaxtyping import Float, Int, Bool

from common.plot import show_images
from data.cached_impl import RealIADDevidedByAngle
from data.detection_dataset import TensorSample, TensorSampleBatch
from data.rotate import RandomRotatedDetectionDataset
from data.utils import ImageSize, Transform, from_cv2_image, to_cv2_image
from evaluator.dinov2 import DINOv2VisionTransformer
from evaluator.reproducibility import get_reproducible_dataloader
import evaluator.reproducibility as repro


def compute_background_mask(
    img_features: Float[torch.Tensor, "P D"],
    grid_size: tuple[int, int],
    threshold: float = 10.0,
    kernel_size: int = 3,
    border: float = 0.2,
) -> np.ndarray:
    """
    参考 AnomalyDINO/src/backbones.py 的 compute_background_mask 实现
    使用 PCA 第一主成分区分前景和背景

    Args:
        img_features: [P, D] DINOv2 特征
        grid_size: (H, W) patch网格大小
        threshold: PCA阈值
        kernel_size: 形态学操作的核大小
        border: 中心裁剪比例，用于自适应判断前景/背景

    Returns:
        mask: [P] bool数组，True表示前景patch
    """
    if isinstance(img_features, torch.Tensor):
        img_features = img_features.cpu().numpy()

    # 使用 PCA 提取第一主成分
    pca = PCA(n_components=1, svd_solver="randomized")
    first_pc = pca.fit_transform(img_features.astype(np.float32))

    # 初始假设：第一主成分大于阈值的是前景
    mask = first_pc > threshold

    # 自适应判断前景/背景：检查中心区域的保留比例
    m = mask.reshape(grid_size)[
        int(grid_size[0] * border) : int(grid_size[0] * (1 - border)),
        int(grid_size[1] * border) : int(grid_size[1] * (1 - border)),
    ]

    # 如果中心区域保留的patch太少，说明前景/背景判断反了
    if m.sum() <= m.size * 0.35:
        mask = -first_pc > threshold

    # 形态学操作：填充小孔洞，略微扩大前景区域
    mask_2d = mask.reshape(grid_size).astype(np.uint8)
    mask_2d = cv2.dilate(mask_2d, np.ones((kernel_size, kernel_size), np.uint8))
    mask_2d = cv2.morphologyEx(
        mask_2d, cv2.MORPH_CLOSE, np.ones((kernel_size, kernel_size), np.uint8)
    )

    return mask_2d.flatten().astype(bool)


def match_patches_robust(
    feat1: Float[torch.Tensor, "P D"],
    feat2: Float[torch.Tensor, "P D"],
    image_size: ImageSize,
    patch_size: int = 14,
    mask1: Bool[torch.Tensor, "P"] | None = None,
    mask2: Bool[torch.Tensor, "P"] | None = None,
    topk: int = 5,
) -> tuple[np.ndarray, float]:
    """
    输入:
        feat1, feat2: [P, D] DINOv2 特征
        image_size: 原始图片尺寸
        mask1, mask2: [P] 前景掩码，True表示前景patch
    输出:
        M: 变换矩阵
    """
    # 应用掩码，只考虑前景patch
    if mask1 is not None:
        feat1 = feat1[mask1]
        origin_indices1 = torch.where(mask1)[0]
    else:
        origin_indices1 = torch.arange(len(feat1))

    if mask2 is not None:
        feat2 = feat2[mask2]
        origin_indices2 = torch.where(mask2)[0]
    else:
        origin_indices2 = torch.arange(len(feat2))

    # 1. 归一化并计算相似度矩阵 [N, N]
    feat1 = F.normalize(feat1, dim=1)
    feat2 = F.normalize(feat2, dim=1)
    sim_matrix = torch.mm(feat1, feat2.t())  # 行是 feat1，列是 feat2

    # 2. Top-K 双向匹配
    # A -> B 的 Top-K 匹配
    val_1to2, idx_1to2 = torch.topk(sim_matrix, k=min(topk, sim_matrix.size(1)), dim=1)
    # B -> A 的 Top-K 匹配
    val_2to1, idx_2to1 = torch.topk(
        sim_matrix.t(), k=min(topk, sim_matrix.size(0)), dim=1
    )

    # 找双向 Top-K 匹配：只要双方的 topk 中都有对方就算匹配
    match_indices_list = []
    for i1 in range(len(feat1)):
        # feat1[i1] 的 Top-K 候选
        candidates_from_1 = idx_1to2[i1]  # [K]
        vals_from_1 = val_1to2[i1]  # [K]

        for k_idx, i2 in enumerate(candidates_from_1):
            i2 = int(i2.item())
            # 检查 feat2[i2] 的 Top-K 中是否包含 i1
            candidates_from_2 = idx_2to1[i2]  # [K]

            if i1 in candidates_from_2:
                # 双向都在 Top-K 中，额外加相似度阈值过滤
                if vals_from_1[k_idx] > 0.5:
                    # 映射回原始索引
                    orig_i1 = int(origin_indices1[i1].item())
                    orig_i2 = int(origin_indices2[i2].item())
                    match_indices_list.append((orig_i1, orig_i2))

    if len(match_indices_list) < 5:
        raise ValueError(
            f"双向匹配点过少，无法计算可靠的变换矩阵: {len(match_indices_list)} 点"
        )

    match_indices: Int[torch.Tensor, "N 2"] = torch.tensor(
        match_indices_list, dtype=torch.int64
    )

    # 3. 坐标映射 (Patch Index -> Pixel Coordinate)
    patch_size = 14
    assert image_size.h % patch_size == 0 and image_size.w % patch_size == 0
    ph, pw = image_size.h // patch_size, image_size.w // patch_size

    def get_coords(indices):
        y = (indices // pw) * patch_size + patch_size / 2
        x = (indices % pw) * patch_size + patch_size / 2
        coords = np.stack([x, y], axis=1)
        return coords.astype(int)

    points_1: Int[np.ndarray, "N 2"] = get_coords(match_indices[:, 0])
    points_2: Int[np.ndarray, "N 2"] = get_coords(match_indices[:, 1])

    # 调试信息：检查匹配点的分布
    coord_diff = np.abs(points_1 - points_2)
    print(
        f"匹配点坐标差异: mean={coord_diff.mean(axis=0)}, max={coord_diff.max(axis=0)}"
    )

    # 4. RANSAC 解算
    M, inliers = cv2.estimateAffinePartial2D(
        points_2, points_1, method=cv2.RANSAC, ransacReprojThreshold=10.0
    )
    # inliers 是一个掩码，1 表示被 RANSAC 采纳的点，0 表示被剔除的错误点
    valid_count = int(np.sum(inliers)) if inliers is not None else 0
    print(f"双向匹配点数: {len(match_indices)}, RANSAC 采纳点数: {valid_count}")

    # 对 points_2 旋转 180 度后再试一次（处理旋转图像的情况）
    points_2_rotated = np.array(image_size.wh()) - points_2  # 关于中心旋转 180 度
    M2, inliers2 = cv2.estimateAffinePartial2D(
        points_2_rotated, points_1, method=cv2.RANSAC, ransacReprojThreshold=10.0
    )
    valid_count2 = int(np.sum(inliers2)) if inliers2 is not None else 0
    print(f"尝试旋转180度后，RANSAC 采纳点数: {valid_count2}")
    if M2 is not None and valid_count2 > valid_count:
        M = M2
        valid_count = valid_count2

    if M is None or valid_count < 5:
        raise ValueError(f"RANSAC 失败，无法计算可靠的变换矩阵: {valid_count} 点")

    return M, valid_count / len(match_indices)


if __name__ == "__main__":
    seed = 42
    repro.init(seed)
    dataset = RealIADDevidedByAngle()
    dataset = RandomRotatedDetectionDataset(dataset, seed=seed)
    dino = DINOv2VisionTransformer()
    category = dataset.get_categories()[0]

    image_size = ImageSize(h=518, w=518)
    batch_size = 16
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    image_transform = Compose(
        [
            CenterCrop(image_size.hw()),
            Normalize(mean=mean, std=std),
        ]
    )
    transform = Transform(resize=image_size.h, image_transform=image_transform)
    tensor_dataset = dataset.get_tensor(category, transform)
    tensor_dataset_origin = dataset.get_tensor(category, Transform(resize=image_size.h))
    dataloader = get_reproducible_dataloader(
        dataset=tensor_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        collate_fn=TensorSample.collate_fn,
    )
    save_dir = Path(f"results/match_patch/{category}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # 计算 grid_size
    grid_h = image_size.h // dino.patch_size
    grid_w = image_size.w // dino.patch_size
    grid_size = (grid_h, grid_w)

    for batch_idx, batch in enumerate(dataloader):
        batch: TensorSampleBatch
        images = batch.images
        features: Float[torch.Tensor, "B P D"] = dino(batch.images)
        images_list = [
            tensor_dataset_origin[i].image
            for i in range(batch_idx * batch_size, (batch_idx + 1) * batch_size)
        ]
        show_images(
            images_list,
            save_path=save_dir / f"{batch_idx:03d}_origin.png",
            column_n=8,
        )

        # 为每张图片计算背景掩码
        masks = [
            compute_background_mask(
                features[i],
                grid_size=grid_size,
                threshold=10.0,
            )
            for i in range(len(features))
        ]
        print(f"Batch {batch_idx}: 前景patch比例 = {[m.sum() / len(m) for m in masks]}")

        title_list = ["Img 0 (reference)"]
        for i in range(1, len(features)):
            try:
                M, score = match_patches_robust(
                    features[0],
                    features[i],
                    image_size=image_size,
                    patch_size=dino.patch_size,
                    mask1=torch.from_numpy(masks[0]),
                    mask2=torch.from_numpy(masks[i]),
                )
                # 检查是否接近单位矩阵
                identity_check = np.array([[1, 0, 0], [0, 1, 0]])
                diff = np.abs(M - identity_check).max()
                # print(f"图片 {i}: 变换矩阵与单位矩阵的最大差异 = {diff:.4f}")
                if diff < 0.01:
                    print(f"  ⚠️  变换接近单位矩阵，图片可能已对齐")
                print(f"计算得到变换矩阵:\n{M}")
                warped_img = cv2.warpAffine(
                    to_cv2_image(images_list[i]),
                    M,
                    dsize=image_size.hw(),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                images_list[i] = from_cv2_image(warped_img)
                title_list.append(f"Img {i} (score: {score:.2f})")
            except ValueError as e:
                print(f"匹配失败: {e}")
        show_images(
            images_list,
            titles=title_list,
            save_path=save_dir / f"{batch_idx:03d}_aligned.png",
            column_n=8,
        )
