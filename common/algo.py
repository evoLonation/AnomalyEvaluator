from jaxtyping import Float, jaxtyped, Int, Bool
import torch
import torch.nn.functional as F
from torch import Tensor

from data.utils import ImageSize


@jaxtyped(typechecker=None)
def shift_image(
    images: Float[Tensor, "B 3 H W"],
    dx: int,
    dy: int,
) -> Float[Tensor, "B 3 H W"]:
    """
    对图像进行平移，使用反射填充处理边界
    dx: 水平方向平移（正值为右移）
    dy: 垂直方向平移（正值为下移）
    """
    # 使用 torch.nn.functional.pad 进行反射填充
    # pad 格式: (left, right, top, bottom)
    pad_left = abs(dx)
    pad_right = abs(dx)
    pad_top = abs(dy)
    pad_bottom = abs(dy)
    # 反射填充
    padded = F.pad(images, (pad_left, pad_right, pad_top, pad_bottom), mode="reflect")
    h, w = images.shape[2:4]

    shifted = padded[
        :, :, pad_top + dy : pad_top + h + dy, pad_left + dx : pad_left + w + dx
    ]
    return shifted


@jaxtyped(typechecker=None)
def aggregate_shifted_features(
    features: Float[Tensor, "X 4 P D"],
    grid_size: tuple[int, int],
) -> Float[Tensor, "X P D"]:
    X, _, D = features.shape
    PH, PW = grid_size
    feats: Float[Tensor, "X 4 PH PW D"] = features.view(X, 4, PH, PW, D)
    origin_features: Float[Tensor, "X PH PW D"] = feats[:, 0, ...].clone()
    left_features: Float[Tensor, "X PH-2 PW-2 D"] = feats[:, 1, 1:-1, 1:-1, :]
    right_features: Float[Tensor, "X PH-2 PW-2 D"] = feats[:, 1, 1:-1, 2:, :]
    up_features: Float[Tensor, "X PH-2 PW-2 D"] = feats[:, 2, 1:-1, 1:-1, :]
    down_features: Float[Tensor, "X PH-2 PW-2 D"] = feats[:, 2, 2:, 1:-1, :]
    leftup_features: Float[Tensor, "X PH-2 PW-2 D"] = feats[:, 3, 1:-1, 1:-1, :]
    leftdown_features: Float[Tensor, "X PH-2 PW-2 D"] = feats[:, 3, 2:, 1:-1, :]
    rightup_features: Float[Tensor, "X PH-2 PW-2 D"] = feats[:, 3, 1:-1, 2:, :]
    rightdown_features: Float[Tensor, "X PH-2 PW-2 D"] = feats[:, 3, 2:, 2:, :]
    origin_features[:, 1:-1, 1:-1, :] += (
        (left_features + right_features) / 2
        + (up_features + down_features) / 2
        + (leftup_features + leftdown_features + rightup_features + rightdown_features)
        / 4
    )
    origin_features[:, 1:-1, 1:-1, :] /= 4
    return origin_features.view(X, -1, D)


def get_avg_pool_features(
    features: Float[Tensor, "B P D"],
    grid_size: tuple[int, int],
    r: int,
) -> Float[Tensor, "B P D"]:
    if r > 1:
        assert r % 2 == 1, "r should be odd."
        PH, PW = grid_size
        B, P, D = features.shape
        feats: Float[Tensor, "B D P"] = features.permute(0, 2, 1)
        feats: Float[Tensor, "B D PH PW"] = feats.reshape(*feats.shape[0:2], PH, PW)
        padding = r // 2
        feats: Float[Tensor, f"B D*{r*r} P"] = F.unfold(
            feats, kernel_size=(r, r), padding=padding, stride=1, dilation=1
        )
        feats: Float[Tensor, f"B P D*{r*r}"] = feats.permute(0, 2, 1)
        feats: Float[Tensor, f"B*P D*{r*r}"] = feats.reshape(-1, feats.shape[-1])
        pool_batch_size = 2048
        # pool_batch_size = features.shape[0]
        pooled_features_list = []
        for i in range(0, feats.shape[0], pool_batch_size):
            batch_features = feats[i : i + pool_batch_size]
            pooled_batch_features: Float[Tensor, "_ D"] = torch.adaptive_avg_pool1d(
                batch_features, D
            )
            pooled_features_list.append(pooled_batch_features)
        feats: Float[Tensor, "B*P D"] = torch.cat(pooled_features_list, dim=0)
        feats: Float[Tensor, "B P D"] = feats.reshape(B, P, D)
        return feats
    return features


# 对于每个 patch，计算其对应的 patch 与周围 4 个 patch 对应 patch 的平均距离
@jaxtyped(typechecker=None)
def compute_patch_offset_distance(
    match_pindices: Int[Tensor, "X P"],
    grid_size: tuple[int, int],
) -> Float[Tensor, "X P"]:
    _, P = match_pindices.shape
    PH, PW = grid_size

    match_indices: Int[Tensor, "X PH PW"] = match_pindices.view(-1, PH, PW)
    match_coords: Int[Tensor, "X PH PW 2"] = torch.stack(
        [match_indices // PW, match_indices % PW],
        dim=-1,
    )
    pad_coords: Int[Tensor, "X PH+2 PW+2 2"] = F.pad(
        match_coords.permute(0, 3, 1, 2),
        (1, 1, 1, 1),
        mode="replicate",
    ).permute(0, 2, 3, 1)
    neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    distance_list = []
    for dy, dx in neighbor_offsets:
        neighbor_coords = pad_coords[:, dy + 1 : dy + 1 + PH, dx + 1 : dx + 1 + PW, :]
        distance: Float[Tensor, "X PH PW"] = torch.norm(
            match_coords.float() - neighbor_coords.float(), dim=-1, p=2
        )
        distance_list.append(distance)
    distances: Float[Tensor, "X PH PW"] = torch.stack(distance_list, dim=-1).mean(
        dim=-1
    )
    distances: Float[Tensor, "X P"] = distances.view(distances.shape[0], -1)
    return distances


@jaxtyped(typechecker=None)
def pca_background_mask(
    features: Float[Tensor, "N P D"],
    grid_size: tuple[int, int],
    threshold: float = 0.5,
) -> Bool[Tensor, "N P"]:
    """
    使用PCA进行背景检测，返回背景掩码
    Args:
        features: [N, P, D] 特征矩阵
    Returns:
        background_mask: [N, P] 布尔型背景掩码, True表示前景
    """
    # PCA降至1维
    features_centered = features - features.mean(dim=1, keepdim=True)
    U, S, V = torch.pca_lowrank(features_centered, q=1, niter=10)
    pca_features: Float[Tensor, "N P 1"] = torch.matmul(
        features_centered, V
    )  # [N, P, 1]
    # MinMax归一化
    norm_features: Float[Tensor, "N P 1"] = (pca_features - pca_features.min()) / (
        pca_features.max() - pca_features.min()
    )
    # threshold background
    background_mask: Bool[Tensor, "N P"] = norm_features[:, :, 0] > threshold  # [N, P]

    # 如果边缘有 0.9 以上的为 True，则反转
    background_mask_hw = background_mask.view(-1, *grid_size)  # [N, H, W]
    edge_mask = torch.cat(
        [
            background_mask_hw[:, 0, :],
            background_mask_hw[:, -1, :],
            background_mask_hw[:, :, 0],
            background_mask_hw[:, :, -1],
        ],
        dim=1,
    )  # [N, H*2 + W*2]
    edge_mask_means = edge_mask.float().mean(dim=1)  # [N]
    for i in range(background_mask.shape[0]):
        if edge_mask_means[i] > 0.6:
            background_mask[i] = ~background_mask[i]
            print("  Inverted background mask based on edge analysis")

    return background_mask
