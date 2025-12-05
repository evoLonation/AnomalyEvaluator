import cv2
from torch import Tensor
import torch
from jaxtyping import Bool, Shaped, jaxtyped, Float
import torch.nn.functional as F

from data.utils import ImageSize, from_cv2_image, to_cv2_image


@jaxtyped(typechecker=None)
def aggregate_features(
    features_: Float[Tensor, "X P D"],
    r: int,
    grid_size: ImageSize,
) -> Float[Tensor, "X P D"]:
    if r > 1:
        assert r % 2 == 1, "r should be odd."
        embed_dim = features_.shape[-1]
        features: Float[Tensor, "X D PH PW"] = features_.reshape(
            *features_.shape[0:2], *grid_size.hw()
        )
        padding = r // 2
        features: Float[Tensor, f"X D*{r*r} P"] = F.unfold(
            features, kernel_size=(r, r), padding=padding, stride=1, dilation=1
        )
        features: Float[Tensor, f"X P D*{r*r}"] = features.permute(0, 2, 1)
        features: Float[Tensor, f"X*P D*{r*r}"] = features.reshape(
            -1, features.shape[-1]
        )
        pool_batch_size = 2048
        # pool_batch_size = features.shape[0]
        pooled_features_list = []
        for i in range(0, features.shape[0], pool_batch_size):
            batch_features = features[i : i + pool_batch_size]
            pooled_batch_features: Float[Tensor, "_ D"] = F.adaptive_avg_pool1d(
                batch_features, embed_dim
            )
            pooled_features_list.append(pooled_batch_features)
        features: Float[Tensor, "X*P D"] = torch.cat(pooled_features_list, dim=0)
        features: Float[Tensor, "X P D"] = features.reshape(
            features_.shape[0], features_.shape[1], features_.shape[-1]
        )
        return features
    return features_


def draw_rectangle(
    image: Shaped[Tensor, "*C H W"],
    index: int,
    patch_size: int,
    color: tuple[int, int, int] = (255, 0, 0),
    thickness: int = 1,
) -> Shaped[Tensor, "*C H W"]:
    color = tuple(reversed(color))  # type: ignore # Convert RGB to BGR for OpenCV
    image_np = to_cv2_image(image.clone())
    H, W = image_np.shape[:2]
    h_step = patch_size
    w_step = patch_size
    grid_w = W // w_step
    row = index // grid_w
    col = index % grid_w
    top_left = (col * w_step, row * h_step)
    bottom_right = ((col + 1) * w_step - 1, (row + 1) * h_step - 1)
    cv2.rectangle(
        image_np,
        top_left,
        bottom_right,
        color,
        thickness,
    )
    return from_cv2_image(image_np)


@jaxtyped(typechecker=None)
def get_image_patch(
    image: Shaped[Tensor, "*C H W"],
    patch_size: int,
    index: int,
) -> Shaped[Tensor, "*C h w"]:
    if len(image.shape) == 2:
        image = image.unsqueeze(0)
    C, H, W = image.shape
    assert (
        H % patch_size == 0 and W % patch_size == 0
    ), f"Image dimensions must be divisible by patch size: H {H}, W {W}, patch_size {patch_size}"
    patch_h, patch_w = H // patch_size, W // patch_size
    h_step = patch_size
    w_step = patch_size
    row = index // patch_w
    col = index % patch_w
    patch = image[
        :, row * h_step : (row + 1) * h_step, col * w_step : (col + 1) * w_step
    ]
    if patch.shape[0] == 1:
        patch = patch.squeeze(0)
    return patch


@jaxtyped(typechecker=None)
def get_mask_anomaly_indices(
    mask: Bool[Tensor, "H W"],
    patch_size: int,
) -> list[int]:
    patch_h, patch_w = mask.shape[0] // patch_size, mask.shape[1] // patch_size
    anomaly_patches = []
    anomaly_sums = []
    patch_num = patch_h * patch_w
    for i in range(patch_num):
        patch = get_image_patch(mask, patch_size, i)
        anomaly_sum = patch.sum().item()
        if anomaly_sum > 0:
            anomaly_patches.append(i)
            anomaly_sums.append(anomaly_sum)
    sorted_anomaly_patches = [
        x[0]
        for x in sorted(
            zip(anomaly_patches, anomaly_sums), key=lambda x: x[1], reverse=True
        )
    ]
    return sorted_anomaly_patches
