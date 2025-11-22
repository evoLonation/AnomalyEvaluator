from pathlib import Path

import torch
from data.cached_impl import RealIADDevidedByAngle
from data.utils import (
    Transform,
    from_cv2_image,
    resize_image,
    to_cv2_image,
    to_numpy_image,
)
from align.rect import AlignedDataset
from evaluator.analysis import analyze_errors_by_csv, read_scores_csv
from evaluator.musc2 import MuScConfig2, MuScDetector2
import evaluator.reproducibility as repro
from jaxtyping import Int, Shaped, Bool
from torch import Tensor
import matplotlib.pyplot as plt
from torchvision.transforms import CenterCrop

patch_size = 14


def get_image_patch(
    image: Shaped[Tensor, "*C H W"],
    index: int,
) -> Shaped[Tensor, "*C h w"]:
    if len(image.shape) == 2:
        image = image.unsqueeze(0)
    C, H, W = image.shape
    assert (
        H % patch_size == 0 and W % patch_size == 0
    ), "Image dimensions must be divisible by patch size"
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


def reorder_image(
    image: Shaped[Tensor, "*C H W"],
    indices: Int[Tensor, "P"],
) -> Shaped[Tensor, "*C H W"]:
    ph, pw = image.shape[1] // patch_size, image.shape[2] // patch_size
    assert ph * pw == indices.shape[0]
    reordered_patches = []
    for idx in indices:
        patch = get_image_patch(image, int(idx.item()))
        reordered_patches.append(patch)
    reorder_image = torch.zeros_like(image)
    for i in range(ph):
        for j in range(pw):
            patch = reordered_patches[i * pw + j]
            h_start = i * patch_size
            w_start = j * patch_size
            reorder_image[
                :, h_start : h_start + patch_size, w_start : w_start + patch_size
            ] = patch
    return reorder_image


def get_mask_anomaly_indices(
    mask: Bool[Tensor, "H W"],
) -> list[int]:
    patch_h, patch_w = mask.shape[0] // patch_size, mask.shape[1] // patch_size
    anomaly_patches = []
    anomaly_sums = []
    patch_num = patch_h * patch_w
    for i in range(patch_num):
        patch = get_image_patch(mask, i)
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


def draw_rectangle(
    image: Shaped[Tensor, "*C H W"],
    index: int,
    color: tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
) -> Shaped[Tensor, "*C H W"]:
    import cv2

    image_np = to_cv2_image(image.clone())
    H, W = image_np.shape[:2]
    patch_h, patch_w = H // patch_size, W // patch_size
    h_step = patch_size
    w_step = patch_size
    row = index // patch_w
    col = index % patch_w
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


def show_images(
    images: list[Shaped[Tensor, "*C H W"]],
    save_path: str,
    center_crop_size: int = 336,
):
    import matplotlib.pyplot as plt

    rows = (len(images) + 3) // 4
    fig, axs = plt.subplots(rows, 4, figsize=(15, 4 * rows), dpi=300)
    for i, image in enumerate(images):
        image = CenterCrop(center_crop_size)(image)
        image = to_numpy_image(image)
        axs[i // 4, i % 4].imshow(image)
        axs[i // 4, i % 4].axis("off")
    plt.tight_layout()
    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    seed = 42
    repro.init(seed)
    categories = [
        # "audiojack_C1",
        "audiojack_C2",
        # "audiojack_C3",
        # "audiojack_C4",
        # "audiojack_C5",
    ]
    scores_dir = Path("results/musc_11_21_act/MuSc2(r13)(k1)_RealIAD(angle)_s42_scores")
    dataset = RealIADDevidedByAngle().filter_categories(categories)
    batch_size = 16
    config = MuScConfig2()
    config.k_list = [1]
    config.r_list = [1, 3]
    detector = MuScDetector2(config)
    for category in categories:
        meta_dataset = dataset.get_meta(category)
        tensor_dataset = dataset.get_tensor(category, detector.transform)
        csv_path = scores_dir / f"{category}.csv"
        index = 499

        # compute
        csv_indices, _, csv_scores = read_scores_csv(csv_path)
        score = [x[1] for x in zip(csv_indices, csv_scores) if x[0] == index][0]
        batched_csv_indices = [
            csv_indices[i : i + batch_size]
            for i in range(0, len(csv_indices), batch_size)
        ]
        batch_indices = list(filter(lambda x: index in x, batched_csv_indices))[0]
        batch_index = batch_indices.index(index)
        images = torch.stack([tensor_dataset[i].image for i in batch_indices])
        output = detector(images, "")
        assert score == round(float(output.pred_scores[batch_index].item()), 4), (
            score,
            output.pred_scores[batch_index].item(),
        )
        min_indices, max_indices = output.other
        min_indices: Int[Tensor, "B L R (B-1) P"]
        max_indices: Int[Tensor, "B"]
        min_indices: Int[Tensor, "(B-1) P"] = min_indices[batch_index, 0, 0]
        max_index = int(max_indices[batch_index].item())

        # get images
        tensor_dataset = dataset.get_tensor(
            category, Transform(detector.transform.resize)
        )
        images = torch.stack([tensor_dataset[i].image for i in batch_indices])
        image = images[batch_index]
        mask = tensor_dataset[batch_indices[batch_index]].mask
        anomaly_indices = get_mask_anomaly_indices(mask)
        other_images = [
            images[i] for i in range(len(batch_indices)) if i != batch_index
        ]
        # other_images = [
        #     reorder_image(other_images[i], min_indices[i])
        #     for i in range(len(other_images))
        # ]

        # max_index = anomaly_indices[0]
        origin_patch = get_image_patch(image, max_index)
        other_patches = [
            get_image_patch(image, int(indices[max_index]))
            for image, indices in zip(other_images, min_indices)
        ]

        origin_image_with_rect = draw_rectangle(image, max_index)
        other_images_with_rect = [
            draw_rectangle(img, int(indices[max_index]))
            for img, indices in zip(other_images, min_indices)
        ]
        show_images(
            [origin_image_with_rect] + other_images_with_rect,
            save_path=f"results/{category}_images.png",
        )
