from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm
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
from jaxtyping import Int, Shaped, Bool, Float
from torch import Tensor, device, flatten
import matplotlib.pyplot as plt
from torchvision.transforms import CenterCrop
import cv2
import torch.nn.functional as F

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
    thickness: int = 1,
) -> Shaped[Tensor, "*C H W"]:
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


def draw_text(
    image: Shaped[Tensor, "*C H W"],
    index: int,
    text: str,
) -> Shaped[Tensor, "*C H W"]:
    image_np = to_cv2_image(image.clone())
    H, W = image_np.shape[:2]
    patch_h, patch_w = H // patch_size, W // patch_size
    h_step = patch_size
    w_step = patch_size
    row = index // patch_w
    col = index % patch_w
    # 计算文本位置（patch 中心）
    center_x = col * w_step + w_step // 2
    center_y = row * h_step + h_step // 2
    # 设置字体参数
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    color = (0, 0, 255)
    thickness = 1
    # 获取文本尺寸以居中显示
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = center_x - text_size[0] // 2
    text_y = center_y + text_size[1] // 2
    cv2.putText(
        image_np,
        text,
        (text_x, text_y),
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
    return from_cv2_image(image_np)


def get_sub_image(
    image: Shaped[Tensor, "*C H W"],
    center_index: int,
    size: int = 5,
) -> Shaped[Tensor, "*C h w"]:
    ph, pw = image.shape[-2] // patch_size, image.shape[-1] // patch_size
    center_row = center_index // pw
    center_col = center_index % pw
    half_size = size // 2
    start_row = max(0, center_row - half_size)
    end_row = min(ph, center_row + half_size + 1)
    start_col = max(0, center_col - half_size)
    end_col = min(pw, center_col + half_size + 1)
    h_start = start_row * patch_size
    h_end = end_row * patch_size
    w_start = start_col * patch_size
    w_end = end_col * patch_size
    sub_image = image[:, h_start:h_end, w_start:w_end]
    return sub_image


def show_images(
    images: list[Shaped[Tensor, "*C H W"]],
    save_path: Path,
    titles: list[str] | None = None,
    center_crop_size: int = 336,
):
    rows = (len(images) + 3) // 4
    fig, axs = plt.subplots(rows, 4, figsize=(15, 4 * rows), dpi=300)
    # 处理单行的情况
    if rows == 1:
        axs = axs.reshape(1, -1)

    # 找到最大尺寸
    max_h = max(img.shape[-2] for img in images)
    max_w = max(img.shape[-1] for img in images)

    for i, image in enumerate(images):
        # 如果图像尺寸小于最大尺寸，则缩放
        if image.shape[-2] < max_h or image.shape[-1] < max_w:
            if len(image.shape) == 2:
                image = image.unsqueeze(0)
                need_squeeze = True
            else:
                need_squeeze = False
            image = F.interpolate(
                image.unsqueeze(0),
                size=(max_h, max_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            if need_squeeze:
                image = image.squeeze(0)

        image = CenterCrop(center_crop_size)(image)
        image = to_numpy_image(image)
        axs[i // 4, i % 4].imshow(image)
        axs[i // 4, i % 4].axis("off")
        if titles and i < len(titles):
            axs[i // 4, i % 4].set_title(titles[i], fontsize=10)
    # 隐藏空白子图
    for i in range(len(images), rows * 4):
        axs[i // 4, i % 4].axis("off")
    plt.tight_layout()
    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )


def get_batch_indices(
    all_indices: list[int],
    batch_size: int,
    target_index: int,
) -> list[int]:
    all_batch_indices = [
        all_indices[i : i + batch_size] for i in range(0, len(all_indices), batch_size)
    ]
    target_batch_indices = list(filter(lambda x: target_index in x, all_batch_indices))
    assert len(target_batch_indices) == 1
    return target_batch_indices[0]


def compute_results(
    detector: MuScDetector2,
    images: Float[Tensor, "B C H W"],
    target_index: int,
) -> tuple[
    float,  # 异常分数
    int,  # 最大异常分数对应的 patch 索引
    Int[Tensor, "L R B-1 P"],  # 匹配的 patch 索引
    Int[Tensor, "L R topmink P"],  # topkmin 匹配的 图像索引
    Float[Tensor, "L R topmink P"],  # topkmin 匹配的 分数
]:
    output = detector(images, "")
    min_indices, max_indices, topmink_indices, topmink_scores = output.other
    min_indices: Int[Tensor, "B L R (B-1) P"]
    max_indices: Int[Tensor, "B"]
    topmink_indices: Int[Tensor, "B L R topmink P"]
    topmink_scores: Float[Tensor, "B L R topmink P"]
    score = float(output.pred_scores[target_index].item())
    min_indices: Int[Tensor, "L R B-1 P"] = min_indices[target_index]
    max_index = int(max_indices[target_index].item())
    topmink_indices: Int[Tensor, "L R topmink P"] = topmink_indices[target_index]
    topmink_scores: Float[Tensor, "L R topmink P"] = topmink_scores[target_index]
    return (
        score,
        max_index,
        min_indices,
        topmink_indices,
        topmink_scores,
    )


def generate_relative_patch_fig(
    image: Float[Tensor, "C H W"],
    other_images: list[Float[Tensor, "C H W"]],
    target_pidx: int,
    match_pindices_: Int[Tensor, "L R B-1 P"],
    topmink_imgindices_: Int[Tensor, "L R topmink P"],
    save_path: Path,
):
    # 根据 topkmin 的索引对 min_indices进行筛选
    match_pindices: Int[Tensor, "L R B-1"] = match_pindices_[:, :, :, target_pidx]
    topmink_imgindices: Int[Tensor, "L R topmink"] = topmink_imgindices_[
        :, :, :, target_pidx
    ]
    # print(f"topmink scores: {topmink_imgindices}")
    match_pindices: Int[Tensor, "L*R B-1"] = match_pindices.reshape(
        -1, match_pindices.shape[-1]
    )
    topmink_imgindices: Int[Tensor, "L*R topmink"] = topmink_imgindices.reshape(
        -1, topmink_imgindices.shape[-1]
    )
    other_image_patches: dict[int, list[int]] = {}
    for pindices, imgindices in zip(match_pindices, topmink_imgindices):
        for img_idx in imgindices:
            other_image_patches.setdefault(int(img_idx), []).append(
                int(pindices[img_idx])
            )
    other_image_patch_nums: dict[int, dict[int, int]] = {}
    for img_idx, pindices in other_image_patches.items():
        other_image_patch_nums[img_idx] = {}
        for patch_idx in pindices:
            other_image_patch_nums[img_idx][patch_idx] = (
                other_image_patch_nums[img_idx].get(patch_idx, 0) + 1
            )

    other_images_with_rect = other_images
    for img_idx, patch_nums in other_image_patch_nums.items():
        for patch_idx, num in patch_nums.items():
            other_images_with_rect[img_idx] = draw_text(
                other_images_with_rect[img_idx], patch_idx, str(num)
            )
            other_images_with_rect[img_idx] = draw_rectangle(
                other_images_with_rect[img_idx],
                patch_idx,
            )
    origin_image_with_rect = draw_rectangle(image, target_pidx)

    show_images(
        [origin_image_with_rect] + other_images_with_rect,
        save_path=save_path,
    )


def generate_relative_patch_fig_v2(
    image: Float[Tensor, "C H W"],
    other_images: list[Float[Tensor, "C H W"]],
    score: float,
    target_pidx: int,
    match_pindices_: Int[Tensor, "L R B-1 P"],
    topmink_imgindices_: Int[Tensor, "L R topmink P"],
    topmink_scores_: Float[Tensor, "L R topmink P"],
    save_path: Path,
):
    # 根据 topkmin 的索引对 min_indices进行筛选
    match_pindices: Int[Tensor, "L R B-1"] = match_pindices_[:, :, :, target_pidx]
    topmink_imgindices: Int[Tensor, "L R topmink"] = topmink_imgindices_[
        :, :, :, target_pidx
    ]
    topmink_scores: Float[Tensor, "L R topmink"] = topmink_scores_[:, :, :, target_pidx]
    titles = []
    images = []
    image = draw_rectangle(image, target_pidx)
    images.append(image)
    titles.append(f"Origin Image Score: {score:.3f}")
    images.append(torch.ones_like(image))
    images.append(torch.ones_like(image))
    images.append(torch.ones_like(image))
    titles.extend(["", "", ""])
    for l, (sub_pindices, sub_imgindices, sub_scores) in enumerate(
        zip(match_pindices, topmink_imgindices, topmink_scores)
    ):
        for r, (pindices, imgindices, scores) in enumerate(
            zip(sub_pindices, sub_imgindices, sub_scores)
        ):
            for img_idx, pscore in zip(imgindices, scores):
                image = other_images[int(img_idx)]
                pidx = int(pindices[img_idx])
                image = draw_rectangle(image, pidx)
                image = get_sub_image(image, pidx, size=7)
                images.append(image)
                titles.append(f"L{l}_R{r}_S{pscore:.3f}")

    show_images(images, titles=titles, save_path=save_path)


def main():
    seed = 42
    repro.init(seed)
    categories = [
        # "audiojack_C1",
        "audiojack_C2",
        # "audiojack_C3",
        # "audiojack_C4",
        # "audiojack_C5",
    ]
    indices = (
        [499, 734, 527, 342, 471, 695, 703, 419, 451, 482, 383]
        + [671, 627, 762, 387, 306, 439, 626, 467, 630]
        + [267, 336, 442, 585, 247, 393, 299, 494, 686, 466, 278, 367, 453, 479, 506]
        + [262, 670, 754, 427, 598, 726, 331, 682, 249, 391, 591, 757, 718, 431]
        + [685, 246, 394, 274, 743, 603, 552, 399, 375]
    )
    scores_dir = Path("results/musc_11_21_act/MuSc2(r13)(k1)_RealIAD(angle)_s42_scores")
    dataset = RealIADDevidedByAngle().filter_categories(categories)
    batch_size = 16
    config = MuScConfig2()
    config.k_list = [1]
    config.r_list = [1, 3]
    detector = MuScDetector2(config)
    category = categories[0]
    meta_dataset = dataset.get_meta(category)
    tensor_dataset = dataset.get_tensor(category, detector.transform)
    tensor_dataset_only_resize = dataset.get_tensor(
        category, Transform(detector.transform.resize)
    )
    train_dataset = dataset.get_train_tensor(category, detector.transform)
    train_dataset_only_resize = dataset.get_train_tensor(
        category, Transform(detector.transform.resize)
    )
    csv_path = scores_dir / f"{category}.csv"
    # compute
    csv_indices, _, csv_scores = read_scores_csv(csv_path)
    is_train = True
    if not is_train:
        save_dir = Path(f"results/error_analysis_v2/{category}")
    else:
        save_dir = Path(f"results/error_analysis_v2_train/{category}")
    save_dir.mkdir(parents=True, exist_ok=True)
    for img_idx in tqdm(indices):
        batch_indices = get_batch_indices(csv_indices, batch_size, img_idx)

        if not is_train:
            input_images = torch.stack([tensor_dataset[i].image for i in batch_indices])
            batch_idx = batch_indices.index(img_idx)
        else:
            input_images = torch.stack(
                [tensor_dataset[img_idx].image]
                + [train_dataset[i] for i in range(batch_size - 1)]
            )
            batch_idx = 0
        score, max_pidx, min_pindices, topmink_imgindices, topmink_scores = (
            compute_results(detector, input_images, batch_idx)
        )
        min_pindices: Int[Tensor, "L R B-1 P"]
        topmink_imgindices: Int[Tensor, "L R topmink P"]
        topmink_scores: Float[Tensor, "L R topmink P"]
        if not is_train:
            score_ = [x[1] for x in zip(csv_indices, csv_scores) if x[0] == img_idx][0]
            assert round(score, 2) == round(score_, 2), (score, score_)
        score__ = torch.mean(topmink_scores[:, :, :, max_pidx]).item()
        assert round(score, 2) == round(score__, 2), (score, score__)

        # get images
        if not is_train:
            images = torch.stack(
                [tensor_dataset_only_resize[i].image for i in batch_indices]
            )
        else:
            images = torch.stack(
                [tensor_dataset_only_resize[img_idx].image]
                + [train_dataset_only_resize[i] for i in range(batch_size - 1)]
            )
        image = images[batch_idx]
        mask = tensor_dataset_only_resize[img_idx].mask
        anomaly_indices = get_mask_anomaly_indices(mask)
        other_images = [image for i, image in enumerate(images) if i != batch_idx]

        generate_relative_patch_fig_v2(
            image,
            other_images,
            score,
            max_pidx,
            min_pindices,
            topmink_imgindices,
            topmink_scores,
            save_path=save_dir / f"{img_idx}.png",
        )
        gt_patch_index = int(anomaly_indices[0])
        gt_score = torch.mean(topmink_scores[:, :, :, gt_patch_index]).item() 
        generate_relative_patch_fig_v2(
            image,
            other_images,
            gt_score,
            gt_patch_index,
            min_pindices,
            topmink_imgindices,
            topmink_scores,
            save_path=save_dir / f"{img_idx}_gt.png",
        )

        # other_images_with_rect = []
        # for other_image, i_match_indices in zip(other_images, match_indices):
        #     i_match_indices_dict:dict[int, int]= {}
        #     for index in i_match_indices.flatten():
        #         index = int(index)
        #         i_match_indices_dict[index] = i_match_indices_dict.get(index, 0) + 1
        #     img_with_rect = other_image
        #     for i, n in i_match_indices_dict.items():
        #         img_with_rect = draw_rectangle(img_with_rect, i, thickness=n)
        #     other_images_with_rect.append(img_with_rect)
        # origin_image_with_rect = draw_rectangle(image, focus_index)
        # show_images(
        #     [origin_image_with_rect] + other_images_with_rect,
        #     save_path=f"results/{category}_images.png",
        # )

        # other_images_with_rect = [
        #     draw_rectangle(img, int(indices[max_index]))
        #     for img, indices in zip(other_images, min_indices)
        # ]


if __name__ == "__main__":
    main()
