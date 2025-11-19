from pathlib import Path

import torch
from data.cached_impl import RealIADDevidedByAngle
from data.utils import Transform
from evaluator.align import AlignedDataset
from evaluator.analysis import analyze_errors_by_csv, read_scores_csv
from evaluator.musc2 import MuScConfig2, MuScDetector2
import evaluator.reproducibility as repro
from jaxtyping import Int, Shaped
from torch import Tensor


def reorder_image(
    image: Shaped[Tensor, "*C H W"],
    indices: Int[Tensor, "PH PW"],
) -> Shaped[Tensor, "*C H W"]:
    squeezed = False
    if len(image.shape) == 2:
        squeezed = True
        image = image.unsqueeze(0)
    C, H, W = image.shape
    PH, PW = indices.shape
    assert (
        H % PH == 0 and W % PW == 0
    ), "Image dimensions must be divisible by patch grid size"
    h_step = H // PH
    w_step = W // PW
    reordered_patches = []
    for i in range(PH):
        for j in range(PW):
            idx = indices[i, j]
            row = idx // PW
            col = idx % PW
            patch = image[
                :, row * h_step : (row + 1) * h_step, col * w_step : (col + 1) * w_step
            ]
            reordered_patches.append(patch)
    reordered_image = torch.zeros_like(image)
    for i in range(PH):
        for j in range(PW):
            reordered_image[
                :,
                i * h_step : (i + 1) * h_step,
                j * w_step : (j + 1) * w_step,
            ] = reordered_patches[i * PW + j]
    if squeezed:
        reordered_image = reordered_image.squeeze(0)
    return reordered_image


if __name__ == "__main__":
    seed = 42
    repro.init(seed)
    categories = [
        "audiojack_C1",
        "audiojack_C2",
        "audiojack_C3",
        "audiojack_C4",
        "audiojack_C5",
    ]
    datasets = [AlignedDataset(RealIADDevidedByAngle().filter_categories(categories))]
    batch_size = 16
    config = MuScConfig2()
    detector = MuScDetector2(config)
    for category in categories:
        meta_dataset = datasets[0].get_meta(category)
        tensor_dataset = datasets[0].get_tensor(category, detector.transform)
        csv_path = Path(
            f"results/musc_aligned/MuSc2_RealIAD(angle)(aligned)_scores/{category}.csv"
        )
        result = analyze_errors_by_csv(scores_csv=csv_path, dataset=meta_dataset)
        index = result.fn_indices[0]
        csv_indices, _, _ = read_scores_csv(csv_path)
        batched_csv_indices = [
            csv_indices[i : i + batch_size]
            for i in range(0, len(csv_indices), batch_size)
        ]
        batch_indice = list(filter(lambda x: index in x, batched_csv_indices))[0]
        batch_index = batch_indice.index(index)
        images = torch.stack([tensor_dataset[i].image for i in batch_indice])
        output = detector(images, "")
        min_indices: Int[Tensor, "B L R (B-1) P"] = output.other
        min_indices: Int[Tensor, "(B-1) P"] = min_indices[batch_index, 0, 0]
        min_indices: Int[Tensor, "(B-1) PH PW"] = min_indices.view(
            -1, int(min_indices.shape[-1] ** 0.5), int(min_indices.shape[-1] ** 0.5)
        )
        tensor_dataset = datasets[0].get_tensor(
            category, Transform(detector.transform.resize)
        )
        images = torch.stack([tensor_dataset[i].image for i in batch_indice])
        image = images[batch_index]
        other_images = [images[i] for i in range(len(batch_indice)) if i != batch_index]
        other_images = [
            reorder_image(other_images[i], min_indices[i])
            for i in range(len(other_images))
        ]
        # print image on left, other_images on right
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(4, 4, figsize=(15, 15), dpi=300)
        axs[0][0].imshow(image.permute(1, 2, 0).cpu())
        # axs[0][0].set_title(f"Original Image (Index: {index})")
        axs[0][0].axis("off")
        for i, other_image in enumerate(other_images):
            i = i + 1
            axs[i // 4, i % 4].imshow(other_image.permute(1, 2, 0).cpu())
            # axs[i // 4, i % 4].set_title(f"Reordered Image {i}")
            axs[i // 4, i % 4].axis("off")
        plt.tight_layout()
        plt.savefig(
            f"results/reordered_images_{category}_index_{index}.png",
            dpi=300,
            bbox_inches="tight",
        )
