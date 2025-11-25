from pathlib import Path
import matplotlib.pyplot as plt
import torch.nn.functional as F
from jaxtyping import Shaped
from torch import Tensor
from torchvision.transforms import CenterCrop

from data.utils import to_numpy_image




def show_images(
    images: list[Shaped[Tensor, "*C H W"]],
    save_path: Path,
    titles: list[str] | None = None,
    center_crop_size: int = 518,
    column_n: int = 4,
):
    rows = (len(images) + column_n - 1) // column_n
    fig, axs = plt.subplots(rows, column_n, figsize=(4 * column_n, 4 * rows), dpi=200)
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
        axs[i // column_n, i % column_n].imshow(image)
        axs[i // column_n, i % column_n].axis("off")
        if titles and i < len(titles):
            axs[i // column_n, i % column_n].set_title(titles[i], fontsize=10)
    # 隐藏空白子图
    for i in range(len(images), rows * column_n):
        axs[i // column_n, i % column_n].axis("off")
    plt.tight_layout()
    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )
