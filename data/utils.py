from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from PIL import Image
import numpy as np
from jaxtyping import Float, Bool, UInt8
import torch

type ImageTransform = Callable[
    [Float[torch.Tensor, "... C H W"]], Float[torch.Tensor, "... C H2 W2"]
]
type MaskTransform = Callable[
    [Bool[torch.Tensor, "... H W"]], Bool[torch.Tensor, "... H2 W2"]
]


@dataclass(kw_only=True)
class ImageSize:
    h: int
    w: int

    def hw(self) -> tuple[int, int]:
        return (self.h, self.w)

    def numpy(self) -> tuple[int, int]:
        return self.hw()

    def tensor(self) -> tuple[int, int]:
        return self.hw()

    def wh(self) -> tuple[int, int]:
        return (self.w, self.h)

    def pil(self) -> tuple[int, int]:
        return self.wh()

    @staticmethod
    def fromwh(size: tuple[int, int]) -> "ImageSize":
        return ImageSize(h=size[1], w=size[0])

    @staticmethod
    def frompil(size: tuple[int, int]) -> "ImageSize":
        return ImageSize.fromwh(size)

    @staticmethod
    def fromnumpy(shape: tuple[int, ...]) -> "ImageSize":
        return ImageSize(h=shape[-2], w=shape[-1])

    @staticmethod
    def fromtensor(shape: torch.Size) -> "ImageSize":
        return ImageSize(h=shape[-2], w=shape[-1])

    @staticmethod
    def square(size: int) -> "ImageSize":
        return ImageSize(h=size, w=size)


type ImageResize = ImageSize | int
"""
if int, means shortest side
"""


def compute_image_size(origin: ImageSize, shortest_side: int) -> ImageSize:
    old_short, old_long = min(origin.hw()), max(origin.hw())
    new_short = shortest_side
    new_long = int(old_long * (new_short / old_short))
    if origin.h < origin.w:
        return ImageSize(h=new_short, w=new_long)
    else:
        return ImageSize(h=new_long, w=new_short)


def generate_image(
    image_path: Path, resize: ImageResize | None = None
) -> UInt8[np.ndarray, "C=3 H W"]:
    img = Image.open(image_path).convert("RGB")
    if isinstance(resize, int):
        resize = compute_image_size(ImageSize.frompil(img.size), resize)
    if resize is not None and img.size != resize.pil():
        img = img.resize(resize.pil(), Image.Resampling.BICUBIC)
    img = np.array(img)
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    return img


def resize_image(
    image: UInt8[np.ndarray, "C=3 H W"], resize: ImageResize
) -> UInt8[np.ndarray, "C=3 H2 W2"]:
    origin = ImageSize.fromnumpy(image.shape)
    if isinstance(resize, int):
        resize = compute_image_size(origin, resize)
    if origin != resize:
        img = np.transpose(image, (1, 2, 0))  # CHW to HWC
        img = Image.fromarray(img)
        img = img.resize(resize.pil(), Image.Resampling.BICUBIC)
        img = np.array(img)
        img = np.transpose(img, (2, 0, 1))  # HWC to CHW
        return img
    return image


def normalize_image(
    image: UInt8[np.ndarray, "C=3 H W"],
) -> Float[np.ndarray, "C=3 H W"]:
    """将图像像素值归一化到[0, 1]范围内"""
    return image.astype(np.float32) / 255.0


def generate_mask(
    mask_path: Path, resize: ImageResize | None = None
) -> Bool[np.ndarray, "H W"]:
    img_mask = Image.open(mask_path).convert("L")
    img_mask = (np.array(img_mask) > 0).astype(
        np.uint8
    ) * 255  # 将图片中的掩码部分变为255，非掩码部分为0
    img_mask = Image.fromarray(img_mask)
    # size: (W, H)
    if isinstance(resize, int):
        resize = compute_image_size(ImageSize.frompil(img_mask.size), resize)
    if resize is not None and img_mask.size != resize.pil():
        img_mask = img_mask.resize(resize.pil(), Image.Resampling.BILINEAR)
    img_mask = np.array(img_mask)
    img_mask = img_mask > 127  # 二值化
    return img_mask.astype(bool)


def resize_mask(
    mask: Bool[np.ndarray, "H W"], resize: ImageResize
) -> Bool[np.ndarray, "H2 W2"]:
    origin = ImageSize.fromnumpy(mask.shape)
    if isinstance(resize, int):
        resize = compute_image_size(origin, resize)
    if origin != resize:
        img_mask = Image.fromarray((mask.astype(np.uint8)) * 255)
        img_mask = img_mask.resize(resize.pil(), Image.Resampling.BILINEAR)
        img_mask = np.array(img_mask)
        img_mask = img_mask > 127  # 二值化
        return img_mask.astype(bool)
    return mask


def generate_empty_mask(image_size: ImageSize) -> Bool[np.ndarray, "H W"]:
    return np.zeros(image_size.numpy(), dtype=bool)
