from pathlib import Path
from PIL import Image
import numpy as np
from jaxtyping import Float, Bool, UInt8

type ImageSize = tuple[int, int]  # (width, height)


def compute_image_size(origin: ImageSize, shortest_side: int):
    old_short, old_long = min(origin), max(origin)
    new_short = shortest_side
    new_long = int(old_long * (new_short / old_short))
    if origin[0] < origin[1]:
        return (new_short, new_long)
    else:
        return (new_long, new_short)


def generate_image(
    image_path: Path, image_size: ImageSize | int | None = None
) -> UInt8[np.ndarray, "C=3 H W"]:
    img = Image.open(image_path).convert("RGB")
    if isinstance(image_size, int):
        image_size = compute_image_size(img.size, image_size)
    if image_size is not None and img.size != image_size:
        img = img.resize(image_size, Image.Resampling.BICUBIC)
    img = np.array(img)
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    return img


def resize_image(
    image: UInt8[np.ndarray, "C=3 H W"], image_size: ImageSize | int
) -> UInt8[np.ndarray, "C=3 H2 W2"]:
    origin = (image.shape[2], image.shape[1])
    if isinstance(image_size, int):
        image_size = compute_image_size(origin, image_size)
    if origin != image_size:
        img = np.transpose(image, (1, 2, 0))  # CHW to HWC
        img = Image.fromarray(img)
        img = img.resize(image_size, Image.Resampling.BICUBIC)
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
    mask_path: Path, image_size: ImageSize | int | None = None
) -> Bool[np.ndarray, "H W"]:
    img_mask = Image.open(mask_path).convert("L")
    img_mask = (np.array(img_mask) > 0).astype(
        np.uint8
    ) * 255  # 将图片中的掩码部分变为255，非掩码部分为0
    img_mask = Image.fromarray(img_mask)
    # size: (W, H)
    if isinstance(image_size, int):
        image_size = compute_image_size(img_mask.size, image_size)
    if image_size is not None and img_mask.size != image_size:
        img_mask = img_mask.resize(image_size, Image.Resampling.BILINEAR)
    img_mask = np.array(img_mask)
    img_mask = img_mask > 127  # 二值化
    return img_mask.astype(bool)


def resize_mask(
    mask: Bool[np.ndarray, "H W"], image_size: ImageSize | int
) -> Bool[np.ndarray, "H2 W2"]:
    origin = (mask.shape[1], mask.shape[0])
    if isinstance(image_size, int):
        image_size = compute_image_size(origin, image_size)
    if origin != image_size:
        img_mask = Image.fromarray((mask.astype(np.uint8)) * 255)
        img_mask = img_mask.resize(image_size, Image.Resampling.BILINEAR)
        img_mask = np.array(img_mask)
        img_mask = img_mask > 127  # 二值化
        return img_mask.astype(bool)
    return mask


def generate_empty_mask(image_size: ImageSize) -> Bool[np.ndarray, "H W"]:
    return np.zeros((image_size[1], image_size[0]), dtype=bool)
