from pathlib import Path
from PIL import Image
import numpy as np
from jaxtyping import Float, Bool

type ImageSize = tuple[int, int]  # (width, height)


def generate_image(
    image_path: Path, image_size: ImageSize | None = None
) -> Float[np.ndarray, "C=3 H W"]:
    img = Image.open(image_path).convert("RGB")
    if image_size is not None and img.size != image_size:
        img = img.resize(image_size, Image.Resampling.LANCZOS)
    img = np.array(img).astype(np.float32) / 255.0  # 归一化到0-1
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    return img


def resize_image(
    image: Float[np.ndarray, "C=3 H W"], image_size: ImageSize
) -> Float[np.ndarray, "C=3 H2 W2"]:
    if image.shape[1:] != (image_size[1], image_size[0]):
        img = np.transpose(image, (1, 2, 0))  # CHW to HWC
        img = Image.fromarray((img * 255).astype(np.uint8))
        img = img.resize(image_size, Image.Resampling.LANCZOS)
        img = np.array(img).astype(np.float32) / 255.0  # 归一化到0-1
        img = np.transpose(img, (2, 0, 1))  # HWC to CHW
        return img
    return image


def generate_mask(
    mask_path: Path, image_size: ImageSize | None = None
) -> Bool[np.ndarray, "H W"]:
    img_mask = Image.open(mask_path).convert("L")
    img_mask = (np.array(img_mask) > 0).astype(
        np.uint8
    ) * 255  # 将图片中的掩码部分变为255，非掩码部分为0
    img_mask = Image.fromarray(img_mask)
    # size: (W, H)
    if image_size is not None and img_mask.size != image_size:
        img_mask = img_mask.resize(image_size, Image.Resampling.BILINEAR)
    img_mask = np.array(img_mask)
    img_mask = img_mask > 127  # 二值化
    return img_mask.astype(bool)


def resize_mask(
    mask: Bool[np.ndarray, "H W"], image_size: ImageSize
) -> Bool[np.ndarray, "H2 W2"]:
    if mask.shape != (image_size[1], image_size[0]):
        img_mask = Image.fromarray((mask.astype(np.uint8)) * 255)
        img_mask = img_mask.resize(image_size, Image.Resampling.BILINEAR)
        img_mask = np.array(img_mask)
        img_mask = img_mask > 127  # 二值化
        return img_mask.astype(bool)
    return mask


def generate_empty_mask(image_size: ImageSize) -> Bool[np.ndarray, "H W"]:
    return np.zeros((image_size[1], image_size[0]), dtype=bool)
