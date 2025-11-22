from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Union
from PIL import Image
import h5py
import numpy as np
from jaxtyping import Float, Bool, UInt8, Shaped
from torch import Tensor, tensor
import torch
from torchvision.transforms import functional as F


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
    def fromnumpy(array: np.ndarray) -> "ImageSize":
        shape = array.shape
        assert 2 <= 2 <= len(shape) <= 3, "Array must be 2D or 3D."
        return ImageSize(h=shape[0], w=shape[1])

    @staticmethod
    def fromtensor(tensor: Tensor) -> "ImageSize":
        shape = tensor.shape
        assert 2 <= len(shape) <= 3, "Tensor must be 2D or 3D."
        return ImageSize(h=shape[-2], w=shape[-1])

    @staticmethod
    def square(size: int) -> "ImageSize":
        return ImageSize(h=size, w=size)

    def __str__(self) -> str:
        return f"{self.w}x{self.h}"


type ImageResize = ImageSize | int
"""
if int, means shortest side
"""

type ImageTransform = Callable[
    [Float[Tensor, "... C H W"]], Float[Tensor, "... C H2 W2"]
]
type MaskTransform = Callable[[Bool[Tensor, "... H W"]], Bool[Tensor, "... H2 W2"]]


@dataclass
class Transform:
    resize: ImageResize | None = None
    image_transform: ImageTransform = lambda x: x
    mask_transform: MaskTransform = lambda x: x


def compute_image_size(origin: ImageSize, shortest_side: int) -> ImageSize:
    old_short, old_long = min(origin.hw()), max(origin.hw())
    new_short = shortest_side
    new_long = int(old_long * (new_short / old_short))
    if origin.h < origin.w:
        return ImageSize(h=new_short, w=new_long)
    else:
        return ImageSize(h=new_long, w=new_short)


def resize_to_size(origin: ImageSize, resize: ImageResize) -> ImageSize:
    if isinstance(resize, int):
        return compute_image_size(origin, resize)
    else:
        return resize


def generate_image(
    image_path: Path, resize: ImageResize | None = None
) -> UInt8[Tensor, "C=3 H W"]:
    img = Image.open(image_path).convert("RGB")
    if isinstance(resize, int):
        resize = compute_image_size(ImageSize.frompil(img.size), resize)
    if resize is not None and img.size != resize.pil():
        img = img.resize(resize.pil(), Image.Resampling.BICUBIC)
    return to_tensor_image(img)


def save_image(image: Shaped[Tensor, "*C H H"], save_path: Path):
    img = to_pil_image(image)
    img.save(save_path)


def resize_image(
    image: UInt8[Tensor, "C=3 H W"], resize: ImageResize
) -> UInt8[Tensor, "C=3 H2 W2"]:
    device = image.device
    origin = ImageSize.fromtensor(image)
    resize = resize_to_size(origin, resize)
    if origin != resize:
        img = to_pil_image(image)
        img = img.resize(resize.pil(), Image.Resampling.BICUBIC)
        return to_tensor_image(img).to(device)
    return image


def normalize_image(
    image: UInt8[Tensor, "*C H W"],
) -> Float[Tensor, "*C H W"]:
    """将图像像素值归一化到[0, 1]范围内"""
    return image.to(torch.float32) / 255.0


def binarize_image(
    image: UInt8[Tensor, "*C H W"],
) -> Bool[Tensor, "*C H W"]:
    """将图像像素值二值化"""
    return image > 127


def denormalize_image(
    image: Float[Tensor, "*C H W"] | Bool[Tensor, "*C H W"],
) -> UInt8[Tensor, "*C H W"]:
    """将图像像素值从[0, 1]范围内反归一化到[0, 255]范围内"""
    image = torch.clip(image * 255.0, 0, 255)
    return image.to(torch.uint8)


def to_numpy_image(
    image: Shaped[Tensor, "*C H W"],
) -> UInt8[np.ndarray, "H W *C"]:
    if image.is_floating_point() or image.dtype == torch.bool:
        image = denormalize_image(image)
    else:
        assert image.dtype == torch.uint8, "Image tensor must be of type uint8."
    image_np = image.cpu().numpy()
    if len(image) == 3:
        image_np = np.transpose(image_np, (1, 2, 0))  # CHW to HWC
    return image_np


def to_pil_image(
    image: Shaped[Tensor, "*C H W"],
) -> Image.Image:
    return Image.fromarray(to_numpy_image(image))


def to_cv2_image(
    image: Shaped[Tensor, "*C H W"],
) -> UInt8[np.ndarray, "H W *C"]:
    """Convert a PyTorch tensor image to a CV2 image (BGR format)."""
    image_np = to_numpy_image(image)
    if len(image_np.shape) == 3:
        import cv2

        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        return image_np
    return image_np


def from_cv2_image(
    image: UInt8[np.ndarray, "H W *C"],
) -> UInt8[Tensor, "*C H W"]:
    """Convert a CV2 image (BGR format) to a PyTorch tensor image."""
    import cv2

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return to_tensor_image(image)


def to_tensor_image(
    image: Image.Image | UInt8[np.ndarray, "H W *C"],
) -> UInt8[Tensor, "*C H W"]:
    if isinstance(image, Image.Image):
        image = np.array(image)
        assert image.dtype == np.uint8, "Image array must be of type uint8."
    if len(image.shape) == 3:
        image = np.transpose(image, (2, 0, 1))  # HWC to CHW
    return tensor(image)


def generate_mask(
    mask_path: Path, resize: ImageResize | None = None
) -> Bool[Tensor, "H W"]:
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
    return torch.from_numpy(img_mask.astype(bool))


def resize_mask(
    mask: Bool[Tensor, "H W"], resize: ImageResize
) -> Bool[Tensor, "H2 W2"]:
    device = mask.device
    origin = ImageSize.fromtensor(mask)
    resize = resize_to_size(origin, resize)
    if origin != resize:
        img_mask = to_pil_image(mask)
        img_mask = img_mask.resize(resize.pil(), Image.Resampling.BILINEAR)
        img_mask = np.array(img_mask)
        img_mask = img_mask > 127  # 二值化
        return torch.from_numpy(img_mask.astype(bool)).to(device)
    return mask


def generate_empty_mask(image_size: ImageSize) -> Bool[Tensor, "H W"]:
    return torch.zeros(*image_size.tensor(), dtype=torch.bool)


def pad_to_square(image_tensor: Shaped[Tensor, "*C H W"], pad_value=0):
    """
    将宽高不一致的图片 Tensor 通过零填充变成方形。

    Args:
        image_tensor (Tensor): 输入图片 Tensor，形状为 (C, H, W) 或 (H, W)。
                                     如果为 (H, W)，会自动unsqueeze(0)变为 (1, H, W)。
        pad_value (int/float): 用于填充的值，默认为 0。

    Returns:
        Tensor: 填充后的方形图片 Tensor。
    """
    if image_tensor.dim() == 2:  # 处理灰度图 (H, W)
        image_tensor = image_tensor.unsqueeze(0)  # 变为 (1, H, W)
    C, H, W = image_tensor.shape
    max_dim = max(H, W)
    pad_h = max_dim - H
    pad_w = max_dim - W
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    # 使用 torchvision.transforms.functional.pad 进行填充
    # 参数顺序为 (left, top, right, bottom)
    # 注意：F.pad 接受的参数顺序是 (padding_left, padding_top, padding_right, padding_bottom)
    # 但实际应用时，为了方便理解，通常按照 (left, right, top, bottom) 来思考
    # F.pad(input, padding, fill=0, padding_mode='constant')
    padded_tensor = F.pad(
        image_tensor,
        [pad_left, pad_top, pad_right, pad_bottom],
        fill=pad_value,
        padding_mode="constant",
    )
    if padded_tensor.shape[0] == 1:
        padded_tensor = padded_tensor.squeeze(0)  # 恢复为 (H, W)
    return padded_tensor


@contextmanager
def h5writer(save_path: Path):
    tmp_save_path = save_path.with_suffix(".tmp")
    try:
        with h5py.File(tmp_save_path, "w") as h5f:
            yield h5f
    except BaseException:
        if tmp_save_path.exists():
            tmp_save_path.unlink()
        raise
    tmp_save_path.rename(save_path)
