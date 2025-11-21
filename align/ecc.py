from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import torch
from torch import Tensor, float32, isin, tensor
from jaxtyping import Float, Bool
import cv2
import numpy as np
from tqdm import tqdm
import torchvision.utils as vutils
from torchvision.transforms import CenterCrop

from align.base import Contour, get_contour_by_mask, get_image_product_mask
from align.rect import align_image, get_align_transform_matrix, get_image_min_area_rect
from data.base import (
    Dataset,
    DatasetInline,
    DatasetOverrideGetItem,
    ListDataset,
    ZipedDataset,
    tuple_collate_fn,
)
from data.cached_impl import RealIAD, RealIADDevidedByAngle
from data.detection_dataset import (
    DetectionDataset,
    MetaInfo,
    MetaSample,
    TensorSample,
)
from data.summary import generate_summary_view
from data.utils import (
    ImageResize,
    ImageSize,
    Transform,
    binarize_image,
    from_cv2_image,
    generate_empty_mask,
    generate_image,
    generate_mask,
    h5writer,
    normalize_image,
    resize_to_size,
    to_cv2_image,
    to_numpy_image,
)
from torch.utils.data import DataLoader
import h5py


def get_centroid(contour: Contour):
    """计算轮廓重心"""
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return np.array([0.0, 0.0])
    return np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]])


def align_contours_fine(src_cnt: Contour, dst_cnt: Contour, image_size: ImageSize):
    """
    使用 ECC 算法精细对齐两个轮廓，使其重合面积最大。
    """
    # 1. 预计算重心平移量 (作为优化的初始猜测)
    # ECC 算法对初始位置敏感，先对齐重心能极大提高成功率和速度
    src_center = get_centroid(src_cnt)
    dst_center = get_centroid(dst_cnt)
    translation = dst_center - src_center

    # 初始化仿射变换矩阵 [ [1, 0, tx], [0, 1, ty] ]
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    warp_matrix[0, 2] = translation[0]
    warp_matrix[1, 2] = translation[1]

    # 2. 光栅化：将轮廓转换为二值图像 (Mask)
    # 必须使用 uint8 单通道图像
    mask_src = np.zeros(image_size.hw(), dtype=np.uint8)
    mask_dst = np.zeros(image_size.hw(), dtype=np.uint8)

    # thickness=-1 表示填充轮廓内部
    cv2.drawContours(mask_src, [src_cnt], -1, 255, thickness=-1)
    cv2.drawContours(mask_dst, [dst_cnt], -1, 255, thickness=-1)

    # 3. 使用 ECC 算法寻找最佳变换矩阵
    # MOTION_AFFINE: 支持平移、旋转、缩放、剪切
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)
    # mask_src 是 template, mask_dst 是 input?
    # 注意：findTransformECC 寻找的是将 input (src) 映射到 template (dst) 的矩阵
    # 如果 mask 有很多噪点，可以先做一次高斯模糊，但这里轮廓画出来的图很干净，不需要
    cc, warp_matrix = cv2.findTransformECC(
        mask_src,
        mask_dst,
        warp_matrix,
        cv2.MOTION_EUCLIDEAN,
        criteria,
        inputMask=None,  # 可选：关注区域掩码 # type: ignore
        gaussFiltSize=1,  # 高斯平滑尺寸，1表示不平滑
    )  # type: ignore

    # 4. 将计算出的矩阵应用到源轮廓点上
    aligned_cnt = cv2.transform(src_cnt, warp_matrix)

    return aligned_cnt, warp_matrix


class ECCAlignedDataset(DetectionDataset):
    def __init__(self, base_dataset: DetectionDataset, meta_resize: ImageResize = 518):
        self._base_dataset = base_dataset

        name = base_dataset.get_name() + "(ecc)"
        self._meta_save_dir = Path("data_cache/images") / name
        self._contours_dir = Path("data_cache/contours") / name

        super().__init__(name, self.generate_meta_info(meta_resize))

    def get_contours_path(self, category: str, resize: ImageResize | None) -> Path:
        return (
            self._contours_dir
            / (str(resize) if resize else "default")
            / (category + ".h5")
        )

    def cache_contours(self, category: str, resize: ImageResize | None):
        h5_path = self.get_contours_path(category, resize)
        if h5_path.exists():
            print(f"Contours for category {category} already cached at {h5_path}")
            return
        h5_path.parent.mkdir(parents=True, exist_ok=True)
        tensor_dataset = self._base_dataset.get_tensor(
            category, Transform(resize=resize)
        )
        with h5writer(h5_path) as h5f:
            contour_ranges = []
            contours = []
            start = 0
            for idx, sample in tqdm(
                enumerate(tensor_dataset),
                desc=f"Caching contours for category {category}",
            ):
                mask = get_image_product_mask(sample.image)
                contour = get_contour_by_mask(mask)
                vis = to_cv2_image(sample.image).copy()
                cv2.drawContours(vis, [contour], -1, (0, 0, 255), 2)
                cv2.imwrite(f"results/contours/{idx}.png", vis)
                contour_ranges.append(np.array([start, start + len(contour)]))
                contours.append(contour)
                start += len(contour)
            h5f.create_dataset("contours", data=np.concatenate(contours, axis=0))
            h5f.create_dataset("contour_ranges", data=np.stack(contour_ranges))
        print(f"Cached contours for category {category} at {h5_path}")

    def get_contours(
        self, category: str, resize: ImageResize | None
    ) -> Dataset[Contour]:
        h5_path = self.get_contours_path(category, resize)
        assert (
            h5_path.exists()
        ), f"Contours for category {category} not cached yet at {h5_path}"
        length = len(self._base_dataset.get_labels(category))

        def get_item(idx: int) -> Contour:
            with h5py.File(h5_path, "r") as h5f:
                range = list(h5f["contour_ranges"][idx])  # type: ignore
                contour = h5f["contours"][range[0] : range[1]]  # type: ignore
                return contour  # type: ignore

        return DatasetInline(length, get_item)

    def get_tensor(self, category: str, transform: Transform):
        meta_dataset = self.generate_meta_info(transform.resize).category_datas[
            category
        ]
        tensor_dataset = self._base_dataset.get_tensor(
            category, Transform(resize=transform.resize)
        )
        contours = self.get_contours(category, transform.resize)

        first_contour = contours[0]
        first_rect = get_image_min_area_rect(first_contour)
        trans1, trans2 = get_align_transform_matrix(first_rect, None)
        first_contour = cv2.transform(first_contour, trans1)
        first_contour = cv2.transform(first_contour, trans2)

        length = len(meta_dataset)

        def get_item(idx: int) -> TensorSample:
            sample = tensor_dataset[idx]
            contour = contours[idx]
            if idx == 0:
                sample.image = align_image(sample.image, first_rect)
                sample.mask = align_image(sample.mask, first_rect)
            else:
                image_size = ImageSize.fromtensor(sample.image)
                contour_aligned, warp_matrix = align_contours_fine(
                    contour, first_contour, image_size
                )
                image_cv2 = to_cv2_image(sample.image)
                image_cv2 = cv2.warpAffine(
                    image_cv2,
                    warp_matrix,
                    dsize=image_size.hw(),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                sample.image = normalize_image(from_cv2_image(image_cv2))
                sample.image = transform.image_transform(sample.image)
                mask_cv2 = to_cv2_image(sample.mask)
                mask_cv2 = cv2.warpAffine(
                    mask_cv2,
                    warp_matrix,
                    dsize=image_size.hw(),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                sample.mask = binarize_image(from_cv2_image(mask_cv2))
                sample.mask = transform.mask_transform(sample.mask)
            return sample

        return DatasetInline(length, get_item)

    def generate_meta_info(self, resize: ImageResize | None = None) -> MetaInfo:
        save_dir = self._meta_save_dir / ("default" if resize is None else str(resize))
        category_datas: dict[str, Dataset[MetaSample]] = {
            category: DatasetOverrideGetItem(
                meta, lambda idx: self.get_aligned_meta(meta[idx], save_dir)
            )
            for category, meta in self._base_dataset.get_meta_info().category_datas.items()
        }
        meta_info = MetaInfo(data_dir=save_dir, category_datas=category_datas)
        return meta_info

    def get_aligned_meta(self, x: MetaSample, save_dir: Path) -> MetaSample:
        image_path = save_dir / Path(x.image_path).resolve().relative_to(
            self._base_dataset.get_data_dir().resolve()
        ).as_posix().replace("/", "_")
        if x.label:
            mask_path = image_path.with_name(image_path.stem + "_mask.png")
        else:
            mask_path = None
        return MetaSample(
            image_path=image_path.as_posix(),
            mask_path=mask_path.as_posix() if mask_path is not None else None,
            label=x.label,
        )


if __name__ == "__main__":
    if False:

        src_path = Path(
            "/mnt/ssd/home/zhaozy/hdd/Real-IAD/realiad_1024/audiojack/NG/BX/S0003/audiojack__0003_NG_BX_C2_20231023111640.jpg"
        )
        src_image = normalize_image(generate_image(src_path))
        src_contour = get_contour_by_mask(get_image_product_mask(src_image))
        dst_path = Path(
            "/mnt/ssd/home/zhaozy/hdd/Real-IAD/realiad_1024/audiojack/NG/BX/S0001/audiojack__0001_NG_BX_C2_20231023111538.jpg"
        )
        dst_image = normalize_image(generate_image(dst_path))
        dst_contour = get_contour_by_mask(get_image_product_mask(dst_image))
        aligned_contour, warp_matrix = align_contours_fine(
            src_contour, dst_contour, ImageSize.fromtensor(src_image)
        )
        src_image_np = to_cv2_image(src_image)
        aligned_src_image = cv2.warpAffine(
            src_image_np,
            warp_matrix,
            dsize=ImageSize.fromtensor(src_image).hw(),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        cv2.imwrite(
            "results/aligned_example.png",
            aligned_src_image,
        )
        exit(0)

    categories = [
        "audiojack_C1",
        "audiojack_C2",
        "audiojack_C3",
        "audiojack_C4",
        "audiojack_C5",
    ]
    dataset = RealIADDevidedByAngle().filter_categories(categories)
    dataset = ECCAlignedDataset(dataset, meta_resize=518)
    for category in categories:
        dataset.cache_contours(category, resize=518)
