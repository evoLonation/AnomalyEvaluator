from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import binarize
import torch
from torch import Tensor, float32, tensor
from jaxtyping import Float, Bool
import cv2
import numpy as np
from tqdm import tqdm
import torchvision.utils as vutils
from torchvision.transforms import CenterCrop

from data.base import (
    Dataset,
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
    normalize_image,
    resize_to_size,
    to_cv2_image,
)
from torch.utils.data import DataLoader

from align.base import Contour, get_image_product_mask, get_contour_by_mask


@dataclass
class Rect:
    """
    angle: 顺时针旋转角度， (-45, 45]
    size: (width, height)
    """

    center: tuple[float, float]
    size: tuple[float, float]  # width, height
    angle: float


def get_image_min_area_rect(contour: Contour) -> Rect:
    """
    获取图像中产品的最小外接矩形
    """
    # 可视化轮廓
    # vis = pil_image.convert("RGB")
    # vis = np.array(vis)
    # cv2.drawContours(vis, contours, -1, 255, thickness=10)  # 红色轮廓
    # cv2.imwrite("results/contours_overlay.png", vis)
    # 获取最小外接矩形 (中心点, (宽, 高), 旋转角度)
    rect = cv2.minAreaRect(contour)
    center, (width, height), angle = rect
    # 4.5版本定义为，x轴顺时针旋转最先重合的边为w，angle为x轴顺时针旋转的角度，angle取值为(0,90]
    # https://zhuanlan.zhihu.com/p/491547614
    # 如果角度接近 90，说明只需要很小的角度就能摆正，同时宽高交换
    if angle > 45:
        angle = angle - 90
        width, height = height, width
    # print(f"Min area rect: center={center}, size=({width}, {height}), angle={angle}")
    return Rect(
        center=center,  # type: ignore
        size=(width, height),
        angle=angle,
    )


def get_align_transform_matrix(
    rect: Rect,
    target_size: ImageSize | None = None,
):
    """
    计算对齐变换矩阵
    target_size: 如果指定，填充到该大小，否则不进行缩放
    """
    angle = rect.angle
    rect_w, rect_h = rect.size

    # 现在角度在 [-45, 45) 范围内，这就是最小旋转角度
    # 计算旋转矩阵 (逆时针为正方向)
    rotation_matrix = cv2.getRotationMatrix2D(rect.center, angle, 1.0)

    # 计算缩放比例使产品填满图片（保持宽高比）
    if target_size is None:
        scale = 1.0
    else:
        scale = min(target_size.w / rect_w, target_size.h / rect_h)
    # 计算平移量使产品居中
    tx = target_size.w / 2 - rect.center[0] * scale if target_size else 0
    ty = target_size.h / 2 - rect.center[1] * scale if target_size else 0
    # print(f"Align transform: scale={scale}, tx={tx}, ty={ty}")
    # 计算缩放平移矩阵
    transform_matrix = np.array([[scale, 0, tx], [0, scale, ty]], dtype=np.float32)
    return rotation_matrix, transform_matrix


def align_image(
    image: Float[Tensor, "C H W"] | Bool[Tensor, "H W"],
    rect: Rect,
    target_size: ImageSize | None = None,
) -> Float[Tensor, "C H W"] | Bool[Tensor, "H W"]:
    """
    根据最小外接矩形对图像进行对齐变换, 让其水平放置并居中，指定大小
    target_size: 如果指定，填充到该大小，否则不进行缩放
    """
    image_size = ImageSize.fromtensor(image)

    rotation_matrix, transform_matrix = get_align_transform_matrix(rect, target_size)

    target_device = image.device
    is_mask = image.dtype == torch.bool

    img_np = to_cv2_image(image)
    img_np = cv2.warpAffine(
        img_np,
        rotation_matrix,
        image_size.hw(),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    img_np = cv2.warpAffine(
        img_np,
        transform_matrix,
        image_size.hw(),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # 转换回tensor格式
    result_image = from_cv2_image(img_np)
    result_image = result_image.to(target_device)
    if is_mask:
        result_image = binarize_image(result_image)
    else:
        result_image = normalize_image(result_image)
    return result_image


class AlignedDataset(DetectionDataset):
    """
    has_meta return False, but actually save the image to save_dir to cache when iter TensorDataset
    """

    def __init__(
        self, base_dataset: DetectionDataset, save_dir: Path = Path("data_cache/images")
    ):
        self.base_dataset = base_dataset
        dataset_name = base_dataset.get_name() + "(aligned)"
        self.save_dir = save_dir / dataset_name
        category_datas = self.base_dataset.get_meta_info().category_datas

        category_datas_: dict[str, Dataset[MetaSample]] = {
            c: ListDataset([self.get_aligned_meta(x, 518) for x in ds])
            for c, ds in category_datas.items()
        }
        meta_info = MetaInfo(
            data_dir=self.save_dir,
            category_datas=category_datas_,
        )
        self.rects_cache: dict[str, dict[int, Rect]] = {}
        self.rects_last_len: dict[str, int] = {}

        super().__init__(dataset_name, meta_info)

    def get_rects(self, category: str) -> dict[int, Rect]:
        csv_path = self.save_dir / "rects" / f"{category}.csv"
        if not csv_path.exists():
            return {}
        df = pd.read_csv(csv_path)
        indices = df["index"].tolist()
        centers_x = df["center_x"].tolist()
        centers_y = df["center_y"].tolist()
        sizes_w = df["size_w"].tolist()
        sizes_h = df["size_h"].tolist()
        angles = df["angle"].tolist()
        rects = [
            Rect(center=(cx, cy), size=(sw, sh), angle=ang)
            for cx, cy, sw, sh, ang in zip(
                centers_x, centers_y, sizes_w, sizes_h, angles
            )
        ]
        return dict(zip(indices, rects))

    def save_rects(self, category: str, rects: dict[int, Rect]):
        csv_path = self.save_dir / "rects"
        csv_path.mkdir(parents=True, exist_ok=True)
        csv_file = csv_path / f"{category}.csv"
        data = {
            "index": [],
            "center_x": [],
            "center_y": [],
            "size_w": [],
            "size_h": [],
            "angle": [],
        }
        for index, rect in sorted(rects.items(), key=lambda x: x[0]):
            data["index"].append(index)
            data["center_x"].append(rect.center[0])
            data["center_y"].append(rect.center[1])
            data["size_w"].append(rect.size[0])
            data["size_h"].append(rect.size[1])
            data["angle"].append(rect.angle)
        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False)

    def save_all_cached_rects(self):
        for category, rects in self.rects_cache.items():
            self.save_rects(category, rects)

    def get_aligned_meta(self, x: MetaSample, resize: ImageResize | None) -> MetaSample:
        save_dir = self.save_dir / ("default" if resize is None else str(resize))
        image_path = save_dir / Path(x.image_path).resolve().relative_to(
            self.base_dataset.get_data_dir().resolve()
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

    def get_tensor(self, category: str, transform: Transform) -> Dataset[TensorSample]:
        if category not in self.rects_cache:
            self.rects_cache[category] = self.get_rects(category)
            self.rects_last_len[category] = len(self.rects_cache[category])
        tensor_dataset = self.base_dataset.get_tensor(category, Transform())
        origin_getitem = tensor_dataset.__getitem__

        # todo: getitem中最好不要有共享操作，可能会有多进程问题
        def getitem_override(index: int) -> TensorSample:
            meta_sample = self.base_dataset.get_meta(category)[index]
            meta_sample = self.get_aligned_meta(meta_sample, transform.resize)
            image_path = Path(meta_sample.image_path)
            mask_path = Path(meta_sample.mask_path) if meta_sample.mask_path else None
            if not image_path.exists() or (
                mask_path is not None and not mask_path.exists()
            ):
                image_path.parent.mkdir(parents=True, exist_ok=True)
                if mask_path is not None:
                    mask_path.parent.mkdir(parents=True, exist_ok=True)
                # print(f"Aligning image {index}...", end="\r")
                sample = origin_getitem(index)
                # 获取对齐变换函数
                image_size = ImageSize.fromtensor(sample.image)
                target_size = (
                    resize_to_size(image_size, transform.resize)
                    if transform.resize is not None
                    else image_size
                )
                if index in self.rects_cache[category]:
                    rect = self.rects_cache[category][index]
                else:
                    assert False
                    rect = get_image_min_area_rect(sample.image)
                    self.rects_cache[category][index] = rect
                    now_len = len(self.rects_cache[category])
                    if now_len - self.rects_last_len[category] >= 10:
                        self.save_rects(category, self.rects_cache[category])
                        self.rects_last_len[category] = now_len
                # 对图像和mask应用相同的变换
                image = align_image(sample.image, rect, target_size)
                mask = align_image(sample.mask, rect, target_size)
                if transform.resize is not None:
                    # centercrop
                    center_crop = CenterCrop(target_size.hw())
                    image = center_crop(image)
                    mask = center_crop(mask)
                vutils.save_image(
                    image, image_path.as_posix(), normalize=True, scale_each=True
                )
                if mask_path is not None:
                    vutils.save_image(
                        mask.to(float32),
                        mask_path.as_posix(),
                        normalize=True,
                        scale_each=True,
                    )
            else:
                # print(f"Loading aligned image {index} from cache...", end="\r")
                image = tensor(
                    normalize_image(generate_image(image_path, transform.resize))
                )
                if mask_path is not None:
                    mask = tensor(generate_mask(mask_path, transform.resize))
                else:
                    mask = tensor(generate_empty_mask(ImageSize.fromtensor(image)))

            # 应用额外的transform
            image = transform.image_transform(image)
            mask = transform.mask_transform(mask)
            return TensorSample(image=image, mask=mask, label=meta_sample.label)

        tensor_dataset = DatasetOverrideGetItem(tensor_dataset, getitem_override)
        return tensor_dataset


if __name__ == "__main__":
    if True:
        image_path = Path(
            "/mnt/ssd/home/zhaozy/hdd/Real-IAD/realiad_1024/audiojack/NG/HS/S0064/audiojack__0064_NG_HS_C1_20231023092610.jpg"
        )
        image = tensor(normalize_image(generate_image(image_path)))
        contour = get_contour_by_mask(get_image_product_mask(image))
        rect = get_image_min_area_rect(contour)
        aligned_image = align_image(image, rect, ImageSize.square(518))
        import torchvision.utils as vutils

        vutils.save_image(
            aligned_image,
            "results/aligned_example.png",
            normalize=True,
            scale_each=True,
        )
        exit(0)

    categories = [
        # "audiojack_C1",
        # "audiojack_C2",
        # "audiojack_C3",
        "audiojack_C4",
        "audiojack_C5",
    ]
    dataset = RealIADDevidedByAngle().filter_categories(categories)
    aligned_dataset = AlignedDataset(dataset)
    generate_summary_view(aligned_dataset)
    exit(0)

    aligned_dataset.set_transform(Transform(resize=518))

    for category in categories:
        print(f"Processing category {category}...")
        dataloader = DataLoader(
            ZipedDataset(dataset.get_meta(category), aligned_dataset[category]),
            batch_size=4,
            num_workers=0,
            shuffle=False,
            collate_fn=lambda x: tuple_collate_fn(
                x, lambda a: a, TensorSample.collate_fn
            ),
        )
        # save_dir = Path("results/aligned")
        # save_dir.mkdir(parents=True, exist_ok=True)
        # total_num = 0
        for i, (meta_batch, tensor_batch) in tqdm(enumerate(dataloader)):
            i: int
            # total_num += len(meta_batch)
            # if total_num % 40 == 0:
            #     print(f"Processed {total_num} images...")
            # import torchvision.utils as vutils

            # rel_path = (
            #     Path(meta_batch[0].image_path)
            #     .resolve()
            #     .relative_to(dataset.get_data_dir().resolve())
            # )
            # image_path = save_dir / rel_path.as_posix().replace("/", "_")
            # vutils.save_image(
            #     tensor_batch.images[0],
            #     image_path.as_posix(),
            #     normalize=True,
            #     scale_each=True,
            # )
            # if meta_batch[0].label:
            #     mask_path = image_path.with_name(image_path.stem + "_mask.png")
            #     vutils.save_image(
            #         tensor_batch.masks[0],
            #         mask_path.as_posix(),
            #         normalize=True,
            #         scale_each=True,
            #     )
    aligned_dataset.save_all_cached_rects()
