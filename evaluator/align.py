from dataclasses import dataclass
import fcntl
import os
from pathlib import Path
import pandas as pd
import torch
from torch import Tensor, float32, isin, tensor
from jaxtyping import Float, Shaped, Bool
import cv2
import numpy as np
from tqdm import tqdm
from transformers import SamModel, SamProcessor
from PIL import Image
from typing import Callable
import torchvision.utils as vutils
from torchvision.transforms import CenterCrop

from data.base import Dataset, DatasetOverrideGetItem, ListDataset, ZipedDataset, tuple_collate_fn
from data.cached_impl import RealIAD, RealIADDevidedByAngle
from data.detection_dataset import (
    DetectionDataset,
    DetectionDatasetByFactory,
    MetaInfo,
    MetaSample,
    TensorSample,
    TensorSampleBatch,
)
from data.summary import generate_summary_view
from data.utils import (
    ImageResize,
    ImageSize,
    Transform,
    denormalize_image,
    generate_empty_mask,
    generate_image,
    generate_mask,
    normalize_image,
    resize_image,
    resize_mask,
    resize_to_size,
    to_pil_image,
)
from torch.utils.data import DataLoader


# 全局SAM模型实例，避免重复加载
_SAM_MODEL = None
_SAM_PROCESSOR = None


def _get_sam_model_and_processor():
    """获取或初始化SAM模型和处理器"""
    global _SAM_MODEL, _SAM_PROCESSOR

    if _SAM_MODEL is None or _SAM_PROCESSOR is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _SAM_MODEL = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)  # type: ignore
        _SAM_PROCESSOR = SamProcessor.from_pretrained("facebook/sam-vit-huge")
        _SAM_MODEL.eval()

    return _SAM_MODEL, _SAM_PROCESSOR


def _generate_grid_points(H: int, W: int, grid_size: int = 32) -> np.ndarray:
    """生成网格采样点用于自动分割"""
    points = []
    step_h = H // grid_size
    step_w = W // grid_size
    for i in range(grid_size):
        for j in range(grid_size):
            y = int((i + 0.5) * step_h)
            x = int((j + 0.5) * step_w)
            points.append([x, y])
    return np.array(points)


@dataclass
class Rect:
    """
    angle: 顺时针旋转角度， (-45, 45]
    size: (width, height)
    """

    center: tuple[float, float]
    size: tuple[float, float]  # width, height
    angle: float


def get_image_min_area_rect(
    image: Float[Tensor, "C H W"],
) -> Rect:
    """
    获取图像中产品的最小外接矩形
    """
    H, W = image.shape[1:]
    # 转换为PIL Image
    pil_image = to_pil_image(image.cpu().numpy())

    # 获取SAM模型和处理器
    model, processor = _get_sam_model_and_processor()

    each_point_as_predict = True

    # 生成网格采样点进行自动分割
    input_points = [[W // 2, H // 2]]
    for x in [0.4]:
        input_points.append([int(W * x), int(H * x)])
        input_points.append([int(W * x), int(H * (1 - x))])
        input_points.append([int(W * (1 - x)), int(H * x)])
        input_points.append([int(W * (1 - x)), int(H * (1 - x))])
    # input_points = [[[W // 2, H // 2], [W // 5, H // 5], [4 * W // 5, 4 * H // 5], [W // 5, 4 * H // 5], [4 * W // 5, H // 5]]]
    if not each_point_as_predict:
        input_points = [[input_points]]
    else:
        # each point as a predict
        input_points = [[[x] for x in input_points]]

    inputs = processor(
        pil_image,
        input_points=input_points,
        # input_boxes=input_boxes,
        return_tensors="pt",
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        outputs = model(**inputs)

    # 后处理masks list(1 3 H W), 3个候选掩码
    masks = processor.image_processor.post_process_masks(  # type: ignore
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),  # type: ignore
        inputs["reshaped_input_sizes"].cpu(),  # type: ignore
    )

    assert isinstance(masks, list) and len(masks) == 1
    masks = masks[0]
    if not each_point_as_predict:
        assert masks.shape == (1, 3, H, W), {masks.shape}
        masks = masks[0].numpy()
        mask_areas = [np.sum(m) for m in masks]
        # cv2.imwrite("results/debug_mask_0.png", (masks[0] * 255).astype(np.uint8))
        # cv2.imwrite("results/debug_mask_1.png", (masks[1] * 255).astype(np.uint8))
        # cv2.imwrite("results/debug_mask_2.png", (masks[2] * 255).astype(np.uint8))
    else:
        assert masks.shape == ((len(input_points[0])), 3, H, W), masks[0].shape
        masks = masks[:, 2].numpy()  # 选择每组的第3个mask作为结果
        mask_areas = [np.sum(m) for m in masks]
        # cv2.imwrite("results/debug_mask_0.png", (masks[0] * 255).astype(np.uint8))
        # cv2.imwrite("results/debug_mask_1.png", (masks[1] * 255).astype(np.uint8))
        # cv2.imwrite("results/debug_mask_2.png", (masks[2] * 255).astype(np.uint8))

    largest_mask = masks[np.argmax(mask_areas)]
    # 将 bool 掩码转换为 uint8 (0 或 255) 以便 OpenCV 使用
    mask = (largest_mask * 255).astype(np.uint8)
    # print(mask.shape)
    # save mask for debug
    # cv2.imwrite("results/debug_mask.png", mask)

    # 获取轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    assert len(contours) > 0, "No contours found in the image."
    largest_contour = max(contours, key=cv2.contourArea)
    # 获取最小外接矩形 (中心点, (宽, 高), 旋转角度)
    rect = cv2.minAreaRect(largest_contour)
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


def align_image(
    image: Float[Tensor, "C H W"] | Bool[Tensor, "H W"],
    rect: Rect,
    target_size: ImageSize | None = None,
) -> Float[Tensor, "C H W"] | Bool[Tensor, "H W"]:
    """
    根据最小外接矩形对图像进行对齐变换, 让其水平放置并居中，填充到原图大小或指定大小
    """
    image_size = ImageSize.fromtensor(image.shape)
    if target_size is None:
        target_size = image_size

    angle = rect.angle
    rect_w, rect_h = rect.size

    # 现在角度在 [-45, 45) 范围内，这就是最小旋转角度
    # 计算旋转矩阵 (逆时针为正方向)
    rotation_matrix = cv2.getRotationMatrix2D(rect.center, angle, 1.0)

    # 计算缩放比例使产品填满图片（保持宽高比）
    scale = min(target_size.w / rect_w, target_size.h / rect_h)
    # 计算平移量使产品居中
    tx = image_size.w / 2 - rect.center[0] * scale
    ty = image_size.h / 2 - rect.center[1] * scale
    # print(f"Align transform: scale={scale}, tx={tx}, ty={ty}")
    # 计算缩放平移矩阵
    transform_matrix = np.array([[scale, 0, tx], [0, scale, ty]], dtype=np.float32)

    target_device = image.device
    target_dtype = image.dtype
    if len(image.shape) == 2:
        image = image.unsqueeze(0)  # HW to 1HW
    img_np = image.cpu().permute(1, 2, 0).numpy()  # CHW to HWC
    img_np = img_np.astype(np.float32)
    img_np = (img_np * 255).astype(np.uint8)

    # 应用旋转
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
    result_tensor = torch.from_numpy(img_np).to(
        device=target_device, dtype=target_dtype
    )
    if len(result_tensor.shape) == 3:
        result_tensor = result_tensor.permute(2, 0, 1)

    if result_tensor.dtype == float32:
        result_tensor = result_tensor / 255.0

    return result_tensor


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
                image_size = ImageSize.fromtensor(sample.image.shape)
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
                    mask = tensor(
                        generate_empty_mask(ImageSize.fromtensor(image.shape))
                    )

            # 应用额外的transform
            image = transform.image_transform(image)
            mask = transform.mask_transform(mask)
            return TensorSample(image=image, mask=mask, label=meta_sample.label)

        tensor_dataset = DatasetOverrideGetItem(tensor_dataset, getitem_override)
        return tensor_dataset


if __name__ == "__main__":
    if False:
        image_path = Path(
            "/mnt/ssd/home/zhaozy/hdd/Real-IAD/realiad_1024/audiojack/NG/HS/S0064/audiojack__0064_NG_HS_C1_20231023092610.jpg"
        )
        image = tensor(normalize_image(generate_image(image_path)))
        rect = get_image_min_area_rect(image)
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
