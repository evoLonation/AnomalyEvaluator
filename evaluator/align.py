import os
from pathlib import Path
import torch
from torch import Tensor, float32, isin, tensor
from jaxtyping import Float, Shaped
import cv2
import numpy as np
from transformers import SamModel, SamProcessor
from PIL import Image
from typing import Callable
import torchvision.utils as vutils
from torchvision.transforms import CenterCrop

from data.base import Dataset, DatasetOverrideGetItem, ZipedDataset, tuple_collate_fn
from data.cached_impl import RealIAD, RealIADDevidedByAngle
from data.detection_dataset import (
    DetectionDataset,
    DetectionDatasetByFactory,
    MetaInfo,
    MetaSample,
    TensorSample,
    TensorSampleBatch,
)
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


def align_image(
    image: Float[Tensor, "C H W"], target_size: ImageSize | None = None
) -> Callable[[Float[Tensor, "C H W"]], Float[Tensor, "C H W"]]:
    """
    生成对齐变换函数，用于对齐图像中的产品

    步骤：
    1. 使用SAM分割得到产品的最大轮廓
    2. 获取最小外接矩形
    3. 计算旋转角度和缩放平移参数
    4. 返回变换函数，可应用于图像和mask

    Args:
        image: 输入图像 tensor (C, H, W)

    Returns:
        变换函数，接受tensor (C, H, W) 返回对齐后的tensor (C, H, W)
    """
    device = image.device

    H, W = image.shape[1:]
    # 转换为numpy格式 (H, W, C)，范围[0, 255]
    img_np = image.cpu().permute(1, 2, 0).numpy()
    img_np = (img_np * 255).astype(np.uint8)
    # 转换为PIL Image
    pil_image = Image.fromarray(img_np)

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
    if len(contours) == 0:
        return lambda x: x

    # 获取最大轮廓
    largest_contour = max(contours, key=cv2.contourArea)

    # 获取最小外接矩形 (中心点, (宽, 高), 旋转角度)
    # 4.5版本定义为，x轴顺时针旋转最先重合的边为w，angle为x轴顺时针旋转的角度，angle取值为(0,90]
    rect = cv2.minAreaRect(largest_contour)
    center, (width, height), angle = rect
    print(f"Min area rect: center={center}, size=({width}, {height}), angle={angle}")

    # 如果角度接近 90，说明只需要很小的角度就能摆正
    if angle > 45:
        angle = angle - 90

    # 现在角度在 [-45, 45) 范围内，这就是最小旋转角度

    # 计算旋转矩阵 (逆时针为正方向)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 旋转mask以获取新的边界
    rotated_mask = cv2.warpAffine(
        mask,
        rotation_matrix,
        (W, H),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # 获取旋转后物体的边界框
    contours_rotated, _ = cv2.findContours(
        rotated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours_rotated) == 0:
        return lambda x: x

    x, y, w, h = cv2.boundingRect(max(contours_rotated, key=cv2.contourArea))

    # 计算缩放比例使产品填满图片（保持宽高比）
    if target_size is None:
        scale = min(W / w, H / h)
    else:
        scale = min(target_size.w / w, target_size.h / h)

    # 计算平移量使产品居中
    new_center_x = x + w / 2
    new_center_y = y + h / 2
    tx = W / 2 - new_center_x * scale
    ty = H / 2 - new_center_y * scale
    print(f"Align transform: scale={scale}, tx={tx}, ty={ty}")

    # 计算缩放平移矩阵
    transform_matrix = np.array([[scale, 0, tx], [0, scale, ty]], dtype=np.float32)

    # 创建并返回变换函数
    def apply_transform(
        input_tensor: Shaped[Tensor, "*C H W"],
    ) -> Shaped[Tensor, "*C H W"]:
        """应用对齐变换到输入tensor"""
        if len(input_tensor.shape) != 3:
            input_tensor.unsqueeze_(0)

        target_device = input_tensor.device
        target_dtype = input_tensor.dtype
        is_normalized = True

        # 转换为numpy格式
        input_np = input_tensor.cpu().permute(1, 2, 0).numpy()
        if is_normalized:
            input_np = (input_np * 255).astype(np.uint8)
        else:
            input_np = input_np.astype(np.uint8)

        # 应用旋转
        rotated = cv2.warpAffine(
            input_np,
            rotation_matrix,
            (W, H),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        # 应用缩放和平移
        transformed = cv2.warpAffine(
            rotated,
            transform_matrix,
            (W, H),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        # 转换回tensor格式
        result_tensor = torch.from_numpy(transformed).to(
            device=target_device, dtype=target_dtype
        )
        if len(result_tensor.shape) == 3:
            result_tensor = result_tensor.permute(2, 0, 1)

        # 如果原图在[0,1]范围，则归一化回去
        if is_normalized and result_tensor.dtype == float32:
            result_tensor = result_tensor / 255.0

        return result_tensor

    return apply_transform


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

        category_datas = {
            c: [self.get_aligned_meta(x, 518) for x in ds]
            for c, ds in category_datas.items()
        }
        meta_info = MetaInfo(
            data_dir=self.save_dir,
            category_datas=category_datas,
        )

        super().__init__(dataset_name, meta_info)

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
        tensor_dataset = self.base_dataset.get_tensor(category, Transform())
        origin_getitem = tensor_dataset.__getitem__

        def getitem_override(index: int) -> TensorSample:
            meta_sample = self.base_dataset.get_meta(category)[index]
            meta_sample = self.get_aligned_meta(meta_sample, transform.resize)
            image_path = Path(meta_sample.image_path)
            mask_path = Path(meta_sample.mask_path) if meta_sample.mask_path else None
            if not image_path.exists() or (
                mask_path is not None and not mask_path.exists()
            ):
                # print(f"Aligning image {index}...", end="\r")
                sample = origin_getitem(index)
                # 获取对齐变换函数
                image_size = ImageSize.fromtensor(sample.image.shape)
                target_size = (
                    resize_to_size(image_size, transform.resize)
                    if transform.resize is not None
                    else image_size
                )
                align_transform = align_image(sample.image, target_size)
                # 对图像和mask应用相同的变换
                image = align_transform(sample.image)
                mask = align_transform(sample.mask)
                if transform.resize is not None:
                    # centercrop
                    center_crop = CenterCrop(target_size.hw())
                    image = center_crop(image)
                    mask = center_crop(mask)
                    # image = tensor(
                    #     normalize_image(
                    #         resize_image(
                    #             denormalize_image(image.cpu().numpy()), transform.resize
                    #         )
                    #     )
                    # )
                    # mask = tensor(resize_mask(mask.cpu().numpy(), transform.resize))
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
    if True:
        image_path = Path(
            "/mnt/ssd/home/zhaozy/hdd/Real-IAD/realiad_1024/audiojack/NG/HS/S0064/audiojack__0064_NG_HS_C1_20231023092610.jpg"
        )
        image = tensor(normalize_image(generate_image(image_path)))
        align_fn = align_image(image, ImageSize.square(518))
        aligned_image = align_fn(image)
        import torchvision.utils as vutils

        vutils.save_image(
            aligned_image,
            "results/aligned_example.png",
            normalize=True,
            scale_each=True,
        )
        exit(0)

    dataset = RealIADDevidedByAngle()
    aligned_dataset = AlignedDataset(dataset)
    aligned_dataset.set_transform(Transform(resize=518))

    categories = [
        "audiojack_C1",
        "audiojack_C2",
        "audiojack_C3",
        "audiojack_C4",
        "audiojack_C5",
    ]
    for category in categories:
        print(f"Processing category {category}...")
        dataloader = DataLoader(
            ZipedDataset(dataset.get_meta(category), aligned_dataset[category]),
            batch_size=4,
            num_workers=4,
            shuffle=False,
            collate_fn=lambda x: tuple_collate_fn(
                x, lambda a: a, TensorSample.collate_fn
            ),
        )
        # save_dir = Path("results/aligned")
        # save_dir.mkdir(parents=True, exist_ok=True)
        total_num = 0
        for i, (meta_batch, tensor_batch) in enumerate(dataloader):
            i: int
            total_num += len(meta_batch)
            if total_num % 40 == 0:
                print(f"Processed {total_num} images...")
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
