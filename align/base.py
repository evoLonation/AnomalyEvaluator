import cv2
from transformers import SamModel, SamProcessor
from torch import Tensor
import torch
import numpy as np
from jaxtyping import Float, UInt8, Int
from data.utils import ImageSize, to_numpy_image, to_pil_image

# 全局SAM模型实例，避免重复加载
_SAM_MODEL = None
_SAM_PROCESSOR = None

type Contour = Int[np.ndarray, "N 1 2"]


def _get_sam_model_and_processor():
    """获取或初始化SAM模型和处理器"""
    global _SAM_MODEL, _SAM_PROCESSOR

    if _SAM_MODEL is None or _SAM_PROCESSOR is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _SAM_MODEL = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)  # type: ignore
        _SAM_PROCESSOR = SamProcessor.from_pretrained("facebook/sam-vit-huge")
        _SAM_MODEL.eval()

    return _SAM_MODEL, _SAM_PROCESSOR


def get_image_product_mask(
    image: Float[Tensor, "C H W"],
    round_points: list[tuple[float, float]] = [
        (0.5, 0.5),
        (0.4, 0.4),
        (0.4, 0.6),
        (0.6, 0.4),
        (0.6, 0.6),
    ],
) -> UInt8[np.ndarray, "H W"]:
    H, W = ImageSize.fromtensor(image).hw()
    # 转换为PIL Image
    pil_image = to_pil_image(image)
    # 获取SAM模型和处理器
    model, processor = _get_sam_model_and_processor()
    each_point_as_predict = True
    # 生成网格采样点进行自动分割
    input_points = []
    for _x, _y in round_points:
        x, y = int(W * _x), int(H * _y)
        input_points.append([x, y])
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
    return mask


def get_contour_by_mask(mask: UInt8[np.ndarray, "H W"]) -> Contour:
    # 查找轮廓
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,  # 只检索外部轮廓
        cv2.CHAIN_APPROX_SIMPLE,  # 压缩水平、垂直和对角线段，只保留端点
    )
    assert len(contours) > 0, "No contours found in the image."
    # 选择最大轮廓
    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour
