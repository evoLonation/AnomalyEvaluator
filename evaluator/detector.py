from abc import abstractmethod
from dataclasses import dataclass
import numpy as np
from jaxtyping import Float, Bool
import torch

from data.utils import ImageResize, ImageTransform, MaskTransform


@dataclass
class DetectionResult:
    pred_scores: Float[torch.Tensor, "N"]
    anomaly_maps: Float[torch.Tensor, "N H W"]


@dataclass
class DetectionGroundTruth:
    true_labels: Bool[torch.Tensor, "N"]
    true_masks: Bool[torch.Tensor, "N H W"]


class Detector:
    """
    mask_transform: 用于对原始掩码进行预处理的函数, 用于将 groundtruth 与 检测结果对齐
    """

    def __init__(self, name: str, mask_transform: MaskTransform | None):
        self.name = name
        self.mask_transform = mask_transform

    @abstractmethod
    def __call__(self, image_paths: list[str], class_name: str) -> DetectionResult:
        pass


class TensorDetector:
    """
    resize: 指定图像/掩码从原始分辨率调整到的大小(如果为 int，则为最短边长度)
    image_tranform: 用于对调整后的图像进行预处理的函数, 用于__call__前
    mask_transform: 用于对调整后的掩码进行预处理的函数, 用于将 groundtruth 与 检测结果对齐
    """

    def __init__(
        self,
        name: str,
        resize: ImageResize | None,
        image_transform: ImageTransform | None,
        mask_transform: MaskTransform | None,
    ):
        self.name = name
        self.resize = resize
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    @abstractmethod
    def __call__(
        self, images: Float[torch.Tensor, "N C H W"], class_name: str
    ) -> DetectionResult:
        pass
