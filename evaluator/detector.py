from abc import abstractmethod
from dataclasses import dataclass
import numpy as np
from jaxtyping import Float, Bool
import torch

from .data import ImageSize, TensorSampleBatch


@dataclass
class DetectionResult:
    pred_scores: Float[np.ndarray, "N"]
    anomaly_maps: Float[np.ndarray, "N H W"]


@dataclass
class DetectionGroundTruth:
    true_labels: Bool[np.ndarray, "N"]
    true_masks: Bool[np.ndarray, "N H W"]


class Detector:
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def __call__(self, image_paths: list[str], class_name: str) -> DetectionResult:
        pass


class TensorDetector:
    def __init__(self, name: str, image_size: ImageSize):
        self.name = name
        self.image_size = image_size

    @abstractmethod
    def __call__(
        self, images: Float[torch.Tensor, "N C H W"], class_name: str
    ) -> DetectionResult:
        pass
