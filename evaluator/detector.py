from abc import abstractmethod
from dataclasses import dataclass
import numpy as np
from jaxtyping import Float, Bool


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

# 会将一个 batch 的所有图像联合进行检测
class BatchJointDetector:
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def __call__(self, image_paths: list[str], class_name: str) -> DetectionResult:
        pass