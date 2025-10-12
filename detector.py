from abc import abstractmethod
from dataclasses import dataclass
import numpy as np
from jaxtyping import Float, Bool, jaxtyped
from typeguard import typechecked as typechecker


@jaxtyped(typechecker=typechecker)
@dataclass
class DetectionResult:
    pred_scores: Float[np.ndarray, "N"]
    anomaly_maps: Float[np.ndarray, "N H W"]


@jaxtyped(typechecker=typechecker)
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
