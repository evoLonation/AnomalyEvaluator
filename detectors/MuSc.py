from pathlib import Path
from typing import override

import torch
from data.utils import ImageSize
from my_ipc.ipc_client import IPCClient
from my_ipc.public import ShmArrayInfo
from jaxtyping import Float
import numpy as np
from torchvision.transforms import CenterCrop, Compose, Normalize

from evaluator.detector import DetectionResult, Detector, TensorDetector


class MuSc(Detector, IPCClient):
    def __init__(
        self,
        working_dir: Path = Path("~/MuSc").expanduser(),
    ):
        Detector.__init__(self, f"MuSc", mask_transform=CenterCrop(518))
        server_cmd = f"""
        cd {working_dir} && \
        source .venv/bin/activate && \
        python anomaly_detection.py \
            --id {{id}}
        """
        IPCClient.__init__(
            self,
            server_cmd=server_cmd,
        )

    @override
    def __call__(self, image_paths: list[str], class_name: str) -> DetectionResult:
        response, anomaly_mask = self.send_request(
            {"image_paths": image_paths},
            tmp_shm=ShmArrayInfo(
                shape=(len(image_paths), 518, 518),
                dtype=np.float32,
            ),
        )
        anomaly_scores = response["anomaly_scores"]

        return DetectionResult(
            pred_scores=torch.tensor(anomaly_scores),
            anomaly_maps=torch.tensor(anomaly_mask),
        )


class MuScTensor(TensorDetector, IPCClient):
    def __init__(
        self,
        max_batch_size: int = 64,
        working_dir: Path = Path("~/MuSc").expanduser(),
    ):
        image_transform = Compose(
            [
                CenterCrop(518),
                Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        TensorDetector.__init__(
            self,
            f"MuScTensor",
            518,
            image_transform=image_transform,
            mask_transform=CenterCrop(518),
        )
        server_cmd = f"""
        cd {working_dir} && \
        source .venv/bin/activate && \
        python anomaly_detection.py \
            --id {{id}}
        """
        IPCClient.__init__(
            self,
            server_cmd=server_cmd,
            shm_arrs={
                "images": ShmArrayInfo(
                    shape=(max_batch_size, 3, 518, 518),
                    dtype=np.float32,
                ),
            },
        )
        self.max_batch_size = max_batch_size

    @override
    def __call__(
        self, images: Float[torch.Tensor, "N C H W"], class_name: str
    ) -> DetectionResult:
        images_shm = self.get_shared_array("images")
        images_np = np.concatenate(
            [
                images.cpu().numpy(),
                np.zeros(
                    (self.max_batch_size - len(images), 3, 518, 518),
                    dtype=np.float32,
                ),
            ]
        )
        images_shm.write(images_np)

        response, anomaly_mask = self.send_request(
            {"image_tensors_num": len(images)},
            tmp_shm=ShmArrayInfo(
                shape=(len(images), 518, 518),
                dtype=np.float32,
            ),
        )
        anomaly_scores = response["anomaly_scores"]

        return DetectionResult(
            pred_scores=torch.tensor(anomaly_scores),
            anomaly_maps=torch.tensor(anomaly_mask),
        )
