from pathlib import Path
from typing import Literal, override

import torch
from my_ipc.ipc_client import IPCClient
from my_ipc.public import ShmArrayInfo
import numpy as np
from torchvision.transforms import CenterCrop

from evaluator.detector import Detector, DetectionResult


class AdaCLIP(Detector, IPCClient):
    def __init__(
        self,
        batch_size: int,
        working_dir: Path = Path("~/AdaCLIP").expanduser(),
        type: Literal["mvtec", "visa"] = "mvtec",
    ):
        mask_transform = CenterCrop(518)
        Detector.__init__(self, f"AdaCLIP({type})", mask_transform=mask_transform)

        server_cmd = f"""
        cd {working_dir} && \
        source .venv/bin/activate && \
        python anomaly_detection.py \
            --type {type} \
            --id {{id}}
        """
        IPCClient.__init__(
            self,
            server_cmd=server_cmd,
            shm_arrs=ShmArrayInfo(
                shape=(batch_size, 518, 518),
                dtype=np.float32,
            ),
        )
        self.batch_size = batch_size

    @override
    def __call__(self, image_paths: list[str], class_name: str) -> DetectionResult:
        assert len(image_paths) <= self.batch_size
        response = self.send_request(
            {"image_paths": image_paths, "class_name": class_name}
        )

        anomaly_masks = self.get_shared_array().read()
        if self.batch_size != len(image_paths):
            anomaly_masks = anomaly_masks[: len(image_paths)]
        anomaly_scores = response["anomaly_scores"]
        return DetectionResult(
            pred_scores=torch.tensor(anomaly_scores),
            anomaly_maps=torch.tensor(anomaly_masks),
        )
