from pathlib import Path
from typing import Literal, override
from my_ipc.ipc_client import IPCClient
from my_ipc.public import ShmArrayInfo
import numpy as np
from torchvision.transforms import CenterCrop
import torch

from evaluator.detector import Detector, DetectionResult


class AACLIP(Detector, IPCClient):
    def __init__(
        self,
        batch_size: int,
        working_dir: Path = Path("~/AA-CLIP").expanduser(),
        type: Literal["mvtec", "visa"] = "mvtec",
        dataset: str = "MVTec",
    ):
        mask_transform = CenterCrop(518)
        Detector.__init__(self, f"AA-CLIP({type})", mask_transform=mask_transform)

        server_cmd = f"""
        cd {working_dir} && \
        source .venv/bin/activate && \
        python anomaly_detection.py \
            --type {type} \
            --dataset {dataset} \
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
        self.dataset = dataset

    @override
    def __call__(self, image_paths: list[str], class_name: str) -> DetectionResult:
        assert len(image_paths) <= self.batch_size
        # deal with devided_by_angle == True
        if self.dataset == "RealIAD":
            if any([f"_C{x}" in class_name for x in range(1, 6)]):
                class_name = class_name.split("_C")[0]
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
