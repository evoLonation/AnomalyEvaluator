from pathlib import Path
from typing import override
from my_ipc.ipc_client import IPCClient
from my_ipc.public import ShmArrayInfo
from detector import Detector, DetectionResult
import numpy as np


class AACLIP(Detector, IPCClient):
    def __init__(
        self,
        batch_size: int,
        working_dir: Path = Path("~/AA-CLIP").expanduser(),
    ):
        Detector.__init__(self, f"AA-CLIP")

        server_cmd = f"""
        cd {working_dir} && \
        source .venv/bin/activate && \
        python anomaly_detection.py \
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
        response = self.send_request(
            {"image_paths": image_paths, "class_name": class_name}
        )

        anomaly_masks = self.read_shared_array()
        if self.batch_size != len(image_paths):
            anomaly_masks = anomaly_masks[: len(image_paths)]
        anomaly_scores = response["anomaly_scores"]
        return DetectionResult(
            pred_scores=np.array(anomaly_scores), anomaly_maps=np.array(anomaly_masks)
        )
