from pathlib import Path
from typing import override
from my_ipc.ipc_client import IPCClient
from my_ipc.public import ShmArrayInfo
from detector import BatchJointDetector, DetectionResult
import numpy as np


class MuSc(BatchJointDetector, IPCClient):
    def __init__(
        self,
        working_dir: Path = Path("~/MuSc").expanduser(),
    ):
        BatchJointDetector.__init__(self, f"MuSc")

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
            pred_scores=np.array(anomaly_scores), anomaly_maps=anomaly_mask
        )
