from pathlib import Path
from typing import Literal, override
from evaluation import *
from my_ipc.ipc_client  import IPCClient
from my_ipc.public import ShmArrayInfo

class AnomalyCLIP(Detector, IPCClient):
    def __init__(
        self,
        working_dir: Path = Path("~/AnomalyCLIP").expanduser(),
        type: Literal["mvtec", "visa"] = "mvtec",
    ):
        Detector.__init__(self, f"AnomalyCLIP_{type}")

        server_cmd = f"""
        cd {working_dir} && \
        source .venv/bin/activate && \
        python anomaly_detection2.py \
            --type {type} \
            --id {{id}}
        """
        IPCClient.__init__(
            self,
            server_cmd=server_cmd,
            shm_arrs=ShmArrayInfo(
                shape=(518, 518),
                dtype=np.float32,
            )
        )

    @override
    def __call__(self, image_paths: list[str], class_name: str) -> DetectionResult:
        anomaly_scores = []
        anomaly_masks = []
        for image_path in image_paths:
            response = self.send_request({"image_path": image_path})

            anomaly_mask = self.read_shared_array()
            anomaly_score = response["anomaly_score"]

            anomaly_masks.append(anomaly_mask)
            anomaly_scores.append(anomaly_score)

        return DetectionResult(
            pred_scores=np.array(anomaly_scores), anomaly_maps=np.array(anomaly_masks)
        )