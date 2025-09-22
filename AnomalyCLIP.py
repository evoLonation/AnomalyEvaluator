import hashlib
import json
from multiprocessing import shared_memory
import os
from pathlib import Path
import signal
import socket
import time
from typing import Literal, override
import uuid
from evaluation2 import *
import subprocess as sp


class AnomalyCLIP(Detector):
    def __init__(
        self,
        working_dir: Path = Path("~/AnomalyCLIP").expanduser(),
        type: Literal["mvtec", "visa"] = "mvtec",
    ):
        super().__init__(f"AnomalyCLIP_{type}")
        self.id = uuid.uuid4().hex
        self.socket_path = f"/tmp/anomalyclip_socket_{self.id}"
        self.shm_name = f"anomaly_mask_{self.id}"

        self.shm = shared_memory.SharedMemory(
            create=True,
            size=np.zeros((518, 518), dtype=np.float32).nbytes,
            name=self.shm_name,
        )

        # cd {working_dir} && \
        cmd = f"""
        cd {working_dir} && \
        source .venv/bin/activate && \
        python anomaly_detection.py \
            --type {type} \
            --host {self.socket_path} \
            --shm_name {self.shm_name}
        """
        self.process = sp.Popen(
            cmd,
            shell=True,
            executable="/bin/bash",
            # stdout=sp.PIPE,
            # stderr=sp.PIPE,
        )
        max_wait = 60
        wait_time = 0
        while wait_time < max_wait:
            if os.path.exists(self.socket_path):
                print("AnomalyCLIP 服务器已启动")
                break
            if self.process.poll() is not None:
                raise RuntimeError("AnomalyCLIP 服务器启动失败，进程意外退出")
            time.sleep(1)
            wait_time += 1
        if not os.path.exists(self.socket_path):
            raise RuntimeError("AnomalyCLIP 服务器启动失败")
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.connect(self.socket_path)

    @override
    def __call__(self, image_paths: list[str]) -> DetectionResult:
        anomaly_scores = []
        anomaly_masks = []
        for image_path in image_paths:
            request = {"image_path": image_path}
            self.socket.send(json.dumps(request).encode("utf-8"))

            response_data = self.socket.recv(4096).decode("utf-8")
            response = json.loads(response_data)

            if response.get("status") != "success":
                raise RuntimeError(f"检测失败: {response.get('message', '未知错误')}")

            shm_anomaly_mask = np.ndarray(
                (518, 518), dtype=np.float32, buffer=self.shm.buf
            )
            anomaly_mask = np.zeros((518, 518), dtype=np.float32)
            np.copyto(anomaly_mask, shm_anomaly_mask)
            anomaly_score = response["anomaly_score"]

            anomaly_masks.append(anomaly_mask)
            anomaly_scores.append(anomaly_score)

        return DetectionResult(
            pred_scores=np.array(anomaly_scores), anomaly_maps=np.array(anomaly_masks)
        )

    def __del__(self):
        if hasattr(self, "socket"):
            try:
                self.socket.send("QUIT".encode("utf-8"))
            except Exception:
                pass
            self.socket.close()
        if hasattr(self, "process"):
            try:
                self.process.wait(timeout=5)
                print("AnomalyCLIP 服务器已关闭")
            except sp.TimeoutExpired:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=5)
                print("AnomalyCLIP 服务器强制关闭")
        if hasattr(self, "shm"):
            self.shm.close()
            self.shm.unlink()
