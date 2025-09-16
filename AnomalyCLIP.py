import hashlib
from multiprocessing import shared_memory
import os
import signal
import socket
import time
from evaluation import *
import subprocess as sp

from evaluation import PixelResult


class AnomalyCLIP(PixelDetector):
    def __init__(self, working_dir: Path = Path("~/AnomalyCLIP").expanduser()):
        super().__init__("AnomalyCLIP")
        self.socket_path = "/tmp/anomalyclip_socket"
        self.shm_name = f"anomaly_mask_{hashlib.md5(b'init').hexdigest()[:16]}"

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
            --host {self.socket_path} \
            --shm_name {self.shm_name}
        """
        self.process = sp.Popen(
            cmd,
            shell=True,
            executable="/bin/bash",
            # stdout=sp.PIPE,
            # stderr=sp.PIPE,
            preexec_fn=os.setsid,  # 创建新的进程组
        )
        # 现在子进程在独立的进程组中：
        # 1. Ctrl+C 不会直接影响子进程
        # 2. 可以精确控制子进程的生命周期
        # 3. 可以杀死整个子进程树
        max_wait = 30
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
    def __call__(self, image_path: str) -> tuple[PixelResult, dict]:
        request = {"image_path": image_path}
        self.socket.send(json.dumps(request).encode("utf-8"))

        response_data = self.socket.recv(4096).decode("utf-8")
        response = json.loads(response_data)

        if response.get("status") != "success":
            raise RuntimeError(f"检测失败: {response.get('message', '未知错误')}")

        shm_anomaly_mask = np.ndarray((518, 518), dtype=np.float32, buffer=self.shm.buf)
        anomaly_mask = np.zeros((518, 518), dtype=np.float32)
        np.copyto(anomaly_mask, shm_anomaly_mask)
        anomaly_score = response["anomaly_score"]

        return PixelResult(anomaly_map=anomaly_mask, pred_score=anomaly_score), {}


    def __del__(self):
        if hasattr(self, "socket"):
            self.socket.send("QUIT".encode("utf-8"))
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
            # self.shm.unlink()
