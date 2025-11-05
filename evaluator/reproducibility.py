import os
from typing import Iterable
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader

global_seed: int | None = None


def init(seed: int):
    """
    设置全局种子并初始化以确保可复现性。
    """
    global global_seed
    global_seed = seed
    # 1. 设置 Python, NumPy, PyTorch 的种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 2. 设置 CUDA 的种子 (如果可用)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 适用于多 GPU

        # 3. 关键：设置 CuDNN 的确定性
        # 这会牺牲一些性能，但对于严格的可复现性是必需的
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True)
    # CUBLAS_WORKSPACE_CONFIG=:4096:8
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    print(f"Global seed set to: {seed}")


def seed_worker(worker_id):
    """
    DataLoader worker 的初始化函数。
    确保每个 worker 都有一个唯一的、可复现的种子。
    Torch 的种子是确定的，因此需要固定 numpy 和 random 的种子
    """
    # torch.initial_seed() 返回主进程的初始种子
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)


def get_global_state() -> dict:
    state = {
        "rng_state_python": random.getstate(),
        "rng_state_numpy": np.random.get_state(),
        "rng_state_torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["rng_state_cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_global_state(state: dict):
    random.setstate(state["rng_state_python"])
    np.random.set_state(state["rng_state_numpy"])
    torch.set_rng_state(state["rng_state_torch"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["rng_state_cuda"])


dataloader_gen_seed_delta = 0


def get_reproducible_dataloader[T](
    dataset: Dataset[T],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    collate_fn=None,
) -> DataLoader[T]:
    """
    创建一个可复现的 DataLoader。
    调用顺序会影响每个 DataLoader 的种子。
    通过设置 worker_init_fn 和 generator 来确保每次运行时数据顺序相同。
    """
    assert (
        global_seed is not None
    ), "Please call init(seed) before creating dataloaders."
    global dataloader_gen_seed_delta
    g = torch.Generator()
    g.manual_seed(global_seed + dataloader_gen_seed_delta)
    dataloader_gen_seed_delta += 1
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=collate_fn,
    )


def get_dataloader_state(dataloader: DataLoader) -> dict:
    generator: torch.Generator = dataloader.generator  # type: ignore
    state = {
        "generator": generator.get_state(),
    }
    return state


def set_dataloader_state(dataloader: DataLoader, state: dict):
    """
    dataloader: 必须是初始化后没进行任何操作的
    """
    generator: torch.Generator = dataloader.generator  # type: ignore
    generator.set_state(state["generator"])
