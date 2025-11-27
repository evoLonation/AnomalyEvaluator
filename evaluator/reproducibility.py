import contextlib
from dataclasses import dataclass
import os
from typing import Iterable
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler

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


def get_global_seed() -> int:
    assert (
        global_seed is not None
    ), "Please call init(seed) before getting the global seed."
    return global_seed


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


_dataloader_gen_seed_delta = 0


def get_reproducible_dataloader[T](
    dataset: Dataset[T],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    collate_fn=None,
    sampler: Sampler | None = None,
) -> DataLoader[T]:
    """
    创建一个可复现的 DataLoader。
    调用顺序会影响每个 DataLoader 的种子。
    通过设置 worker_init_fn 和 generator 来确保每次运行时数据顺序相同。
    """
    assert (
        global_seed is not None
    ), "Please call init(seed) before creating dataloaders."
    global _dataloader_gen_seed_delta
    g = torch.Generator()
    g.manual_seed(global_seed + _dataloader_gen_seed_delta)
    _dataloader_gen_seed_delta += 1
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=collate_fn,
        sampler=sampler,
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


# Can not use list[torch.Tensor] in type hinting because it will cause random check by beartype, change the state
class GlobalRNGStates:
    def __init__(self, python_state, numpy_state, torch_cpu_state, torch_cuda_state):
        self.python_state = python_state
        self.numpy_state = numpy_state
        self.torch_cpu_state = torch_cpu_state
        self.torch_cuda_state = torch_cuda_state

    @staticmethod
    def from_current() -> "GlobalRNGStates":
        python_state = random.getstate()
        numpy_state: tuple = np.random.get_state()  # type: ignore
        torch_cpu_state = torch.get_rng_state()
        torch_cuda_state = None
        if torch.cuda.is_available():
            torch_cuda_state = torch.cuda.get_rng_state_all()
        assert python_state == random.getstate()
        return GlobalRNGStates(
            python_state=python_state,
            numpy_state=numpy_state,
            torch_cpu_state=torch_cpu_state,
            torch_cuda_state=torch_cuda_state,
        )

    def apply(self):
        random.setstate(self.python_state)
        np.random.set_state(self.numpy_state)
        torch.set_rng_state(self.torch_cpu_state)
        if self.torch_cuda_state is not None:
            torch.cuda.set_rng_state_all(self.torch_cuda_state)

    def check_consistent(self):
        self.check_equal(GlobalRNGStates.from_current())

    def check_equal(self, other: "GlobalRNGStates"):
        assert (
            self.python_state == other.python_state
        ), f"Global `python` RNG state has been modified!"

        assert self._compare_np_states(
            self.numpy_state, other.numpy_state
        ), "Global `numpy` RNG state has been modified!"

        assert torch.equal(
            self.torch_cpu_state, other.torch_cpu_state
        ), "Global `torch` (CPU) RNG state has been modified!"

        if self.torch_cuda_state is not None:
            assert other.torch_cuda_state is not None
            assert all(
                torch.equal(b, a)
                for b, a in zip(self.torch_cuda_state, other.torch_cuda_state)
            ), "Global `torch` (CUDA) RNG state has been modified!"

    # --- 辅助函数，用于正确比较 NumPy 状态 ---
    # (NumPy 状态是一个元组，其中包含数组，不能直接用 == 比较)
    @staticmethod
    def _compare_np_states(state1: tuple, state2: tuple) -> bool:
        if len(state1) != len(state2):
            return False
        # 比较元组的非数组部分
        if state1[0] != state2[0] or state1[2:] != state2[2:]:
            return False
        # 比较数组部分
        return np.array_equal(state1[1], state2[1])


rng_state_checkpoint: GlobalRNGStates = GlobalRNGStates.from_current()


@contextlib.contextmanager
def RNGStateChecker(restore: bool = False):
    """
    一个上下文管理器，用于断言在其内部运行的代码
    不会改变全局的 `random`, `numpy`, `torch` RNG 状态。

    用法:
    with RNGStateChecker():
        # 如果这里的代码调用了 `np.random.rand()`，
        # 上下文管理器将在退出时抛出 AssertionError。
        code_to_check()
    参数:
    restore: 如果为 True，则检查到更改后只进行警告并恢复状态，而不是抛出异常。
    """
    # print(" [RNGStateChecker] Capturing Global RNG State...")
    # 1. 获取 "Before" 状态
    global_state = GlobalRNGStates.from_current()
    assert global_state.python_state == random.getstate()

    try:
        # 2. 运行用户的代码
        yield
    finally:
        # print(" [RNGStateChecker] Validating RNG State...")
        if restore:
            try:
                global_state.check_equal(GlobalRNGStates.from_current())
            except AssertionError as e:
                print(" [RNGStateChecker] Assert Warning: ", str(e))
                print(" [RNGStateChecker] Restoring Global RNG State...")
                global_state.apply()
        else:
            global_state.check_equal(GlobalRNGStates.from_current())
            # print(
            #     " [RNGStateChecker] Validation passed. Global RNG state has not changed."
            # )


if __name__ == "__main__":
    init(42)
    with RNGStateChecker():
        pass  # 占位，确保后续代码块的 RNG 状态正确

    try:
        with RNGStateChecker():
            a = np.random.rand(3)
            b = torch.rand(3)
            c = random.random()
            print(a, b, c)
        raise ValueError("Should have raised an assertion error!")
    except AssertionError as e:
        print("Successfully caught an assertion error as expected:", e)
