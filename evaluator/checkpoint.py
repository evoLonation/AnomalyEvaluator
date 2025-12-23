from typing import Any, Callable
import torch
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
from dataclasses import dataclass

import evaluator.reproducibility as repro


def get_model_dynamic_state_keys(model: nn.Module) -> set:
    """
    获取模型中动态状态（可训练参数和缓冲区）的键集合。
    1. 所有的缓冲区 (Buffers)
    2. 仅可训练的 (requires_grad=True) 参数 (Parameters)
    """
    # 不一定在 state_dict 中
    buffer_ids = {buf.data_ptr() for buf in model.buffers()}
    trainable_ids = {
        param.data_ptr() for param in model.parameters() if param.requires_grad
    }
    parameter_ids = {param.data_ptr() for param in model.parameters()}
    all_data_ids = set.union(parameter_ids, buffer_ids)

    dynamic_keys = set()
    for name, value in model.state_dict().items():
        data_ptr = value.data_ptr()
        assert data_ptr in all_data_ids, f"{name} (type {type(value)})"
        if data_ptr in buffer_ids:
            # 缓冲区
            dynamic_keys.add(name)
        elif data_ptr in trainable_ids:
            # 可训练参数
            dynamic_keys.add(name)
        else:
            # 冻结参数，跳过
            pass
    return dynamic_keys


def get_model_dynamic_state_dict(model: nn.Module) -> dict:
    dynamic_state_dict = {}
    dynamic_keys = get_model_dynamic_state_keys(model)
    state_dict = model.state_dict()
    for key in dynamic_keys:
        dynamic_state_dict[key] = state_dict[key]
    print(f"Dynamic state: {dynamic_state_dict.keys()}")
    return dynamic_state_dict


def load_model_dynamic_state_dict(model: nn.Module, dynamic_state_dict: dict):
    full_state_dict = model.state_dict()
    dynamic_keys = get_model_dynamic_state_keys(model)
    assert set(dynamic_state_dict.keys()) == dynamic_keys, (
        set(dynamic_state_dict.keys()),
        dynamic_keys,
    )
    for key, value in dynamic_state_dict.items():
        full_state_dict[key] = value
    model.load_state_dict(full_state_dict)


type StateMapper = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass
class TrainCheckpointState:
    rng_global_state: repro.GlobalRNGStates
    model: dict
    optimizer: dict
    dataloaders: dict[str, dict]

    @staticmethod
    def get_ckpt_file_name(result_dir: Path, epoch_num: int) -> Path:
        return result_dir / "ckpt" / f"ckpt_epoch_{epoch_num}.pt"

    @staticmethod
    def from_ckpt(result_dir: Path, epoch_num: int) -> "TrainCheckpointState":
        ckpt_file = TrainCheckpointState.get_ckpt_file_name(result_dir, epoch_num)
        return torch.load(ckpt_file.as_posix(), weights_only=False)

    def to_ckpt(self, result_dir: Path, epoch_num: int):
        ckpt_file = TrainCheckpointState.get_ckpt_file_name(result_dir, epoch_num)
        ckpt_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_ckpt_file = ckpt_file.with_suffix(".pt.tmp")
        torch.save(self, tmp_ckpt_file.as_posix())
        tmp_ckpt_file.rename(ckpt_file)

    @staticmethod
    def save_ckpt(
        result_dir: Path,
        epoch_num: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        dataloaders: dict[str, DataLoader],
    ):
        checkpoint_state = TrainCheckpointState(
            rng_global_state=repro.GlobalRNGStates.from_current(),
            model=get_model_dynamic_state_dict(model),
            optimizer=optimizer.state_dict(),
            dataloaders={
                name: repro.get_dataloader_state(dataloader)
                for name, dataloader in dataloaders.items()
            },
        )
        checkpoint_state.to_ckpt(result_dir, epoch_num)

    @staticmethod
    def load_ckpt(
        result_dir: Path,
        epoch_num: int,
        model: nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        dataloaders: dict[str, DataLoader] | None = None,
        strict: bool = True,
        model_mapper: StateMapper = lambda x: x,
    ):
        if strict:
            assert model is not None
            assert optimizer is not None
            assert dataloaders is not None
        checkpoint_state = TrainCheckpointState.from_ckpt(result_dir, epoch_num)
        repro.GlobalRNGStates.apply(checkpoint_state.rng_global_state)
        if model is not None:
            state_dict = model_mapper(checkpoint_state.model)
            load_model_dynamic_state_dict(model, state_dict)
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint_state.optimizer)
        if dataloaders is not None:
            for name, dataloader in dataloaders.items():
                repro.set_dataloader_state(
                    dataloader, checkpoint_state.dataloaders[name]
                )
