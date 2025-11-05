from dataclasses import Field, dataclass
import datetime
from email.policy import strict
import json
from operator import ge
from pathlib import Path
import random
import time

import numpy as np
import pytz

import evaluator.reproducibility as repro
from .clip import CLIP, CLIPConfig
import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Iterable
from evaluator.data import MVTecAD, TensorSampleBatch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def get_model_dynamic_state_keys(model: nn.Module) -> set:
    """
    获取模型中动态状态（可训练参数和缓冲区）的键集合。
    1. 所有的缓冲区 (Buffers)
    2. 仅可训练的 (requires_grad=True) 参数 (Parameters)
    """
    all_param_keys = {name for name, _ in model.named_parameters()}
    trainable_param_keys = {
        name for name, param in model.named_parameters() if param.requires_grad
    }

    dynamic_keys = set()
    for name, _ in model.state_dict().items():
        if name not in all_param_keys:
            # 缓冲区
            dynamic_keys.add(name)
        elif name in trainable_param_keys:
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


def get_checkpoint_state(
    **kwargs: nn.Module | torch.optim.Optimizer | DataLoader,
) -> dict:
    checkpoint = {}
    checkpoint["global_state"] = repro.get_global_state()
    for name, obj in kwargs.items():
        if isinstance(obj, nn.Module):
            checkpoint[f"{name}_state_dict"] = get_model_dynamic_state_dict(obj)
        elif isinstance(obj, torch.optim.Optimizer):
            checkpoint[f"{name}_state_dict"] = obj.state_dict()
        elif isinstance(obj, DataLoader):
            checkpoint[f"{name}_dataloader_state"] = repro.get_dataloader_state(obj)
        else:
            raise ValueError(f"Unsupported type for checkpoint: {type(obj)}")
    return checkpoint


def resume_checkpoint_state(
    state: dict,
    **kwargs: nn.Module | torch.optim.Optimizer | DataLoader,
):
    repro.set_global_state(state["global_state"])
    for name, obj in kwargs.items():
        if isinstance(obj, nn.Module):
            load_model_dynamic_state_dict(obj, state[f"{name}_state_dict"])
        elif isinstance(obj, torch.optim.Optimizer):
            obj.load_state_dict(state[f"{name}_state_dict"])
        elif isinstance(obj, DataLoader):
            repro.set_dataloader_state(obj, state[f"{name}_dataloader_state"])
        else:
            raise ValueError(f"Unsupported type for checkpoint: {type(obj)}")


@dataclass
class TrainConfig:
    version: str = "v1"
    model_name: str = "openai/clip-vit-large-patch14-336"
    lr: float = 1e-3
    batch_size: int = 16
    image_size: tuple[int, int] = (336, 336)
    num_epochs: int = 10
    seed: int = 42
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def to_json(self) -> dict:
        json = {}
        for field in self.__dataclass_fields__.values():
            value = getattr(self, field.name)
            if isinstance(value, Path):
                json[field.name] = value.as_posix()
            elif isinstance(value, torch.device):
                json[field.name] = str(value)
            else:
                json[field.name] = value
        return json

    @staticmethod
    def from_json(json: dict) -> "TrainConfig":
        kwargs = {}
        for field in TrainConfig.__dataclass_fields__.values():
            if field.name in json:
                value = json[field.name]
                if field.type == Path:
                    kwargs[field.name] = Path(value)
                elif field.type == torch.device:
                    kwargs[field.name] = torch.device(value)
                elif field.type == tuple[int, int]:
                    kwargs[field.name] = tuple(value)
                else:
                    kwargs[field.name] = value
        return TrainConfig(**kwargs)


def get_model(config: TrainConfig) -> CLIP:
    clip = CLIP(
        CLIPConfig(model_name=config.model_name, input_image_size=config.image_size),
        device=config.device,
    )
    for param in clip.parameters():
        param.requires_grad = False
    for param in clip.normal_prompt.learnables.parameters():
        param.requires_grad = True
    for param in clip.anomaly_prompt.learnables.parameters():
        param.requires_grad = True
    print("Trainable parameters:")
    print(f"{sum(p.numel() for p in clip.parameters() if p.requires_grad):,}")
    return clip


def get_result_dir() -> Path:
    # 北京时间，格式化： MMDD_HH:MM:SS
    now = datetime.datetime.now(pytz.timezone("Asia/Shanghai"))
    date = now.strftime("%m.%d_%H:%M:%S")
    result_dir = Path(f"results/train/{date}")
    return result_dir


def train(config: TrainConfig | None = None, resume_dir: Path | None = None):
    if config is not None:
        repro.init(config.seed)
        global_train_state = {
            "config": config.to_json(),
            "done": False,
            "trained_epoch": 0,
        }
        result_dir = get_result_dir()
        result_dir.mkdir(parents=True, exist_ok=False)
        json.dump(
            global_train_state, (result_dir / "total_state.json").open("w"), indent=4
        )
    else:
        assert resume_dir is not None
        global_train_state = json.load((resume_dir / "total_state.json").open("r"))
        config = TrainConfig.from_json(global_train_state["config"])
        repro.init(config.seed)
        result_dir = resume_dir

    # summary_writer = SummaryWriter(log_dir=(save_dir / "tensorboard").as_posix())
    ckpt_dir = result_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_namer = lambda epoch: ckpt_dir / f"ckpt_epoch_{epoch}.pt"

    clip = get_model(config)
    print(set(clip.state_dict().keys()) - set([x[0] for x in clip.named_parameters()]))
    optimizer = torch.optim.AdamW(
        [p for p in clip.parameters() if p.requires_grad],
        lr=config.lr,
    )
    criterion = nn.CrossEntropyLoss()
    mvtec = MVTecAD()
    dataloaders: dict[str, DataLoader] = {
        category: repro.get_reproducible_dataloader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=dataset.collate_fn,
        )
        for category, dataset in mvtec.get_tensor_dataset(
            image_size=config.image_size,
        ).category_datas.items()
    }
    train_state = {
        "clip": clip,
        "optimizer": optimizer,
        **dataloaders,
    }
    if resume_dir is not None:
        trained_epoch = global_train_state["trained_epoch"]
        print(f"Resuming from epoch {trained_epoch}")
        ckpt_file = ckpt_namer(trained_epoch)
        checkpoint_state = torch.load(ckpt_file.as_posix(), weights_only=False)
        resume_checkpoint_state(checkpoint_state, **train_state)
    else:
        trained_epoch = 0  # means initial state
        ckpt_file = ckpt_namer(trained_epoch)
        checkpoint_state = get_checkpoint_state(**train_state)
        torch.save(get_checkpoint_state(**train_state), ckpt_file.as_posix())

    for epoch in tqdm(
        range(trained_epoch + 1, config.num_epochs + 1),
        initial=trained_epoch + 1,
        total=config.num_epochs + 1,
        desc="epoch",
        position=0,
        leave=True,
    ):
        loss_list = []
        for category, dataloader in tqdm(
            dataloaders.items(), desc=f"category", position=1, leave=False
        ):
            for batch in tqdm(dataloader, desc=f"batch", position=2, leave=False):
                pixel_values = batch.images.to(config.device)
                labels = batch.labels.to(config.device).long()
                logits_per_image = clip(pixel_values=pixel_values)
                loss = criterion(logits_per_image, labels)
                loss_list.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        avg_loss = sum(loss_list) / len(loss_list)
        print(f"Epoch {epoch}, Loss: {avg_loss}")

        ckpt_file = ckpt_namer(epoch)
        checkpoint_state = get_checkpoint_state(**train_state)
        torch.save(checkpoint_state, ckpt_file.as_posix())
        global_train_state["trained_epoch"] = epoch
        global_train_state.setdefault("epoch_loss", []).append(avg_loss)
        if epoch == config.num_epochs:
            global_train_state["done"] = True
        json.dump(
            global_train_state, (result_dir / "total_state.json").open("w"), indent=4
        )
