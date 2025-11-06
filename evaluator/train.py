from dataclasses_json import DataClassJsonMixin, config
from dataclasses import dataclass, field
import datetime
import json
from pathlib import Path

import numpy as np
import pytz

from evaluator.evaluation import evaluation_detection
import evaluator.reproducibility as repro
from .clip import CLIP, CLIPConfig, CLIPDetector
import torch
from torch import nn
from torch.utils.data import DataLoader
from evaluator.data import MVTecAD, VisA
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


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


# def get_checkpoint_state(
#     objects: dict[str, nn.Module | torch.optim.Optimizer | DataLoader],
# ) -> dict:
#     checkpoint = {}
#     checkpoint["global_state"] = repro.GlobalRNGStates.from_current()
#     for name, obj in objects.items():
#         if isinstance(obj, nn.Module):
#             checkpoint[f"{name}_state_dict"] = get_model_dynamic_state_dict(obj)
#         elif isinstance(obj, torch.optim.Optimizer):
#             checkpoint[f"{name}_state_dict"] = obj.state_dict()
#         elif isinstance(obj, DataLoader):
#             checkpoint[f"{name}_dataloader_state"] = repro.get_dataloader_state(obj)
#         else:
#             raise ValueError(f"Unsupported type for checkpoint: {type(obj)}")
#     return checkpoint


# def resume_checkpoint_state(
#     state: dict,
#     objects: dict[str, nn.Module | torch.optim.Optimizer | DataLoader],
#     strict: bool = True,
# ):
#     repro.GlobalRNGStates.apply(state.pop("global_state"))
#     for name, obj in objects.items():
#         if isinstance(obj, nn.Module):
#             load_model_dynamic_state_dict(obj, state.pop(f"{name}_state_dict"))
#         elif isinstance(obj, torch.optim.Optimizer):
#             obj.load_state_dict(state.pop(f"{name}_state_dict"))
#         elif isinstance(obj, DataLoader):
#             repro.set_dataloader_state(obj, state.pop(f"{name}_dataloader_state"))
#         else:
#             raise ValueError(f"Unsupported type for checkpoint: {type(obj)}")
#     if strict:
#         assert len(state) == 0, f"Unused keys in checkpoint state: {state.keys()}"


@dataclass
class TrainConfig(DataClassJsonMixin):
    version: str = "v1"
    model_name: str = "openai/clip-vit-large-patch14-336"
    lr: float = 1e-3
    batch_size: int = 16
    image_size: tuple[int, int] = (336, 336)
    num_epochs: int = 10
    seed: int = 42
    device: torch.device = field(
        default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        metadata=config(
            encoder=lambda d: str(d),
            decoder=lambda s: torch.device(s),
        ),
    )

    # def to_json(self) -> dict:
    #     json = {}
    #     for field in self.__dataclass_fields__.values():
    #         value = getattr(self, field.name)
    #         if isinstance(value, Path):
    #             json[field.name] = value.as_posix()
    #         elif isinstance(value, torch.device):
    #             json[field.name] = str(value)
    #         else:
    #             json[field.name] = value
    #     return json

    # @staticmethod
    # def from_json(json: dict) -> "TrainConfig":
    #     kwargs = {}
    #     for field in TrainConfig.__dataclass_fields__.values():
    #         if field.name in json:
    #             value = json[field.name]
    #             if field.type == Path:
    #                 kwargs[field.name] = Path(value)
    #             elif field.type == torch.device:
    #                 kwargs[field.name] = torch.device(value)
    #             elif field.type == tuple[int, int]:
    #                 kwargs[field.name] = tuple(value)
    #             else:
    #                 kwargs[field.name] = value
    #     return TrainConfig(**kwargs)


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


@dataclass
class GlobalTrainState(DataClassJsonMixin):
    config: TrainConfig
    done: bool
    trained_epoch: int
    epoch_loss: list[float]

    def save(self, result_dir: Path):
        (result_dir / "total_state.json").write_text(self.to_json(indent=4))

    @staticmethod
    def load(result_dir: Path) -> "GlobalTrainState":
        return GlobalTrainState.from_json(
            (result_dir / "total_state.json").open("r").read()
        )


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
        torch.save(self, ckpt_file.as_posix())

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
    ):
        if strict:
            assert model is not None
            assert optimizer is not None
            assert dataloaders is not None
        checkpoint_state = TrainCheckpointState.from_ckpt(result_dir, epoch_num)
        repro.GlobalRNGStates.apply(checkpoint_state.rng_global_state)
        if model is not None:
            load_model_dynamic_state_dict(model, checkpoint_state.model)
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint_state.optimizer)
        if dataloaders is not None:
            for name, dataloader in dataloaders.items():
                repro.set_dataloader_state(
                    dataloader, checkpoint_state.dataloaders[name]
                )


def train(config: TrainConfig | None = None, resume_dir: Path | None = None):
    if config is not None:
        global_train_state = GlobalTrainState(
            config=config,
            done=False,
            trained_epoch=0,
            epoch_loss=[],
        )
        result_dir = get_result_dir()
        result_dir.mkdir(parents=True, exist_ok=False)
        global_train_state.save(result_dir)
    else:
        assert resume_dir is not None
        global_train_state = GlobalTrainState.load(resume_dir)
        config = global_train_state.config
        result_dir = resume_dir
    repro.init(config.seed)

    # summary_writer = SummaryWriter(log_dir=(save_dir / "tensorboard").as_posix())

    clip = get_model(config)
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
    with repro.RNGStateChecker():
        if resume_dir is not None:
            trained_epoch = global_train_state.trained_epoch
            print(f"Resuming from epoch {trained_epoch}")
            TrainCheckpointState.load_ckpt(
                resume_dir,
                trained_epoch,
                model=clip,
                optimizer=optimizer,
                dataloaders=dataloaders,
            )
        else:
            trained_epoch = 0  # means initial state
            TrainCheckpointState.save_ckpt(
                result_dir,
                trained_epoch,
                model=clip,
                optimizer=optimizer,
                dataloaders=dataloaders,
            )

    rng_state = repro.GlobalRNGStates.from_current()
    for epoch in tqdm(
        range(trained_epoch + 1, config.num_epochs + 1),
        initial=trained_epoch,
        total=config.num_epochs,
        desc="epoch",
        position=0,
        leave=True,
    ):
        rng_state.check_consistent()
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

        with repro.RNGStateChecker():
            TrainCheckpointState.save_ckpt(
                result_dir,
                epoch,
                model=clip,
                optimizer=optimizer,
                dataloaders=dataloaders,
            )
            global_train_state.trained_epoch = epoch
            global_train_state.epoch_loss.append(avg_loss)
            if epoch == config.num_epochs:
                global_train_state.done = True
            global_train_state.save(result_dir)

        rng_state = repro.GlobalRNGStates.from_current()


def test(result_dir: Path, epoch_num: int | None = None):
    global_train_state = GlobalTrainState.load(result_dir)
    config = global_train_state.config
    repro.init(config.seed)
    clip = get_model(config)
    if epoch_num is None:
        assert global_train_state.done, "Training not completed yet."
        assert global_train_state.trained_epoch == config.num_epochs
        epoch_num = config.num_epochs
    else:
        assert 0 < epoch_num <= global_train_state.trained_epoch
    TrainCheckpointState.load_ckpt(
        result_dir,
        epoch_num,
        model=clip,
        strict=False,
    )
    detector = CLIPDetector(clip, config.image_size, config.device)
    evaluation_detection(
        result_dir / "evaluation",
        detector,
        dataset=VisA(),
        batch_size=16,
    )
