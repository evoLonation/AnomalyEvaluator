from dataclasses_json import DataClassJsonMixin, config
from dataclasses import dataclass, field
import datetime
import json
from pathlib import Path

import numpy as np
import pytz

from evaluator.evaluation import evaluation_detection
import evaluator.reproducibility as repro
from .clip import CLIP, CLIPConfig
from .checkpoint import TrainCheckpointState
import torch
from torch.utils.data import DataLoader
from data import MVTecAD, VisA
from tqdm import tqdm


@dataclass
class TrainConfig(DataClassJsonMixin):
    version: str = "v1"
    model_name: str = "openai/clip-vit-large-patch14-336"
    lr: float = 1e-3
    batch_size: int = 16
    image_size: tuple[int, int] = (336, 336)
    num_epochs: int = 15
    seed: int = 42
    enable_vvv: bool = False
    device: torch.device = field(
        default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        metadata=config(
            encoder=lambda d: str(d),
            decoder=lambda s: torch.device(s),
        ),
    )


def get_model(config: TrainConfig) -> CLIP:
    clip = CLIP(
        CLIPConfig(
            model_name=config.model_name,
            input_image_size=config.image_size,
            enable_vvv=config.enable_vvv,
            device=config.device,
        ),
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
    epoch_image_loss: list[float]
    epoch_pixel_loss: list[float]

    def save(self, result_dir: Path):
        (result_dir / "total_state.json").write_text(self.to_json(indent=4))

    @staticmethod
    def load(result_dir: Path) -> "GlobalTrainState":
        return GlobalTrainState.from_json(
            (result_dir / "total_state.json").open("r").read()
        )


def train(
    config: TrainConfig | None = None,
    resume_dir: Path | None = None,
    test: bool = True,
    dir_suffix: str = "",
):
    if config is not None:
        global_train_state = GlobalTrainState(
            config=config,
            done=False,
            trained_epoch=0,
            epoch_loss=[],
            epoch_image_loss=[],
            epoch_pixel_loss=[],
        )
        result_dir = get_result_dir()
        if dir_suffix != "":
            result_dir = result_dir.parent / (result_dir.name + "_" + dir_suffix)
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
            resize=config.image_size,
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
        image_loss_list = []
        pixel_loss_list = []
        for category, dataloader in tqdm(
            dataloaders.items(), desc=f"category", position=1, leave=False
        ):
            for batch in tqdm(dataloader, desc=f"batch", position=2, leave=False):
                pixel_values = batch.images.to(config.device)
                labels = batch.labels.to(config.device)
                image_masks = batch.masks.to(config.device)
                logits_per_image, logits_per_pixel = clip(pixel_values=pixel_values)
                loss, image_loss, pixel_loss = clip.get_loss(
                    logits_per_image, logits_per_pixel, labels, image_masks
                )
                loss_list.append(loss.item())
                image_loss_list.append(image_loss.item())
                pixel_loss_list.append(pixel_loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        avg_loss = sum(loss_list) / len(loss_list)
        avg_image_loss = sum(image_loss_list) / len(image_loss_list)
        avg_pixel_loss = sum(pixel_loss_list) / len(pixel_loss_list)
        print(
            f"Epoch {epoch}, Loss: {avg_loss}, Image Loss: {avg_image_loss}, Pixel Loss: {avg_pixel_loss}"
        )

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
            global_train_state.epoch_image_loss.append(avg_image_loss)
            global_train_state.epoch_pixel_loss.append(avg_pixel_loss)
            if epoch == config.num_epochs:
                global_train_state.done = True
            global_train_state.save(result_dir)

        rng_state = repro.GlobalRNGStates.from_current()

    if test:
        detector = CLIPDetector(clip, config.image_size, config.device)
        evaluation_detection(
            result_dir / "evaluation",
            detector,
            dataset=VisA(),
            batch_size=16,
            namer=lambda _, dset: f"{dset.name}_epoch{config.num_epochs}",
        )


def test(
    result_dir: Path,
    epoch_num: int | None = None,
    suffix: str = "",
):
    global_train_state = GlobalTrainState.load(result_dir)
    config = global_train_state.config
    repro.init(config.seed)
    clip = get_model(config)
    if epoch_num is None:
        assert global_train_state.done, "Training not completed yet."
        assert global_train_state.trained_epoch == config.num_epochs
        epoch_num = config.num_epochs
    else:
        assert 0 <= epoch_num <= global_train_state.trained_epoch
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
        namer=lambda _, dset: f"{dset.name}_epoch{epoch_num}{'_' + suffix if suffix else ''}",
    )
