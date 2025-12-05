from typing import Iterable
from dataclasses_json import DataClassJsonMixin
import dataclasses_json
from dataclasses import dataclass, field
import datetime
import json
from pathlib import Path
import torch.nn as nn
from torch import Tensor, cdist
from jaxtyping import Float, jaxtyped
from torch.utils.data.sampler import RandomSampler
from torchvision.transforms import Compose, CenterCrop

import numpy as np
import pytz

from data.detection_dataset import TensorSample, TensorSampleBatch
from data.utils import ImageResize, ImageSize, Transform
from evaluator.dinov3 import DINOv3VisionTransformer
from evaluator.evaluation import evaluation_detection
from evaluator.image_normalize import DINO_NORMALIZE
import evaluator.reproducibility as repro
from .clip import CLIP, CLIPConfig
from .checkpoint import TrainCheckpointState
import torch
from torch.utils.data import DataLoader
from data import MVTecAD, VisA
from tqdm import tqdm


@dataclass
class TrainConfig(DataClassJsonMixin):
    lr: float = 1e-3
    batch_size: int = 16
    image_resize: ImageResize = 512
    image_size: ImageSize = field(default_factory=lambda: ImageSize.square(512))
    num_epochs: int = 30
    seed: int = 42
    device: torch.device = field(
        default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        metadata=dataclasses_json.config(
            encoder=lambda d: str(d),
            decoder=lambda s: torch.device(s),
        ),
    )


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

    @staticmethod
    def new(config: TrainConfig) -> "GlobalTrainState":
        return GlobalTrainState(
            config=config,
            done=False,
            trained_epoch=0,
            epoch_loss=[],
        )


class VisionAdapter(nn.Module):
    def __init__(self, embed_dim: int, bottleneck_dim: int = 768):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(embed_dim, bottleneck_dim, bias=False),  # 降维
            nn.LeakyReLU(inplace=False),  # 激活函数
        )
        self.fc2 = nn.Sequential(
            nn.Linear(bottleneck_dim, embed_dim, bias=False),  # 升维还原
            nn.LeakyReLU(inplace=False),  # 激活函数
        )

    def forward(self, x):
        x = self.fc1(x)
        y = self.fc2(x)
        return y


class DINOv3Matcher(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.vision = DINOv3VisionTransformer(device=device)
        self.adapter = VisionAdapter(embed_dim=self.vision.get_embed_dim())
        self.to(device)

        self.device = device

    @jaxtyped(typechecker=None)
    def forward(
        self, image_pairs: Float[Tensor, "N 2 C H W"]
    ) -> Float[Tensor, "N P P"]:
        image_pairs = image_pairs.to(self.device)
        N, _, C, H, W = image_pairs.shape
        D = self.vision.get_embed_dim()
        PH, PW = H // self.vision.get_patch_size(), W // self.vision.get_patch_size()
        P = PH * PW
        images = image_pairs.view(-1, C, H, W)
        features = self.vision(pixel_values=images)
        features = self.adapter(features)
        features: Float[Tensor, "N 2 P D"] = features.view(N, 2, P, D)
        features = features / features.norm(dim=-1, keepdim=True, p=2)
        features1: Float[Tensor, "N P D"] = features[:, 0, :, :]
        features2: Float[Tensor, "N P D"] = features[:, 1, :, :]
        distances: Float[Tensor, "N P P"] = torch.cdist(features1, features2, p=2)
        return distances




def create_model(config: TrainConfig) -> DINOv3Matcher:
    model = DINOv3Matcher(device=config.device)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.adapter.parameters():
        param.requires_grad = True
    print("Trainable parameters:")
    print(f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print([name for name, p in model.named_parameters() if p.requires_grad])
    return model


def get_result_dir(name: str | None) -> Path:
    if name is None:
        # 北京时间，格式化： MMDD_HH:MM:SS
        now = datetime.datetime.now(pytz.timezone("Asia/Shanghai"))
        date = now.strftime("%m.%d_%H:%M:%S")
        name = date
    result_dir = Path(f"results/train_dinov3/{name}")
    return result_dir


unaligned_classes = ["hazelnut", "metal_nut", "screw"]


def train(
    config: TrainConfig | None = None,
    name: str | None = None,
    resume_name: str | None = None,
):
    if config is not None:
        global_train_state = GlobalTrainState.new(config)
        result_dir = get_result_dir(name)
        resume_dir = None
    else:
        assert resume_name is not None
        resume_dir = get_result_dir(resume_name)
        global_train_state = GlobalTrainState.load(resume_dir)
        config = global_train_state.config
        result_dir = resume_dir
    repro.init(config.seed)

    model = create_model(config)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.lr,
    )
    mvtec = MVTecAD()
    transform = Transform(
        resize=config.image_resize,
        image_transform=Compose([CenterCrop(config.image_size.hw()), DINO_NORMALIZE]),
        mask_transform=CenterCrop(config.image_size.hw()),
    )
    with repro.RNGStateChecker():
        if resume_dir is not None:
            trained_epoch = global_train_state.trained_epoch
            TrainCheckpointState.load_ckpt(resume_dir, trained_epoch, model, optimizer)
        else:
            trained_epoch = 0
            TrainCheckpointState.save_ckpt(
                result_dir, trained_epoch, model, optimizer, {}
            )
    for epoch in tqdm(
        range(trained_epoch + 1, config.num_epochs + 1),
        initial=trained_epoch,
        total=config.num_epochs,
        desc="Epoch",
        position=0,
        leave=True,
    ):
        categories = mvtec.get_categories()
        assert set(unaligned_classes).issubset(categories)
        categories = sorted(list(set(categories) - set(unaligned_classes)))
        loss_list = []
        for category in tqdm(categories, desc=f"category", position=1, leave=False):
            dataset = mvtec.get_tensor(category, transform)
            dataloader1 = repro.get_reproducible_dataloader(
                dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=4,
                collate_fn=TensorSample.collate_fn,
                sampler=RandomSampler(
                    dataset,
                    replacement=False,
                    generator=torch.Generator().manual_seed(
                        repro.get_global_seed() + epoch * 2
                    ),
                ),
            )
            dataloader2 = repro.get_reproducible_dataloader(
                dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=4,
                collate_fn=TensorSample.collate_fn,
                sampler=RandomSampler(
                    dataset,
                    replacement=False,
                    generator=torch.Generator().manual_seed(
                        repro.get_global_seed() + epoch * 2 + 1
                    ),
                ),
            )
            patch_size = model.vision.get_patch_size()
            PH = config.image_size.h // patch_size
            PW = config.image_size.w // patch_size
            P = PH * PW
            for batch1, batch2 in tqdm(
                zip(dataloader1, dataloader2),
                initial=0,
                total=len(dataloader1),
                desc=f"{category}",
                position=2,
                leave=False,
            ):
                batch1: TensorSampleBatch
                batch2: TensorSampleBatch
                images1 = batch1.images.to(config.device)
                images2 = batch2.images.to(config.device)
                image_pairs = torch.stack([images1, images2], dim=1)
                distances: Float[Tensor, "N P P"] = model(image_pairs=image_pairs)
                # target_distances 是 patch 之间的坐标距离
                patch_indices = torch.arange(distances.shape[1], device=config.device)
                patch_coord1 = torch.stack(
                    [patch_indices // PH * patch_size, patch_indices % PW * patch_size],
                    dim=-1,
                )
                patch_coord2 = patch_coord1.clone()
                target_distances: Float[Tensor, "P P"] = cdist(
                    patch_coord1.float(), patch_coord2.float(), p=2
                )
                target_distances = target_distances / target_distances.max() * 2
                target_distances: Float[Tensor, "N P P"] = target_distances.unsqueeze(
                    0
                ).repeat(distances.shape[0], 1, 1)
                loss = nn.functional.mse_loss(
                    distances,
                    target_distances,
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())
        avg_loss = sum(loss_list) / len(loss_list)
        print(f"Epoch {epoch}, Loss: {avg_loss}")

        with repro.RNGStateChecker():
            TrainCheckpointState.save_ckpt(result_dir, epoch, model, optimizer, {})
            global_train_state.trained_epoch = epoch
            global_train_state.epoch_loss.append(avg_loss)
            if epoch == config.num_epochs:
                global_train_state.done = True
            global_train_state.save(result_dir)


if __name__ == "__main__":
    config = TrainConfig()
    train(config, name="test")
