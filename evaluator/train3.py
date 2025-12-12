from typing import Iterable, override
from dataclasses_json import DataClassJsonMixin
import dataclasses_json
from dataclasses import dataclass, field
import datetime
import json
from pathlib import Path
import torch.nn as nn
from torch import Tensor, cdist
from jaxtyping import Float, jaxtyped, Int
from torch.utils.data.sampler import RandomSampler
from torch.utils.data import Subset
from torchvision.transforms import Compose, CenterCrop
import torch.nn.functional as F

import numpy as np
import pytz

from data.detection_dataset import TensorSample, TensorSampleBatch
from data.utils import ImageResize, ImageSize, Transform
from data.base import Dataset
from evaluator.vit import VisionTransformerBase
from evaluator.dinov3 import DINOv3VisionTransformer
from evaluator.evaluation import evaluation_detection
from evaluator.image_normalize import DINO_NORMALIZE
from evaluator.loss import binary_dice_loss, focal_loss
import evaluator.reproducibility as repro
from evaluator.train2 import VisionAdapter
from evaluator.trainer import BaseTrainer, TrainConfig
from common.utils import generate_call_signature
from .clip import CLIP, CLIPConfig
from .checkpoint import TrainCheckpointState
import torch
from torch.utils.data import DataLoader
from data import MVTecAD, VisA, RealIAD
from tqdm import tqdm


class Model(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.vision = DINOv3VisionTransformer(device=device)
        self.adapter = VisionAdapter(embed_dim=self.vision.get_embed_dim())
        self.to(device)

        self.device = device

    @dataclass
    class Result:
        similarities: Float[Tensor, "H W"]  # 像素级异常分数
        max_similarity: Float[Tensor, ""]  # 图像级异常分数

    @jaxtyped(typechecker=None)
    def forward(
        self,
        images: Float[Tensor, "N1 3 H W"],
        normal_images: Float[Tensor, "N2 3 H W"],
    ) -> tuple[Float[Tensor, "N1"], Float[Tensor, "N1 H W"]]:
        images = images.to(self.device)
        normal_images = normal_images.to(self.device)
        features = self.vision(pixel_values=images)
        features = self.adapter(features)
        features: Float[Tensor, "N1 P D"] = features / features.norm(
            dim=-1, keepdim=True
        )
        normal_features = self.vision(pixel_values=normal_images)
        normal_features = self.adapter(normal_features)
        normal_features: Float[Tensor, "N2 P D"] = (
            normal_features / normal_features.norm(dim=-1, keepdim=True)
        )
        N1, _, H, W = images.shape
        _, P, D = features.shape
        N2 = normal_features.shape[0]
        patch_size = self.vision.get_patch_size()
        PH = images.shape[2] // patch_size
        PW = images.shape[3] // patch_size
        # 计算相似度
        distances = (
            torch.cdist(
                features.reshape(N1 * P, D),
                normal_features.reshape(N2 * P, D),
                p=2,
            )
            .reshape(N1, P, N2, P)
            .permute(0, 2, 1, 3)
        )  # N1 x N2 x P x P
        distances = distances / 2  # 归一化到 [0, 1]
        scores_patch = distances.min(dim=-1).values  # N1 x N2 x P
        scores = scores_patch.max(dim=-1).values  # N1 x N2
        scores_patch = scores_patch.mean(dim=1)  # N1 x P
        scores_patch = scores_patch.reshape(N1, PH, PW)
        scores_pixel = F.interpolate(
            scores_patch.unsqueeze(1),
            size=images.shape[2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(
            1
        )  # N1 x H x W
        scores = torch.mean(scores, dim=-1)  # N1
        return scores, scores_pixel

    @generate_call_signature(forward)
    def __call__(self): ...


class VisionTransformer(VisionTransformerBase):
    def __init__(self, model: Model, device: torch.device):
        super().__init__()
        self.vision = model.vision
        self.adapter = model.adapter
        self.device = device
        self.to(device)

    def forward(self, pixel_values: Float[Tensor, "N 3 H W"]) -> Float[Tensor, "N P D"]:
        pixel_values = pixel_values.to(self.device)
        features = self.vision(pixel_values=pixel_values)
        features = self.adapter(features)
        return features

    @generate_call_signature(forward)
    def __call__(self): ...

    @override
    def get_embed_dim(self) -> int:
        return self.vision.get_embed_dim()

    @override
    def get_patch_size(self) -> int:
        return self.vision.get_patch_size()


def create_model(config: TrainConfig) -> Model:
    model = Model(device=config.device)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.adapter.parameters():
        param.requires_grad = True
    print("Trainable parameters:")
    print(f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print([name for name, p in model.named_parameters() if p.requires_grad])
    return model


class MatchTrainer(BaseTrainer):
    @classmethod
    def setup_model(cls, config: TrainConfig) -> nn.Module:
        model = create_model(config)
        return model

    @classmethod
    def setup_optimizer(
        cls, config: TrainConfig, model: nn.Module
    ) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.lr,
        )
        return optimizer

    def setup_other_components(self):
        # self.dataset = RealIAD()
        self.dataset = MVTecAD()
        self.transform = Transform(
            resize=self.config.image_resize,
            image_transform=Compose(
                [CenterCrop(self.config.centercrop.hw()), DINO_NORMALIZE]
            ),
            mask_transform=CenterCrop(self.config.centercrop.hw()),
        )
        self.categories = self.dataset.get_categories()

    def train_one_epoch(self, epoch: int, model: nn.Module) -> dict[str, float]:
        assert isinstance(model, Model)
        loss_list = []
        image_loss_list = []
        dice_loss_list = []
        focal_loss_list = []
        for category in self.categories:
            tensor_data = self.dataset.get_tensor(category, self.transform)
            dataloader = repro.get_reproducible_dataloader(
                tensor_data,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=4,
                collate_fn=TensorSample.collate_fn,
                sampler=RandomSampler(
                    tensor_data,
                    replacement=False,
                    generator=torch.Generator().manual_seed(
                        repro.get_global_seed() + epoch,
                    ),
                ),
            )
            normal_indices = [
                i for i, x in enumerate(self.dataset.get_labels(category)) if x == 0
            ]
            normal_tensor_data = Dataset.bypt(
                Subset(
                    tensor_data,
                    normal_indices,
                )
            )
            normal_dataloader = repro.get_reproducible_dataloader(
                normal_tensor_data,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=4,
                collate_fn=TensorSample.collate_fn,
                sampler=RandomSampler(
                    normal_tensor_data,
                    replacement=True,
                    num_samples=len(tensor_data),
                    generator=torch.Generator().manual_seed(
                        repro.get_global_seed() + epoch,
                    ),
                ),
            )
            patch_size = model.vision.get_patch_size()
            PH = self.config.centercrop.h // patch_size
            PW = self.config.centercrop.w // patch_size
            P = PH * PW
            D = model.vision.get_embed_dim()
            category_loss_idx = len(loss_list)
            for batch, normal_batch in tqdm(
                zip(dataloader, normal_dataloader),
                desc=f"{category}",
                position=2,
                leave=False,
                initial=0,
                total=len(dataloader),
            ):
                batch: TensorSampleBatch
                normal_batch: TensorSampleBatch
                N = len(batch.images)
                scores, maps = model(
                    images=batch.images,
                    normal_images=normal_batch.images,
                )
                image_loss = F.cross_entropy(
                    scores, batch.labels.to(scores.device).float()
                )
                pixel_loss_focal = focal_loss(
                    torch.stack([1 - maps, maps], dim=1), batch.masks.to(maps.device)
                )
                pixel_loss_dice = binary_dice_loss(maps, batch.masks.to(maps.device))
                pixel_loss = pixel_loss_focal + pixel_loss_dice
                loss = image_loss + pixel_loss
                self.optimize_step(loss)
                loss_list.append(loss.item())
                image_loss_list.append(image_loss.item())
                focal_loss_list.append(pixel_loss_focal.item())
                dice_loss_list.append(pixel_loss_dice.item())
            category_avg_loss = sum(loss_list[category_loss_idx:]) / (
                len(loss_list) - category_loss_idx
            )
            category_avg_image_loss = sum(image_loss_list[category_loss_idx:]) / (
                len(image_loss_list) - category_loss_idx
            )
            category_avg_focal_loss = sum(focal_loss_list[category_loss_idx:]) / (
                len(focal_loss_list) - category_loss_idx
            )
            category_avg_dice_loss = sum(dice_loss_list[category_loss_idx:]) / (
                len(dice_loss_list) - category_loss_idx
            )
            print(
                f"  {category} - loss: {category_avg_loss:.4f}, "
                f"image_loss: {category_avg_image_loss:.4f}, "
                f"focal_loss: {category_avg_focal_loss:.4f}, "
                f"dice_loss: {category_avg_dice_loss:.4f}"
            )

        avg_loss = sum(loss_list) / len(loss_list)
        avg_image_loss = sum(image_loss_list) / len(image_loss_list)
        avg_focal_loss = sum(focal_loss_list) / len(focal_loss_list)
        avg_dice_loss = sum(dice_loss_list) / len(dice_loss_list)
        return {
            "loss": avg_loss,
            "image_loss": avg_image_loss,
            "focal_loss": avg_focal_loss,
            "dice_loss": avg_dice_loss,
        }


def get_vision_transformer(
    name: str, epoch: int, device: torch.device
) -> VisionTransformer:
    model = MatchTrainer.get_trained_model(name, epoch)
    assert isinstance(model, Model)
    vision_transformer = VisionTransformer(model, device=device)
    return vision_transformer



if __name__ == "__main__":
    # config = TrainConfig()
    # trainer = MatchTrainer(config, "test4")
    trainer = MatchTrainer("test4")
    trainer.run()