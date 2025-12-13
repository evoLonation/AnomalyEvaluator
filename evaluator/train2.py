from typing import Iterable
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
from torchvision.transforms import Compose, CenterCrop
import torch.nn.functional as F

import numpy as np
import pytz

from data.detection_dataset import TensorSample, TensorSampleBatch
from data.utils import ImageResize, ImageSize, Transform
from evaluator.dinov3 import DINOv3VisionTransformer
from evaluator.evaluation import evaluation_detection
from evaluator.image_normalize import DINO_NORMALIZE
import evaluator.reproducibility as repro
from evaluator.trainer import BaseTrainer, BaseTrainConfig
from common.utils import generate_call_signature
from .clip import CLIP, CLIPConfig
from .checkpoint import TrainCheckpointState
import torch
from torch.utils.data import DataLoader
from data import MVTecAD, VisA, RealIAD
from tqdm import tqdm


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

    def forward(self, x) -> Tensor:
        x = self.fc1(x)
        y = self.fc2(x)
        return y

    @generate_call_signature(forward)
    def __call__(self): ...


class DINOv3Matcher(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.vision = DINOv3VisionTransformer(device=device)
        self.adapter = VisionAdapter(embed_dim=self.vision.get_embed_dim())
        self.to(device)

        self.device = device

    @jaxtyped(typechecker=None)
    def forward(self, images: Float[Tensor, "N 3 H W"]) -> Float[Tensor, "N P D"]:
        images = images.to(self.device)
        features = self.vision(pixel_values=images)
        features = self.adapter(features)
        return features  # [N, P, D]


patch_coord_distances_dict: dict[tuple[int, int], Float[Tensor, "P P"]] = {}


@jaxtyped(typechecker=None)
def compute_weighted_patch_distance(
    softmax_sims: Float[Tensor, "N Ref P P"],
    grid_size: tuple[int, int],
) -> Float[Tensor, "N Ref P2"]:
    N, REF, _, P = softmax_sims.shape
    PH, PW = grid_size
    if grid_size not in patch_coord_distances_dict:
        patch_indices = torch.arange(softmax_sims.shape[-1], device=softmax_sims.device)
        patch_coords = torch.stack(
            [patch_indices // PH, patch_indices % PW],
            dim=-1,
        )
        patch_coord_distances: Float[Tensor, "P P"] = torch.cdist(
            patch_coords.float(), patch_coords.float(), p=2
        )
        patch_coord_distances_dict[grid_size] = patch_coord_distances
    else:
        patch_coord_distances = patch_coord_distances_dict[grid_size]

    max_indices: Int[Tensor, "N Ref P"] = torch.max(softmax_sims, dim=-1).indices
    # 每个 patch 的最大匹配 patch 与所有其他 patch 的坐标距离
    max_indices_distances: Float[Tensor, "N Ref P P"] = patch_coord_distances[
        max_indices
    ]
    softmax_sims = softmax_sims.reshape(N, REF, PH, PW, P)
    max_indices_distances = max_indices_distances.reshape(N, REF, PH, PW, P)
    softmax_sims_view = softmax_sims[:, :, 1 : PH - 1, 1 : PW - 1, :]
    max_indices_distances_up = max_indices_distances[:, :, 2:PH, 1 : PW - 1, :]
    max_indices_distances_down = max_indices_distances[:, :, 0 : PH - 2, 1 : PW - 1, :]
    max_indices_distances_left = max_indices_distances[:, :, 1 : PH - 1, 2:PW, :]
    max_indices_distances_right = max_indices_distances[:, :, 1 : PH - 1, 0 : PW - 2, :]
    weighted_distance = torch.sum(
        (
            max_indices_distances_up
            + max_indices_distances_down
            + max_indices_distances_left
            + max_indices_distances_right
        )
        * softmax_sims_view
        / 4,
        dim=-1,
    )
    return weighted_distance.reshape(N, REF, -1)


@jaxtyped(typechecker=None)
def compute_weighted_patch_distance_2(
    sims: Float[Tensor, "N Ref P P"],
    softmax_sims: Float[Tensor, "N Ref P P"],
    grid_size: tuple[int, int],
    neighbor_topk: int = 5,
):
    N, REF, _, P = softmax_sims.shape
    PH, PW = grid_size
    K = neighbor_topk
    if grid_size not in patch_coord_distances_dict:
        patch_indices = torch.arange(softmax_sims.shape[-1], device=softmax_sims.device)
        patch_coords = torch.stack(
            [patch_indices // PH, patch_indices % PW],
            dim=-1,
        )
        patch_coord_distances: Float[Tensor, "P P"] = torch.cdist(
            patch_coords.float(), patch_coords.float(), p=2
        )
        patch_coord_distances_dict[grid_size] = patch_coord_distances
    else:
        patch_coord_distances = patch_coord_distances_dict[grid_size]

    # 每个 patch 的所有匹配 patch 的权重
    weights = softmax_sims
    topk_weights, topk_indices = torch.topk(sims, k=K, dim=-1)  # [N Ref P K]
    # 每个 patch 的 top-k 匹配 patch 的权重
    topk_weights: Float[Tensor, "N Ref P K"] = torch.softmax(topk_weights, dim=-1)
    # 每个 patch 的 top-k 匹配 patch 与所有其他 patch 的坐标距离
    topk_distances: Float[Tensor, "N Ref P K P"] = patch_coord_distances[topk_indices]
    weights = weights.reshape(N, REF, PH, PW, P).reshape(-1, PH, PW, P)
    topk_weights = topk_weights.reshape(N, REF, PH, PW, K).reshape(-1, PH, PW, K)
    topk_distances = topk_distances.reshape(N, REF, PH, PW, K, P).reshape(
        -1, PH, PW, K, P
    )
    center_weights = weights[:, 1 : PH - 1, 1 : PW - 1, :]  # [N*Ref (PH-2) (PW-2) P]
    weighted_distances_list: list[Tensor] = []
    for x_start, y_start in [
        (0, 1),  # left
        (2, 1),  # right
        (1, 0),  # up
        (1, 2),  # down
    ]:
        neighbor_topk_weights = topk_weights[
            :, x_start : x_start + PH - 2, y_start : y_start + PW - 2, :
        ]  # [N*Ref (PH-2) (PW-2) K]
        neighbor_topk_distances = topk_distances[
            :, x_start : x_start + PH - 2, y_start : y_start + PW - 2, :, :
        ]  # [N*Ref (PH-2) (PW-2) K P]
        weighted_distance = torch.sum(
            neighbor_topk_distances
            * neighbor_topk_weights.unsqueeze(-1)
            * center_weights.unsqueeze(-2),
            dim=-2,
        )  # [N*Ref (PH-2) (PW-2) P]
        weighted_distances_list.append(weighted_distance)
    total_weighted_distance: Tensor = sum(weighted_distances_list)  # type: ignore
    return total_weighted_distance.reshape(N, REF, -1)


def create_model(config: BaseTrainConfig) -> DINOv3Matcher:
    model = DINOv3Matcher(device=config.device)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.adapter.parameters():
        param.requires_grad = True
    print("Trainable parameters:")
    print(f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print([name for name, p in model.named_parameters() if p.requires_grad])
    return model


unaligned_classes = ["hazelnut", "metal_nut", "screw"]


class MatchTrainer(BaseTrainer):
    @classmethod
    def setup_model(cls, config: BaseTrainConfig) -> nn.Module:
        model = create_model(config)
        return model

    @classmethod
    def setup_optimizer(
        cls, config: BaseTrainConfig, model: nn.Module
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
        categories = self.dataset.get_categories()
        assert set(unaligned_classes).issubset(categories)
        categories = sorted(list(set(categories) - set(unaligned_classes)))
        self.categories = categories

    def train_one_epoch(self, epoch: int, model: nn.Module) -> dict[str, float]:
        assert isinstance(model, DINOv3Matcher)
        loss_list = []
        max_loss_list = []
        coverage_loss_list = []
        distance_loss_list = []
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
            patch_size = model.vision.get_patch_size()
            PH = self.config.centercrop.h // patch_size
            PW = self.config.centercrop.w // patch_size
            P = PH * PW
            D = model.vision.get_embed_dim()
            category_loss_idx = len(loss_list)
            for batch in tqdm(
                dataloader,
                desc=f"{category}",
                position=2,
                leave=False,
            ):
                batch: TensorSampleBatch
                N = len(batch.images)
                if N == 1:
                    continue
                features: Float[Tensor, "N P D"] = model(images=batch.images)
                features = features / features.norm(dim=-1, keepdim=True, p=2)
                sims_list = []
                for i in range(len(features)):
                    feat: Float[Tensor, "P D"] = features[i]
                    ref_feats: Float[Tensor, "Ref P D"] = torch.cat(
                        [features[:i, ...], features[i + 1 :, ...]], dim=0
                    )
                    # ref_sims: Float[Tensor, "P Ref P"] = F.cosine_similarity(
                    #     feat.unsqueeze(1), ref_feats.view(-1, D).unsqueeze(0), dim=-1
                    # ).view(P, N - 1, P)
                    ref_sims: Float[Tensor, "P Ref P"] = torch.cdist(
                        feat, ref_feats.view(-1, D), p=2
                    ).view(P, N - 1, P)
                    # cos = 1 - d^2/2
                    ref_sims = 1 - ref_sims * ref_sims / 2
                    ref_sims = (ref_sims + 1) / 2
                    sims_list.append(ref_sims)
                sims: Float[Tensor, "N P Ref P"] = torch.stack(sims_list, dim=0)
                sims: Float[Tensor, "N Ref P P"] = sims.permute(0, 2, 1, 3)
                softmax_sims: Float[Tensor, "N Ref P P"] = torch.softmax(sims, dim=-1)
                max_sims: Float[Tensor, "N Ref P"] = (
                    torch.max_pool1d(softmax_sims.reshape(-1, P), kernel_size=P)
                    .squeeze(-1)
                    .reshape(N, N - 1, P)
                )
                target_coverage: Float[Tensor, "N Ref P"] = torch.sum(
                    softmax_sims, dim=-2
                )
                weighted_distances: Float[Tensor, "N Ref P2"] = (
                    compute_weighted_patch_distance_2(
                        sims=sims, softmax_sims=softmax_sims, grid_size=(PH, PW)
                    )
                )

                ones1 = torch.ones_like(max_sims)
                ones2 = torch.ones_like(weighted_distances)
                max_loss = F.mse_loss(max_sims, ones1)
                coverage_loss = F.mse_loss(target_coverage, ones1)
                distance_loss = F.mse_loss(weighted_distances, ones2)
                loss = max_loss + coverage_loss + distance_loss
                self.optimize_step(loss)
                loss_list.append(loss.item())
                max_loss_list.append(max_loss.item())
                coverage_loss_list.append(coverage_loss.item())
                distance_loss_list.append(distance_loss.item())
            category_avg_loss = sum(loss_list[category_loss_idx:]) / (
                len(loss_list) - category_loss_idx
            )
            category_avg_max_loss = sum(max_loss_list[category_loss_idx:]) / (
                len(max_loss_list) - category_loss_idx
            )
            category_avg_coverage_loss = sum(coverage_loss_list[category_loss_idx:]) / (
                len(coverage_loss_list) - category_loss_idx
            )
            category_avg_distance_loss = sum(distance_loss_list[category_loss_idx:]) / (
                len(distance_loss_list) - category_loss_idx
            )
            print(
                f"  {category} - loss: {category_avg_loss:.4f}, "
                f"max_loss: {category_avg_max_loss:.4f}, "
                f"coverage_loss: {category_avg_coverage_loss:.4f}, "
                f"distance_loss: {category_avg_distance_loss:.4f}"
            )

        avg_loss = sum(loss_list) / len(loss_list)
        avg_max_loss = sum(max_loss_list) / len(max_loss_list)
        avg_coverage_loss = sum(coverage_loss_list) / len(coverage_loss_list)
        avg_distance_loss = sum(distance_loss_list) / len(distance_loss_list)
        return {
            "loss": avg_loss,
            "max_loss": avg_max_loss,
            "coverage_loss": avg_coverage_loss,
            "distance_loss": avg_distance_loss,
        }


if __name__ == "__main__":
    config = BaseTrainConfig()
    trainer = MatchTrainer(config, "test3")
    trainer.run()
