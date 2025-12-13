from typing import Iterable, override
from dataclasses_json import DataClassJsonMixin, dataclass_json
import dataclasses_json
from dataclasses import dataclass, field
import datetime
import json
from pathlib import Path
import torch.nn as nn
from torch import Tensor, cdist, device
from jaxtyping import Float, jaxtyped, Int
from torch.utils.data.sampler import RandomSampler
from torch.utils.data import Subset
from torchvision.transforms import Compose, CenterCrop
import torch.nn.functional as F

import numpy as np
import pytz

from common.algo import pca_background_mask
from data.detection_dataset import TensorSample, TensorSampleBatch
from data.utils import ImageResize, ImageSize, Transform
from data.base import Dataset
from evaluator.musc2 import MuScConfig2, MuScDetector2
from evaluator.trainer import GlobalTrainState
from evaluator.vit import VisionTransformerBase
from evaluator.dinov3 import DINOv3VisionTransformer
from evaluator.evaluation import evaluation_detection
from evaluator.image_normalize import DINO_NORMALIZE
from evaluator.loss import binary_dice_loss, focal_loss
import evaluator.reproducibility as repro
from evaluator.train2 import VisionAdapter
from evaluator.trainer import BaseTrainer, BaseTrainConfig
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
        use_background_mask: bool = True,
        background_mask_threshold: float = 0.5,
    ) -> tuple[Float[Tensor, "N1"], Float[Tensor, "N1 H W"]]:
        patch_size = self.vision.get_patch_size()
        PH = images.shape[2] // patch_size
        PW = images.shape[3] // patch_size
        images = images.to(self.device)
        normal_images = normal_images.to(self.device)
        features = self.vision(pixel_values=images)
        background_masks = None
        if use_background_mask:
            background_masks = pca_background_mask(
                features, (PH, PW), threshold=background_mask_threshold
            )
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

        # 使用背景掩码过滤
        if background_masks is not None:
            scores_patch = scores_patch * background_masks.unsqueeze(1)

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


def create_model(config: BaseTrainConfig) -> Model:
    model = Model(device=device(config.device))
    for param in model.parameters():
        param.requires_grad = False
    for param in model.adapter.parameters():
        param.requires_grad = True
    print("Trainable parameters:")
    print(f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print([name for name, p in model.named_parameters() if p.requires_grad])
    return model


mvtec_no_background_categories = ["carpet", "grid", "leather", "tile", "wood", "zipper"]
mvtec_special_threshold_categories = {
    "screw": 0.6,
}


@dataclass_json
@dataclass
class TrainConfig(BaseTrainConfig):
    use_background_mask: bool = True


class MatchTrainer(BaseTrainer[TrainConfig, Model]):
    model_type = Model
    config_type = TrainConfig

    @classmethod
    def setup_model(cls, config: TrainConfig) -> Model:
        model = create_model(config)
        return model

    @classmethod
    def setup_optimizer(
        cls, config: TrainConfig, model: Model
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

    def train_one_epoch(self, epoch: int, model: Model) -> dict[str, float]:
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
                use_background_mask = True
                if category in mvtec_no_background_categories:
                    use_background_mask = False
                if category in mvtec_special_threshold_categories:
                    background_mask_threshold = mvtec_special_threshold_categories[
                        category
                    ]
                else:
                    background_mask_threshold = 0.5
                scores, maps = model(
                    images=batch.images,
                    normal_images=normal_batch.images,
                    use_background_mask=use_background_mask,
                    background_mask_threshold=background_mask_threshold,
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


def train_and_evaluate(name: str, config: TrainConfig, resume: bool = False):
    if not resume:
        trainer = MatchTrainer(name=name, config=config)
    else:
        trainer = MatchTrainer(resume_name=name)
    trainer.run()
    config_eval = MuScConfig2()
    config_eval.image_resize = config.image_resize
    config_eval.input_image_size = config.centercrop
    config_eval.device = device(config.device)
    batch_size = 4
    # 每 5 个 epoch 评估一次, 包括最后一个 epoch
    for epoch in range(0, config.num_epochs, 5):
        vision_transformer = get_vision_transformer(
            name=name,
            epoch=epoch,
            device=device(config.device),
        )
        config_eval.custom_vision_model = vision_transformer
        config_eval.custom_name = f"ep{epoch}"
        path = trainer.base_dir / "evaluation"
        path = Path(path)

        def namer(detector, dataset):
            name = ""
            name += f"b{batch_size}"
            name += "_" + detector.name
            name += "_" + dataset.get_name()
            name += f"_s{repro.get_global_seed()}"
            return name

        for dataset in [MVTecAD(), VisA()]:
            repro.init(config.seed)
            detector = MuScDetector2(
                config_eval,
            )
            evaluation_detection(
                path=path,
                detector=detector,
                dataset=dataset,
                batch_size=batch_size,
                sampler_getter=lambda c, d: RandomSampler(
                    d,
                    replacement=False,
                    generator=torch.Generator().manual_seed(repro.get_global_seed()),
                ),
                save_anomaly_score=True,
                namer=namer,
            )


if __name__ == "__main__":
    # config = TrainConfig()
    # trainer = MatchTrainer(config, "test4")
    # name = "default"
    config = TrainConfig()
    # train_and_evaluate(name, config)
    name = "use_background_mask"
    config.use_background_mask = True
    train_and_evaluate(name, config)
