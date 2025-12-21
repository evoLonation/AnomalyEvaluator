from bisect import bisect_left
from typing import Any, Callable, Iterable, Iterator, Literal, cast, override
from dataclasses_json import DataClassJsonMixin, dataclass_json
import dataclasses_json
from dataclasses import dataclass, field
import datetime
import json
from pathlib import Path
import torch.nn as nn
from torch import Tensor, cdist, device
from jaxtyping import Float, jaxtyped, Int
from torch.utils.data.sampler import RandomSampler, Sampler
from torch.utils.data import Subset
from torchvision.transforms import Compose, CenterCrop
import torch.nn.functional as F

import numpy as np
import pytz

from common.algo import pca_background_mask
from data.detection_dataset import DetectionDataset, TensorSample, TensorSampleBatch
from data.mixed import (
    MixedBatchSampler,
    MixedDataset,
    MixedInBatchSampler,
    MixedSample,
    MixedSampleBatch,
)
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
from .checkpoint import StateMapper, TrainCheckpointState
import torch
from torch.utils.data import DataLoader
from data import MVTecAD, VisA, RealIAD
from tqdm import tqdm


class SimpleAdapter(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super(SimpleAdapter, self).__init__()
        self.fc = nn.Sequential(nn.Linear(c_in, c_out, bias=False), nn.LeakyReLU())

    def forward(self, x):
        x = self.fc(x)
        return x


class AdaptedViT(VisionTransformerBase):
    def __init__(
        self,
        device: torch.device,
        adapter_layers: list[int] | None,
        shallow_adapter: int | None,
    ):
        super().__init__()
        self.vision = DINOv3VisionTransformer(device=device)
        self.adapter_layers = adapter_layers
        self.shallow_adapter = shallow_adapter
        if shallow_adapter is None:
            self.adapters = nn.ModuleList(
                [
                    VisionAdapter(embed_dim=self.vision.get_embed_dim())
                    for _ in (adapter_layers if adapter_layers is not None else [-1])
                ]
            )
        else:
            self.adapters = nn.ModuleList(
                [
                    SimpleAdapter(
                        c_in=self.vision.get_embed_dim(),
                        c_out=self.vision.get_embed_dim(),
                    )
                    for _ in range(shallow_adapter)
                ]
            )
            self._hook_handles: list[torch.utils.hooks.RemovableHandle] = []
        self.to(device)
        self.device = device
        self._alpha = 0.1

    def _register_hooks(self):
        assert self.shallow_adapter is not None
        for i in range(self.shallow_adapter):
            handle = self.vision.model.blocks[i].register_forward_hook(
                self._get_hook(self.adapters[i])
            )
            self._hook_handles.append(handle)

    def _get_hook(self, adapter: nn.Module):
        def hook(module, input, output):
            # 检查 output 是否存在 nan
            if torch.isnan(output).any():
                raise ValueError("NaN detected in transformer block output")
            adapted = adapter(output)
            adapted = (
                adapted
                * output.norm(dim=-1, keepdim=True)
                / adapted.norm(dim=-1, keepdim=True)
            )
            result = output * (1 - self._alpha) + self._alpha * adapted
            return result

        return hook

    def _remove_hooks(self):
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []

    @jaxtyped(typechecker=None)
    def forward(
        self,
        pixel_values: Float[Tensor, "N 3 H W"],
        output_layers: list[int] | None = None,
    ) -> list[Float[Tensor, "N P D"]]:
        if self.shallow_adapter is None:
            assert output_layers == self.adapter_layers
            pixel_values = pixel_values.to(self.device)
            features = self.vision(
                pixel_values=pixel_values, output_layers=output_layers
            )
            features = [adapter(f) for adapter, f in zip(self.adapters, features)]
            return features
        else:
            pixel_values = pixel_values.to(self.device)
            self._register_hooks()
            try:
                features = self.vision(
                    pixel_values=pixel_values, output_layers=output_layers
                )
            finally:
                self._remove_hooks()
            return features

    @generate_call_signature(forward)
    def __call__(self): ...

    @override
    def get_embed_dim(self) -> int:
        return self.vision.get_embed_dim()

    @override
    def get_patch_size(self) -> int:
        return self.vision.get_patch_size()

    @override
    def get_layer_num(self) -> int:
        return self.vision.get_layer_num()

    def get_trainable_parameters(self) -> Iterable[Tensor]:
        return self.adapters.parameters()


class Model(nn.Module):
    def __init__(
        self,
        device: torch.device,
        output_layers: list[int] | None,
        shallow_adapter: int | None,
        single_match: bool,
    ):
        super().__init__()
        self.vision = AdaptedViT(
            device=device,
            adapter_layers=output_layers,
            shallow_adapter=shallow_adapter,
        )
        self.device = device
        self.to(device)
        self.output_layers = output_layers
        self.single_match = single_match

    @jaxtyped(typechecker=None)
    def forward(
        self,
        images: Float[Tensor, "N1 3 H W"],
        normal_images: Float[Tensor, "N2 3 H W"],
        bgmask_thresholds: list[float | None],
    ) -> tuple[Float[Tensor, "N1"], Float[Tensor, "N1 H W"]]:
        patch_size = self.vision.get_patch_size()
        PH = images.shape[2] // patch_size
        PW = images.shape[3] // patch_size
        P = PH * PW
        N1, _, H, W = images.shape
        N2 = normal_images.shape[0]
        if self.single_match:
            assert (
                N1 == N2
            ), f"single_match=True 时需要 images 和 normal_images 数量一致，但得到 {N1} 和 {N2}"
        images = images.to(self.device)
        normal_images = normal_images.to(self.device)

        background_masks = torch.ones((images.shape[0], P), device=self.device).bool()
        if any(bgmask_thresholds):
            use_indices = [i for i, thr in enumerate(bgmask_thresholds) if thr]
            feats_list = self.vision.vision(pixel_values=images[use_indices])
            background_masks[use_indices] = pca_background_mask(
                feats_list[-1],
                (PH, PW),
                threshold=[cast(float, bgmask_thresholds[i]) for i in use_indices],
            )
        feats_list: list[Float[Tensor, "N1 P D"]] = self.vision(
            pixel_values=images, output_layers=self.output_layers
        )
        feats_list = [f / f.norm(dim=-1, keepdim=True) for f in feats_list]
        nfeats_list = self.vision(
            pixel_values=normal_images, output_layers=self.output_layers
        )
        nfeats_list = [f / f.norm(dim=-1, keepdim=True) for f in nfeats_list]
        _, _, D = feats_list[-1].shape

        scores_list = []
        scores_pixel_list = []
        for feats, nfeats in zip(feats_list, nfeats_list):
            if self.single_match:
                # 一对一匹配：images[i] 只与 normal_images[i] 计算距离
                distances_list = []
                for i in range(N1):
                    dist = torch.cdist(feats[i], nfeats[i], p=2)  # P x P
                    distances_list.append(dist)
                distances = torch.stack(distances_list, dim=0)  # N1 x P x P
                distances = distances / 2
                scores_patch = distances.min(dim=-1).values  # N1 x P
                scores_patch = scores_patch * background_masks
                scores = scores_patch.max(dim=-1).values  # N1
            else:
                # 交叉匹配：images[i] 与所有 normal_images 计算距离
                distances = (
                    torch.cdist(
                        feats.reshape(N1 * P, D), nfeats.reshape(N2 * P, D), p=2
                    )
                    .reshape(N1, P, N2, P)
                    .permute(0, 2, 1, 3)
                )  # N1 x N2 x P x P
                distances = distances / 2
                scores_patch = distances.min(dim=-1).values  # N1 x N2 x P
                scores_patch = scores_patch * background_masks.unsqueeze(1)
                scores = scores_patch.max(dim=-1).values  # N1 x N2
                scores_patch = scores_patch.mean(dim=1)  # N1 x P
                scores = torch.mean(scores, dim=-1)  # N1

            scores_patch = scores_patch.reshape(N1, PH, PW)
            scores_pixel = F.interpolate(
                scores_patch.unsqueeze(1),
                size=images.shape[2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(
                1
            )  # N1 x H x W
            scores_list.append(scores)
            scores_pixel_list.append(scores_pixel)
        # 融合多层结果
        scores = torch.stack(scores_list, dim=0).mean(dim=0)
        scores_pixel = torch.stack(scores_pixel_list, dim=0).mean(dim=0)
        return scores, scores_pixel

    @generate_call_signature(forward)
    def __call__(self): ...


mvtec_no_background_categories = ["carpet", "grid", "leather", "tile", "wood", "zipper"]
mvtec_special_threshold_categories = {
    "screw": 0.6,
}


@dataclass_json
@dataclass
class TrainConfig(BaseTrainConfig):
    use_background_mask: bool = False
    mixed_data: Literal["None", "Batch", "Single"] = "None"
    feature_layers: list[int] | None = None
    shallow_adapter: int | None = None
    single_match: bool = False

    def __post_init__(self):
        # 当 mixed_data 为 Single 时，single_match 必须为 True
        if self.mixed_data == "Single":
            assert self.single_match


def create_model(config: TrainConfig) -> Model:
    model = Model(
        device=device(config.device),
        output_layers=config.feature_layers,
        shallow_adapter=config.shallow_adapter,
        single_match=config.single_match,
    )
    for param in model.parameters():
        param.requires_grad = False
    for param in model.vision.get_trainable_parameters():
        param.requires_grad = True
    print("Trainable parameters:")
    print(f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print([name for name, p in model.named_parameters() if p.requires_grad])
    return model


class MatchTrainer(BaseTrainer[TrainConfig, Model]):
    base_dir: Path = Path("results/train_12_21")
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
        self.mixed_dataset = MixedDataset(self.dataset, self.transform)

    def get_loss(
        self,
        model: Model,
        batch: MixedSampleBatch,
        normal_batch: MixedSampleBatch,
    ):
        assert batch.categories == normal_batch.categories
        config = self.config
        use_background_mask = config.use_background_mask
        if use_background_mask:
            bgmask_thresholds: list[float | None] = []
            for category in batch.categories:
                if category in mvtec_no_background_categories:
                    bgmask_thresholds.append(None)
                elif category in mvtec_special_threshold_categories:
                    bgmask_thresholds.append(
                        mvtec_special_threshold_categories[category]
                    )
                else:
                    bgmask_thresholds.append(0.5)
        else:
            bgmask_thresholds = [None] * len(batch.categories)

        scores, maps = model(
            images=batch.images,
            normal_images=normal_batch.images,
            bgmask_thresholds=bgmask_thresholds,
        )
        image_loss = F.cross_entropy(scores, batch.labels.to(scores.device).float())
        pixel_loss_focal = focal_loss(
            torch.stack([1 - maps, maps], dim=1), batch.masks.to(maps.device)
        )
        pixel_loss_dice = binary_dice_loss(maps, batch.masks.to(maps.device))
        pixel_loss = pixel_loss_focal + pixel_loss_dice
        loss = image_loss + pixel_loss
        return loss, image_loss, pixel_loss_focal, pixel_loss_dice

    def train_one_epoch(self, epoch: int, model: Model) -> dict[str, float]:
        if self.config.mixed_data == "None" or self.config.mixed_data == "Batch":
            batch_sampler = MixedBatchSampler(
                self.mixed_dataset,
                seed=repro.get_global_seed() + epoch,
                batch_size=self.config.batch_size,
                category_random=self.config.mixed_data == "Batch",
                normal=False,
            )
            normal_sampler = MixedBatchSampler(
                self.mixed_dataset,
                seed=repro.get_global_seed() + epoch,
                batch_size=self.config.batch_size,
                category_random=self.config.mixed_data == "Batch",
                normal=True,
            )
        else:  # Single
            assert self.config.mixed_data == "Single"
            batch_sampler = MixedInBatchSampler(
                self.mixed_dataset,
                seed=repro.get_global_seed() + epoch,
                batch_size=self.config.batch_size,
                normal=False,
            )
            normal_sampler = MixedInBatchSampler(
                self.mixed_dataset,
                seed=repro.get_global_seed() + epoch,
                batch_size=self.config.batch_size,
                normal=True,
            )
        dataloader = repro.get_reproducible_dataloader(
            self.mixed_dataset,
            batch_sampler=batch_sampler,
            num_workers=4,
            collate_fn=MixedSample.collate_fn,
        )
        normal_dataloader = repro.get_reproducible_dataloader(
            self.mixed_dataset,
            batch_sampler=normal_sampler,
            num_workers=4,
            collate_fn=MixedSample.collate_fn,
        )
        now_category = ""
        for batch, normal_batch in tqdm(
            zip(dataloader, normal_dataloader),
            desc=f"Batches",
            position=1,
            leave=False,
            initial=0,
            total=len(dataloader),
        ):
            batch: MixedSampleBatch
            normal_batch: MixedSampleBatch
            if self.config.mixed_data == "None":
                for cat in batch.categories:
                    if cat != now_category:
                        print(f"Category: {cat}")
                    now_category = cat
            loss, image_loss, pixel_loss_focal, pixel_loss_dice = self.get_loss(
                model, batch, normal_batch
            )
            self.optimize_step(loss)
            self.record_loss(
                {
                    "loss": loss.item(),
                    "image_loss": image_loss.item(),
                    "focal_loss": pixel_loss_focal.item(),
                    "dice_loss": pixel_loss_dice.item(),
                }
            )
        self.print_loss()
        return self.compute_total_avg_loss()


def get_vision_transformer(name: str, epoch: int, device: torch.device) -> AdaptedViT:
    model = MatchTrainer.get_trained_model(name, epoch, mapper=mapper)
    return model.vision.to(device)


def evaluate_baseline(batch_size: int, config_eval: MuScConfig2, seed: int = 42):
    repro.init(seed)
    save_path = MatchTrainer.base_dir / "baseline"
    custom_model = DINOv3VisionTransformer(device=device(config_eval.device))
    config_eval.custom_vision_model = custom_model
    config_eval.custom_name = "DINOv3"
    config_eval.image_resize = 512
    config_eval.input_image_size = ImageSize(h=512, w=512)
    detector = MuScDetector2(config_eval)
    datasets = [MVTecAD(), VisA()]

    def namer(detector, dataset):
        name = ""
        name += f"b{batch_size}"
        name += "_" + detector.name
        name += "_" + dataset.get_name()
        name += f"_s{repro.get_global_seed()}"
        return name

    for dataset in datasets:
        evaluation_detection(
            save_path,
            detector,
            dataset,
            batch_size=batch_size,
            sampler_getter=lambda c, d: RandomSampler(
                d,
                replacement=False,
                generator=torch.Generator().manual_seed(repro.get_global_seed()),
            ),
            namer=namer,
        )


@dataclass
class EpochConfig:
    mvtec_epochs: list[int] = field(default_factory=lambda: [5, 10])
    visa_epochs: list[int] = field(default_factory=lambda: [5, 10])


def mapper(state_dict: dict[str, Any]) -> dict[str, Any]:
    new_state_dict = {}
    for key in list(state_dict.keys()):
        new_state_dict[f"vision.{key}"] = state_dict[key]
    return new_state_dict


def evaluate(
    name: str,
    config: TrainConfig,
    config_eval: MuScConfig2,
    config_epoch: EpochConfig,
    state_mapper: StateMapper | None = None,
):
    config_eval.image_resize = config.image_resize
    config_eval.input_image_size = config.centercrop
    config_eval.device = device(config.device)
    batch_size = 4
    for dataset, epochs in [
        (MVTecAD(), config_epoch.mvtec_epochs),
        (VisA(), config_epoch.visa_epochs),
    ]:
        for epoch in epochs:
            print(f"Evaluating {dataset.get_name()} for epoch {epoch}...")
            model = MatchTrainer.get_trained_model(name, epoch, mapper=state_mapper)
            config_eval.custom_vision_model = model.vision.to(device(config.device))
            config_eval.custom_name = f"ep{epoch}"
            path = MatchTrainer.gen_result_dir(name) / "evaluation"
            path = Path(path)

            def namer(detector, dataset):
                name = ""
                name += f"b{batch_size}"
                name += "_" + detector.name
                name += "_" + dataset.get_name()
                name += f"_s{repro.get_global_seed()}"
                return name

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


def train_and_evaluate(
    name: str,
    config: TrainConfig,
    config_eval: MuScConfig2,
    config_epoch: EpochConfig,
    resume: bool = False,
    state_mapper: StateMapper | None = None,
):
    if not resume:
        trainer = MatchTrainer(name=name, config=config)
    else:
        trainer = MatchTrainer(resume_name=name)
    trainer.run()
    evaluate(name, config, config_eval, config_epoch, state_mapper)


def main():
    config = TrainConfig()
    config.use_background_mask = True
    config.feature_layers = [5, 11, 17, 23]
    config.single_match = True
    config.mixed_data = "Batch"
    config_eval = MuScConfig2()
    config_eval.r_list = [1, 3, 5]
    config_eval.feature_layers = [5, 11, 17, 23]
    config_epoch = EpochConfig()
    config_epoch.mvtec_epochs = [1, 5, 10]
    config_epoch.visa_epochs = [1, 3, 5, 7, 10]
    name = "single_match_batch"
    # train_and_evaluate(name, config, config_eval, config_epoch)
    config.mixed_data = "Single"
    name = "single_match_inbatch"
    train_and_evaluate(name, config, config_eval, config_epoch)


if __name__ == "__main__":
    # config = TrainConfig()
    # trainer = MatchTrainer(config, "test4")
    # name = "default"
    # train_and_evaluate(name, config)
    # name = "default_new"
    # name = "background_mixed"
    # train_and_evaluate(name, config, MuScConfig2())
    main()
