from dataclasses import dataclass, field
import time
from typing import Any, Callable, Literal, cast
import cv2
import numpy as np
from torch import adaptive_avg_pool1d, cdist, equal, layer_norm, nn, tensor
import torch.nn.functional as F
from jaxtyping import Float, Bool, jaxtyped, Int
import torch
from torch import Tensor
from torchvision.transforms import CenterCrop, Compose, Normalize

from common.algo import (
    aggregate_shifted_features,
    compute_patch_offset_distance,
    get_avg_pool_features,
    shift_image,
)
from data.base import Dataset
from data.utils import (
    ImageResize,
    ImageSize,
    Transform,
)
from evaluator.clip import generate_call_signature
from evaluator.detector import DetectionResult, TensorDetector
from evaluator.dinov2 import DINOv2VisionTransformer
from evaluator.dinov3 import DINOv3VisionTransformer
from evaluator.image_normalize import CLIP_NORMALIZE, DINO_NORMALIZE
from evaluator.openclip import CLIPVisionTransformer, create_vision_transformer
import evaluator.reproducibility as repro
from evaluator.vit import VisionTransformerBase


@dataclass
class MuScConfig2:
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_resize: ImageResize = 518
    input_image_size: ImageSize = field(default_factory=lambda: ImageSize.square(518))
    feature_layers: list[int] = field(default_factory=lambda: [5, 11, 17, 23])
    r_list: list[int] = field(default_factory=lambda: [1, 3, 5])
    topmin_min: float = 0.0
    topmin_max: float = 0.3

    is_dino: bool = False
    is_dinov3: bool = False
    custom_vision_model: VisionTransformerBase | None = None
    custom_name: str | None = None
    detail_result: bool = False
    # 是否启用shift augmentation: 对每张图片生成3种平移增强（右移、上移、右上移各patch_size/2）
    shift_augmentation: bool = False
    shift_aggregation: bool = False


class MuSc(nn.Module):
    def __init__(self, config: MuScConfig2):
        super().__init__()
        self.r_list = config.r_list
        self.feature_layers = config.feature_layers
        self.topmin_min = config.topmin_min
        self.topmin_max = config.topmin_max
        self.input_H, self.input_W = config.input_image_size.hw()
        self.device = config.device
        self.config = config

        if config.custom_vision_model is not None:
            self.vision_encoder = config.custom_vision_model
        elif config.is_dino:
            self.vision_encoder = DINOv2VisionTransformer(model_name="dinov2_vitl14")
            self.feature_layers = [-1]
        elif config.is_dinov3:
            self.vision_encoder = DINOv3VisionTransformer(model_name="dinov3_vitl16")
            self.feature_layers = [-1]
        else:
            self.vision_encoder = create_vision_transformer(
                image_size=ImageSize(h=self.input_H, w=self.input_W),
                device=config.device,
            )
        assert isinstance(self.vision_encoder, VisionTransformerBase)
        self.embed_dim = self.vision_encoder.get_embed_dim()
        self.patch_size = self.vision_encoder.get_patch_size()
        assert (
            self.input_H % self.patch_size == 0 and self.input_W % self.patch_size == 0
        )
        self.patch_H = self.input_H // self.patch_size
        self.patch_W = self.input_W // self.patch_size
        self.grid_size = (self.patch_H, self.patch_W)
        self.patch_num = self.patch_H * self.patch_W

        self.detail_result = config.detail_result

        self.shift_augmentation = config.shift_augmentation
        self.shift_aggregation = config.shift_aggregation
        # shift augmentation: 3种平移方式（右移、下移、右下移）
        self.shift_offsets = [
            (0, self.patch_size // 2),  # 右移
            (self.patch_size // 2, 0),  # 下移
            (self.patch_size // 2, self.patch_size // 2),  # 右下移
        ]

        self.ref_features_rlist: list[Float[Tensor, "L Ref P D"]] | None = None

    @jaxtyped(typechecker=None)
    def set_ref_features(
        self,
        pixel_values: Float[Tensor, "N 3 {self.input_H} {self.input_W}"],
    ):
        self.ref_features_rlist = []
        features = self.get_features(pixel_values)
        # features: Float[Tensor, "L N P D"] or "L N*4 P D" when shift augmentation enabled
        for r in self.r_list:
            r_features = self.get_r_features(features, r)
            self.ref_features_rlist.append(r_features)

    def get_ref_features(self):
        return self.ref_features_rlist

    def clear_ref_features(self):
        self.ref_features_rlist = None

    @jaxtyped(typechecker=None)
    def get_features(
        self,
        pixel_values: Float[Tensor, "N 3 {self.input_H} {self.input_W}"],
    ) -> Float[Tensor, "L N2 P D"]:
        pixel_values = pixel_values.to(self.device)
        if self.shift_augmentation or self.shift_aggregation:
            augmented_images = []
            for img in pixel_values:
                augmented_images.append(img)  # 原图
                for dy, dx in self.shift_offsets:
                    shifted = shift_image(img.unsqueeze(0), dx, dy)
                    augmented_images.append(shifted.squeeze(0))
            pixel_values = torch.stack(augmented_images, dim=0)  # N*4 x 3 x H x W
        features = self.vision(pixel_values)
        if self.shift_aggregation:
            features: Float[Tensor, "L N 4 P D"] = features.view(
                len(self.feature_layers), -1, 4, self.patch_num, self.embed_dim
            )
            for i in range(features.shape[1]):
                features[:, i, 0, :, :] = aggregate_shifted_features(
                    features[:, i, :, :, :], grid_size=self.grid_size
                )
            features: Float[Tensor, "L N P D"] = features[:, :, 0, :, :]
        return features

    def declare_context(self, axis: str, length: int):
        _: Bool[Tensor, axis] = torch.empty((length,), dtype=torch.bool)

    @dataclass
    class Result:
        scores: Float[Tensor, "B"]
        scores_pixel: Float[Tensor, "B H W"]

    @dataclass
    class ResultWithDetail(Result):
        # 每个图像对其他图像的 patch 级最近邻索引
        min_indices: Int[Tensor, "B L R Ref P"]
        max_indices_image_level: Int[Tensor, "B"]
        # 对于 patch 的所有分数中 topkmin 的那几个的图像索引
        topmink_indices: Int[Tensor, "B L R topmink P"]
        # 对于 patch 的所有分数中 topkmin 的那几个的分数
        topmink_scores: Float[Tensor, "B L R topmink P"]

    @jaxtyped(typechecker=None)
    def forward(
        self,
        pixel_values: Float[Tensor, "B 3 {self.input_H} {self.input_W}"],
    ) -> Result:
        self.declare_context("B", pixel_values.shape[0])
        self.declare_context("H", self.input_H)
        self.declare_context("W", self.input_W)
        self.declare_context("P", self.patch_num)
        self.declare_context("PH", self.patch_H)
        self.declare_context("PW", self.patch_W)
        self.declare_context("L", len(self.feature_layers))
        self.declare_context("R", len(self.r_list))
        self.declare_context("D", self.embed_dim)

        original_batch_size = pixel_values.shape[0]
        features = self.get_features(pixel_values)
        # features: Float[Tensor, "L B P D"] or "L B*4 P D" when shift augmentation enabled
        scores_list = []
        min_indices_list = []
        topmink_indices_list = []
        topmink_scores_list = []
        with jaxtyped("context"):
            for r_i, r in enumerate(self.r_list):
                r_features: Float[Tensor, "L B P D"] = self.get_r_features(features, r)
                ref_features = None
                if self.ref_features_rlist is not None:
                    ref_features = self.ref_features_rlist[r_i]
                r_scores, r_min_indices, r_topmink_indices, r_topmink_scores = self.MSM(
                    r_features, const_ref_features=ref_features
                )
                r_scores: Float[Tensor, "L B P"]
                r_min_indices: Int[Tensor, "L B P Ref"]
                r_topmink_indices: Int[Tensor, "L B P topmink"]
                r_topmink_scores: Float[Tensor, "L B P topmink"]
                if self.detail_result:
                    min_indices_list.append(r_min_indices)
                    topmink_indices_list.append(r_topmink_indices)
                    topmink_scores_list.append(r_topmink_scores)
                r_scores: Float[Tensor, "B P"] = torch.mean(r_scores, dim=0)
                scores_list.append(r_scores)
            if self.detail_result:
                min_indices: Int[Tensor, "R L B P Ref"] = torch.stack(
                    min_indices_list, dim=0
                )
                min_indices: Int[Tensor, "B L R Ref P"] = min_indices.permute(
                    2, 1, 0, 4, 3
                )
                topmink_indices: Int[Tensor, "R L B P topmink"] = torch.stack(
                    topmink_indices_list, dim=0
                )
                topmink_indices: Int[Tensor, "B L R topmink P"] = (
                    topmink_indices.permute(2, 1, 0, 4, 3)
                )
                topmink_scores: Float[Tensor, "R L B P topmink"] = torch.stack(
                    topmink_scores_list, dim=0
                )
                topmink_scores: Float[Tensor, "B L R topmink P"] = (
                    topmink_scores.permute(2, 1, 0, 4, 3)
                )
            scores = torch.mean(torch.stack(scores_list, dim=0), dim=0)

        if self.shift_augmentation:
            # 将增强后的结果聚合回原图
            scores = scores.view(original_batch_size, 4, -1).mean(dim=1)  # B x P

        scores: Float[Tensor, "B P"]
        scores_image_level, max_indices_image_level = torch.max(scores, dim=1)
        scores_image_level: Float[Tensor, "B"]
        max_indices_image_level: Int[Tensor, "B"]

        final_scores: Float[Tensor, "B"] = scores_image_level
        scores_patch: Float[Tensor, "B PH PW"] = scores.view(
            -1, self.patch_H, self.patch_W
        )
        scores_pixel: Float[Tensor, "B H W"] = torch.nn.functional.interpolate(
            scores_patch.unsqueeze(1),
            size=(self.input_H, self.input_W),
            mode="bilinear",
            align_corners=True,
        ).squeeze(1)
        if self.detail_result:
            return self.ResultWithDetail(
                final_scores,
                scores_pixel,
                cast(Tensor, min_indices),  # type: ignore
                max_indices_image_level,
                cast(Tensor, topmink_indices),  # type: ignore
                cast(Tensor, topmink_scores),  # type: ignore
            )
        else:
            return self.Result(
                final_scores,
                scores_pixel,
            )

    @generate_call_signature(forward)
    def __call__(self): ...

    @jaxtyped(typechecker=None)
    def vision(
        self,
        pixel_values: Float[Tensor, "B 3 H W"],
    ) -> Float[Tensor, "L B P D"]:
        features_list = self.vision_encoder(pixel_values, self.feature_layers)
        return torch.stack(features_list, dim=0)

    @jaxtyped(typechecker=None)
    def get_r_features(
        self,
        features: Float[Tensor, "L XN P D"],
        r: int,
    ) -> Float[Tensor, "L XN P D"]:
        r_features_list = []
        for l_features in features:
            r_l_features = get_avg_pool_features(
                l_features, grid_size=self.grid_size, r=r
            )
            r_features_list.append(r_l_features)
        r_features: Float[Tensor, "L XN P D"] = torch.stack(r_features_list, dim=0)
        r_features /= r_features.norm(dim=-1, keepdim=True)
        return r_features

    @jaxtyped(typechecker=None)
    def MSM(
        self,
        features: Float[Tensor, "L B P D"],
        const_ref_features: Float[Tensor, "L Ref P D"] | None,
    ) -> tuple[
        Float[Tensor, "L B P"],
        Int[Tensor, "L B P Ref"],
        Int[Tensor, "L B P topmink"],
        Float[Tensor, "L B P topmink"],
    ]:
        scores_list = []
        min_indices_list = []
        topmink_indices_list = []
        topmink_scores_list = []
        for i in range(features.shape[1]):
            feature = features[:, i, ...]
            if const_ref_features is not None:
                ref_features = const_ref_features
            elif self.shift_augmentation:
                left = i // 4 * 4
                ref_features = torch.cat(
                    [features[:, :left, ...], features[:, left + 4 :, ...]], dim=1
                )
            else:
                ref_features = torch.cat(
                    [features[:, :i, ...], features[:, i + 1 :, ...]], dim=1
                )
            scores, indices, topmink_indices, topmink_scores = self.compute_score(
                feature=feature,
                ref_features=ref_features,
                patch_coords=None,
                ref_patch_coords=None,
                coord_factor=0.0,
                ref_match_indices=None,
            )
            scores_list.append(scores)
            min_indices_list.append(indices)
            topmink_indices_list.append(topmink_indices)
            topmink_scores_list.append(topmink_scores)
        return (
            torch.stack(scores_list, dim=1),
            torch.stack(min_indices_list, dim=1),
            torch.stack(topmink_indices_list, dim=1),
            torch.stack(topmink_scores_list, dim=1),
        )

    @jaxtyped(typechecker=None)
    def compute_distance(
        self,
        features: Float[Tensor, "X P D"],
        ref_features: Float[Tensor, "X Ref P D"],
    ) -> Float[Tensor, "X P Ref P"]:
        self.declare_context("P", features.shape[1])
        self.declare_context("Ref", ref_features.shape[1])
        ref_features_: Float[Tensor, "X Ref*P D"] = ref_features.view(
            ref_features.shape[0], -1, ref_features.shape[3]
        )
        distances: Float[Tensor, "X P Ref*P"] = cdist(features, ref_features_, p=2)
        distances: Float[Tensor, "X P Ref P"] = distances.view(
            *distances.shape[0:2], -1, self.patch_num
        )
        return distances

    @jaxtyped(typechecker=None)
    def compute_score(
        self,
        feature: Float[Tensor, "X P D"],
        ref_features: Float[Tensor, "X Ref P D"],
        patch_coords: Float[Tensor, "X P 2"] | None,
        ref_patch_coords: Float[Tensor, "X Ref P 2"] | None,
        coord_factor: float,
        ref_match_indices: Int[Tensor, "X P Ref"] | None,
    ) -> tuple[
        Float[Tensor, "X P"],
        Int[Tensor, "X P Ref"],
        Int[Tensor, "X P topmink"],
        # 对于每个 patch，选择了哪几个图像的匹配 patch 作为其分数
        Float[Tensor, "X P topmink"],
    ]:
        distances: Float[Tensor, "X P Ref P"] = self.compute_distance(
            feature, ref_features
        )
        if patch_coords is not None and ref_patch_coords is not None:
            patch_distances: Float[Tensor, "X P Ref P"] = self.compute_distance(
                patch_coords, ref_patch_coords
            )
            distances += patch_distances * coord_factor

        if ref_match_indices is not None:
            match_indices = ref_match_indices
        else:
            match_indices: Int[Tensor, "X P Ref"] = torch.argmin(distances, dim=-1)
        scores: Float[Tensor, "X P Ref"] = distances.gather(
            dim=-1, index=match_indices.unsqueeze(-1)
        ).squeeze(-1)
        t_max = max(1, int(self.topmin_max * scores.shape[-1]))
        t_min = min(t_max - 1, int(self.topmin_min * scores.shape[-1]))
        t = t_max - t_min
        scores_topkmax, scores_topkmax_indices = torch.topk(
            scores, k=t_max, largest=False, dim=-1, sorted=True
        )
        scores_topkmax: Float[Tensor, f"X P {t_max}"]
        scores_topkmax_indices: Int[Tensor, f"X P {t_max}"]
        scores_topminmax: Float[Tensor, f"X P {t}"] = scores_topkmax[:, :, t_min:t_max]
        scores_topminmax_indices: Int[Tensor, f"X P {t}"] = scores_topkmax_indices[
            :, :, t_min:t_max
        ]
        scores_final: Float[Tensor, "X P"] = torch.mean(scores_topminmax, dim=-1)
        return scores_final, match_indices, scores_topminmax_indices, scores_topminmax


class MuScDetector2(TensorDetector):
    def __init__(
        self,
        config: MuScConfig2,
        const_features: bool = False,
        train_data: (
            Callable[[str, Transform], Dataset[Float[torch.Tensor, "C H W"]]] | None
        ) = None,
    ):
        """
        const_feature 为 True 时, 构建固定特征库：
            若提供了 train_data，则随机抽样 batch_size - 1 张图像作为参考集
            若没有提供， 在每个类别第一个 batch 中的前 batch_size - 1 张图像作为参考集
        """
        self.model = MuSc(config)
        default_config = MuScConfig2()

        name = "MuSc2"
        name += f"(r{''.join([str(r) for r in config.r_list])})"
        name += f"(l{'-'.join([str(l) for l in config.feature_layers])})"
        name += f"(t{config.topmin_min}-{config.topmin_max})"
        if config.is_dino:
            name += "(dino)"
        if config.is_dinov3:
            name += "(dinov3)"
        if config.custom_vision_model is not None:
            assert config.custom_name is not None
            name += f"({config.custom_name})"
        if config.shift_augmentation:
            name += "(shift)"
        if config.shift_aggregation:
            name += "(shift-agg)"
        if const_features:
            if train_data is not None:
                name += "(train)"
            else:
                name += "(const)"

        self.const_feature = const_features
        self.train_data = train_data
        self.last_class_name = "??"

        if config.is_dino or config.is_dinov3:
            normalize = DINO_NORMALIZE
        else:
            normalize = CLIP_NORMALIZE
        image_transform = Compose([CenterCrop(config.input_image_size.hw()), normalize])
        mask_transform = CenterCrop(config.input_image_size.hw())
        super().__init__(
            name=name,
            transform=Transform(
                resize=config.image_resize,
                image_transform=image_transform,
                mask_transform=mask_transform,
            ),
        )
        self.train_indices = None

    def get_train_indices(self):
        return self.train_indices

    def __call__(
        self, images: Float[torch.Tensor, "N C H W"], class_name: str
    ) -> DetectionResult:
        self.model.eval()
        with torch.no_grad():
            if self.last_class_name != class_name and self.const_feature:
                self.last_class_name = class_name
                if self.train_data is not None:
                    train_tensor_dataset = self.train_data(class_name, self.transform)
                    g = torch.Generator().manual_seed(repro.get_global_seed())
                    self.train_indices = torch.randperm(
                        len(train_tensor_dataset), generator=g
                    )[: images.shape[0] - 1].tolist()
                    subset = torch.stack(
                        [train_tensor_dataset[i] for i in self.train_indices]
                    )
                    self.model.set_ref_features(subset)
                else:
                    self.model.set_ref_features(images[: images.shape[0] - 1, ...])
            single_image = False
            if images.shape[0] == 1:
                single_image = True
                images = torch.cat([images, images], dim=0)
            result = self.model(images)
            if isinstance(result, MuSc.ResultWithDetail):
                patch_distances = (
                    compute_patch_offset_distance(
                        result.min_indices.reshape(-1, result.min_indices.shape[-1]),
                        self.model.grid_size,
                    )
                    .view(result.min_indices.shape[0], -1)
                    .mean(dim=-1)
                )
            else:
                patch_distances = torch.zeros_like(result.scores)
            if single_image:
                result.scores = result.scores[:1, ...]
                result.scores_pixel = result.scores_pixel[:1, ...]
                if isinstance(result, MuSc.ResultWithDetail):
                    result.min_indices = result.min_indices[:1, :, :, 0:0, ...]
                    result.max_indices_image_level = result.max_indices_image_level[
                        :1, ...
                    ]
                    result.topmink_indices = result.topmink_indices[:1, :, :, 0:0, ...]
                    result.topmink_scores = result.topmink_scores[:1, :, :, 0:0, ...]
        # print(f"Avg patch distance: {patch_distances.mean().item():.4f}")
        detection_result = DetectionResult(
            pred_scores=result.scores,
            anomaly_maps=result.scores_pixel,
            patch_distances=patch_distances,
        )
        if isinstance(result, MuSc.ResultWithDetail):
            detection_result.other = (
                result.min_indices,
                result.max_indices_image_level,
                result.topmink_indices,
                result.topmink_scores,
            )
        return detection_result
