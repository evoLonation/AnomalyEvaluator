from dataclasses import dataclass, field
from typing import Callable
from torch import adaptive_avg_pool1d, cdist, equal, layer_norm, nn, tensor
import torch.nn.functional as F
from jaxtyping import Float, Bool, jaxtyped, Int
import torch
from torch import Tensor
from torchvision.transforms import CenterCrop, Compose, Normalize

from data.base import Dataset
from data.utils import ImageResize, ImageSize, Transform
from evaluator.clip import generate_call_signature
from evaluator.detector import DetectionResult, TensorDetector
from evaluator.dinov2 import DINOv2VisionTransformer
from evaluator.openclip import create_vision_transformer


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
    r3indice: bool = False


class MuSc(nn.Module):
    def __init__(self, config: MuScConfig2):
        super().__init__()
        self.r_list = config.r_list
        self.feature_layers = config.feature_layers
        self.topmin_min = config.topmin_min
        self.topmin_max = config.topmin_max
        self.input_H, self.input_W = config.input_image_size.hw()
        # todo: get these from the model
        self.embed_dim = 1024
        self.proj_dim = 768
        self.patch_size = 14
        assert (
            self.input_H % self.patch_size == 0 and self.input_W % self.patch_size == 0
        )
        self.patch_H = self.input_H // self.patch_size
        self.patch_W = self.input_W // self.patch_size
        self.patch_num = self.patch_H * self.patch_W

        if config.is_dino:
            self.vision_encoder = DINOv2VisionTransformer(model_name="dinov2_vitl14")
            self.feature_layers = [-1]
        else:
            self.vision_encoder = create_vision_transformer(
                image_size=ImageSize(h=self.input_H, w=self.input_W),
                device=config.device,
            )
        
        if config.r3indice:
            self.r_list = [3, 1]

        self.ref_features_rlist: list[Float[Tensor, "L B-1 P D"]] | None = None

        self.device = config.device

        self.config = config

    @jaxtyped(typechecker=None)
    def set_ref_features(
        self,
        pixel_values: Float[Tensor, "N 3 {self.input_H} {self.input_W}"],
    ):
        self.ref_features_rlist = []
        pixel_values = pixel_values.to(self.device)
        features = self.vision(pixel_values, self.feature_layers)
        features: Float[Tensor, "L N P D"]
        # features = layer_norm(features, normalized_shape=features.shape[-2:])
        for r in self.r_list:
            r_features_list = []
            for l_features in features:
                r_l_features = self.LNAMD(l_features, r)
                r_features_list.append(r_l_features)
            r_features: Float[Tensor, "L N P D"] = torch.stack(r_features_list, dim=0)
            r_features /= r_features.norm(dim=-1, keepdim=True)
            self.ref_features_rlist.append(r_features)

    def clear_ref_features(self):
        self.ref_features_rlist = None

    @jaxtyped(typechecker=None)
    def forward(
        self,
        pixel_values: Float[Tensor, "B 3 {self.input_H} {self.input_W}"],
    ) -> tuple[
        Float[Tensor, "B"],
        Float[Tensor, "B {self.input_H} {self.input_W}"],
        Int[Tensor, "B L R (B-1) P"],  # 每个图像对其他图像的 patch 级最近邻索引
        Int[Tensor, "B"],  # 每个图像的 patch 级分数中最高的 patch 的索引
        Int[Tensor, "B L R topmink P"],
        # 对于 patch 的所有分数中 topkmin 的那几个的图像索引
        Float[Tensor, "B L R topmink P"],
        # 对于 patch 的所有分数中 topkmin 的那几个的分数
    ]:
        _: Bool[Tensor, "H"] = torch.empty((self.input_H,), dtype=torch.bool)
        _: Bool[Tensor, "W"] = torch.empty((self.input_W,), dtype=torch.bool)
        _: Bool[Tensor, "P"] = torch.empty((self.patch_num,), dtype=torch.bool)
        _: Bool[Tensor, "PH"] = torch.empty((self.patch_H,), dtype=torch.bool)
        _: Bool[Tensor, "PW"] = torch.empty((self.patch_W,), dtype=torch.bool)
        _: Bool[Tensor, "L"] = torch.empty(
            (len(self.feature_layers),), dtype=torch.bool
        )
        _: Bool[Tensor, "D"] = torch.empty((self.embed_dim,), dtype=torch.bool)
        _: Bool[Tensor, "J"] = torch.empty((self.proj_dim,), dtype=torch.bool)

        pixel_values = pixel_values.to(self.device)

        features = self.vision(pixel_values, self.feature_layers)
        features: Float[Tensor, "L B P D"]
        scores_list = []
        min_indices_list = []
        topmink_indices_list = []
        topmink_scores_list = []
        for r_i, r in enumerate(self.r_list):
            r_features_list = []
            for l_features in features:
                r_l_features = self.LNAMD(l_features, r)
                r_features_list.append(r_l_features)
            r_features: Float[Tensor, "L B P D"] = torch.stack(r_features_list, dim=0)
            r_features /= r_features.norm(dim=-1, keepdim=True)
            ref_features = None
            if self.ref_features_rlist is not None:
                ref_features = self.ref_features_rlist[r_i][
                    :, 0 : r_features.shape[1] - 1, ...
                ]
            ref_min_indices = None
            r_scores, r_min_indices, r_topmink_indices, r_topmink_scores = self.MSM(
                r_features, ref_features=ref_features, ref_min_indices=ref_min_indices
            )
            r_scores: Float[Tensor, "L B P"]
            r_min_indices: Int[Tensor, "L B P (B-1)"]
            r_topmink_indices: Int[Tensor, "L B P topmink"]
            r_topmink_scores: Float[Tensor, "L B P topmink"]
            if self.config.r3indice and r == 3:
                ref_min_indices = r_min_indices
                continue
            min_indices_list.append(r_min_indices)
            topmink_indices_list.append(r_topmink_indices)
            topmink_scores_list.append(r_topmink_scores)
            r_scores: Float[Tensor, "B P"] = torch.mean(r_scores, dim=0)
            scores_list.append(r_scores)
        min_indices: Int[Tensor, "R L B P (B-1)"] = torch.stack(min_indices_list, dim=0)
        min_indices: Int[Tensor, "B L R (B-1) P"] = min_indices.permute(2, 1, 0, 4, 3)
        topmink_indices: Int[Tensor, "R L B P topmink"] = torch.stack(
            topmink_indices_list, dim=0
        )
        topmink_indices: Int[Tensor, "B L R topmink P"] = topmink_indices.permute(
            2, 1, 0, 4, 3
        )
        topmink_scores: Float[Tensor, "R L B P topmink"] = torch.stack(
            topmink_scores_list, dim=0
        )
        topmink_scores: Float[Tensor, "B L R topmink P"] = topmink_scores.permute(
            2, 1, 0, 4, 3
        )
        scores: Float[Tensor, "B P"] = torch.mean(
            torch.stack(scores_list, dim=0), dim=0
        )
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
        return (
            final_scores,
            scores_pixel,
            min_indices,
            max_indices_image_level,
            topmink_indices,
            topmink_scores,
        )

    @generate_call_signature(forward)
    def __call__(self): ...

    def vision(
        self,
        pixel_values: Float[Tensor, "B 3 H W"],
        feature_layers: list[int],
    ) -> Float[Tensor, "L B P D"]:
        if self.config.is_dino:
            features = self.vision_encoder(pixel_values)
            features = [features]
        else:
            cls_tokens, features = self.vision_encoder(pixel_values, feature_layers)
        return torch.stack(features, dim=0)

    def compute_similarity(
        self,
        pixel_values: Float[Tensor, "3 H W"],
    ) -> Float[Tensor, "P P"]:
        features = self.vision(pixel_values.unsqueeze(0), self.feature_layers)
        features: Float[Tensor, "L P D"] = features.squeeze(1)
        # features = layer_norm(features, normalized_shape=features.shape[-2:])
        features: Float[Tensor, "P D"] = features.mean(dim=0)
        features /= features.norm(dim=-1, keepdim=True)
        similarity: Float[Tensor, "P P"] = cdist(features, features, p=2)
        similarity = 1 - similarity
        return similarity

    # 对于每个 patch，计算其对应的 patch 与周围 4 个 patch 对应 patch 的平均距离
    @jaxtyped(typechecker=None)
    def compute_patch_offset_distance(
        self,
        image_size: ImageSize,
        match_pindices_: Int[Tensor, "*X P"],
    ) -> Float[Tensor, "*X P"]:
        needed_squeeze = False
        if match_pindices_.ndim == 1:
            match_pindices_ = match_pindices_.unsqueeze(0)
            needed_squeeze = True
        ph, pw = image_size.h // self.patch_size, image_size.w // self.patch_size

        match_indices: Int[Tensor, "X PH PW"] = match_pindices_.view(-1, ph, pw)
        match_coords: Int[Tensor, "X PH PW 2"] = torch.stack(
            [match_indices // pw, match_indices % pw],
            dim=-1,
        )
        pad_coords: Int[Tensor, "X PH+2 PW+2 2"] = F.pad(
            match_coords.permute(0, 3, 1, 2),
            (1, 1, 1, 1),
            mode="replicate",
        ).permute(0, 2, 3, 1)
        neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        distance_list = []
        for dy, dx in neighbor_offsets:
            neighbor_coords = pad_coords[
                :, dy + 1 : dy + 1 + ph, dx + 1 : dx + 1 + pw, :
            ]
            distance: Float[Tensor, "X PH PW"] = torch.norm(
                match_coords.float() - neighbor_coords.float(), dim=-1, p=2
            )
            distance_list.append(distance)
        distances: Float[Tensor, "X PH PW"] = torch.stack(distance_list, dim=-1).mean(
            dim=-1
        )
        distances: Float[Tensor, "X P"] = distances.view(distances.shape[0], -1)
        if needed_squeeze:
            distances = distances.squeeze(0)
        return distances

    def LNAMD(
        self,
        features_: Float[Tensor, "B P D"],
        r: int,
    ) -> Float[Tensor, "B P D"]:
        if r > 1:
            assert r % 2 == 1, "r should be odd."
            features: Float[Tensor, "B D P"] = features_.permute(0, 2, 1)
            features: Float[Tensor, "B D PH PW"] = features.reshape(
                *features.shape[0:2], self.patch_H, self.patch_W
            )
            padding = r // 2
            features: Float[Tensor, f"B D*{r*r} P"] = F.unfold(
                features, kernel_size=(r, r), padding=padding, stride=1, dilation=1
            )
            features: Float[Tensor, f"B P D*{r*r}"] = features.permute(0, 2, 1)
            features: Float[Tensor, f"B*P D*{r*r}"] = features.reshape(
                -1, features.shape[-1]
            )
            pool_batch_size = 2048
            # pool_batch_size = features.shape[0]
            pooled_features_list = []
            for i in range(0, features.shape[0], pool_batch_size):
                batch_features = features[i : i + pool_batch_size]
                pooled_batch_features: Float[Tensor, "_ D"] = adaptive_avg_pool1d(
                    batch_features, self.embed_dim
                )
                pooled_features_list.append(pooled_batch_features)
            features: Float[Tensor, "B*P D"] = torch.cat(pooled_features_list, dim=0)
            features: Float[Tensor, "B P D"] = features.reshape(
                features_.shape[0], features_.shape[1], self.embed_dim
            )
            return features
        return features_

    def MSM(
        self,
        features: Float[Tensor, "L B P D"],
        ref_min_indices: Int[Tensor, "L B P (B-1)"] | None = None,
        ref_features: Float[Tensor, "L B-1 P D"] | None = None,
    ) -> tuple[
        Float[Tensor, "L B P"],
        Int[Tensor, "L B P (B-1)"],
        Int[Tensor, "L B P topmink"],
        Float[Tensor, "L B P topmink"],
    ]:
        scores_list = []
        min_indices_list = []
        topmink_indices_list = []
        topmink_scores_list = []
        for i in range(features.shape[1]):
            scores, indices, topmink_indices, topmink_scores = self.compute_score(
                features,
                i,
                ref_min_indices[:, i, ...] if ref_min_indices is not None else None,
                ref_features,
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

    def compute_score(
        self,
        features: Float[Tensor, "L B P D"],
        i: int,
        ref_match_indices: Int[Tensor, "L P (B-1)"] | None = None,
        ref_features: Float[Tensor, "L B-1 P D"] | None = None,
    ) -> tuple[
        Float[Tensor, "L P"],
        Int[Tensor, "L P (B-1)"],
        Int[Tensor, "L P topmink"],
        # 对于每个 patch，选择了哪几个图像的匹配 patch 作为其分数
        Float[Tensor, "L P topmink"],
    ]:
        feat: Float[Tensor, "L P D"] = features[:, i, :, :]
        if ref_features is not None:
            refs = ref_features
        else:
            refs: Float[Tensor, "L (B-1) P D"] = torch.cat(
                [features[:, :i, :, :], features[:, i + 1 :, :, :]], dim=1
            )
        refs: Float[Tensor, "L (B-1)*P D"] = refs.view(refs.shape[0], -1, refs.shape[3])
        scores: Float[Tensor, "L P (B-1)*P"] = cdist(feat, refs, p=2)
        scores: Float[Tensor, "L P (B-1) P"] = scores.view(
            *scores.shape[0:2], -1, self.patch_num
        )
        match_indices: Int[Tensor, "L P (B-1)"] = torch.argmin(scores, dim=-1)
        scores: Float[Tensor, "L P (B-1)"] = scores.gather(
            dim=-1, index=match_indices.unsqueeze(-1)
        ).squeeze(-1)
        k_min = int(self.topmin_min * scores.shape[-1])
        k_max = int(self.topmin_max * scores.shape[-1])
        k = k_max - k_min
        scores_topkmax, scores_topkmax_indices = torch.topk(
            scores, k=k_max, largest=False, dim=-1, sorted=True
        )
        scores_topkmax: Float[Tensor, f"L P {k_max}"]
        scores_topkmax_indices: Int[Tensor, f"L P {k_max}"]
        scores_topminmax: Float[Tensor, f"L P {k}"] = scores_topkmax[:, :, k_min:k_max]
        scores_topminmax_indices: Int[Tensor, f"L P {k}"] = scores_topkmax_indices[
            :, :, k_min:k_max
        ]
        scores_final: Float[Tensor, "L P"] = torch.mean(scores_topminmax, dim=-1)
        return scores_final, match_indices, scores_topminmax_indices, scores_topminmax

    def RsCIN(
        self,
        cls_similarity: Float[Tensor, "B B"],
        scores: Float[Tensor, "B"],
        top_k: int,
    ) -> Float[Tensor, "B"]:
        cls_masked = cls_similarity.clone()
        if top_k < cls_similarity.shape[1]:
            _, indices = torch.topk(
                cls_similarity, k=cls_similarity.shape[1] - top_k, largest=False, dim=-1
            )
            cls_masked.scatter_(dim=-1, index=indices, value=0)
        cls_masked_sum = torch.sum(cls_masked, dim=-1, keepdim=True)
        cls_masked: Float[Tensor, "B B"] = cls_masked / cls_masked_sum
        weighted_scores: Float[Tensor, "B 1"] = cls_masked @ scores.unsqueeze(1)
        return weighted_scores.squeeze(1)


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
        if config.r_list != default_config.r_list:
            inner = ""
            for r in config.r_list:
                inner += str(r)
            name += f"(r{inner})"
        if config.feature_layers != default_config.feature_layers:
            inner = ""
            for l in config.feature_layers:
                inner += str(l)
            name += f"(l{inner})"
        if (
            config.topmin_max != default_config.topmin_max
            or config.topmin_min != default_config.topmin_min
        ):
            name += f"(top{config.topmin_min}-{config.topmin_max})"
        if config.is_dino:
            name += "(dino)"
        if config.r3indice:
            name += "(r3i)"
        if const_features:
            if train_data is not None:
                name += "(train)"
            else:
                name += "(const)"

        self.const_feature = const_features
        self.train_data = train_data
        self.last_class_name = "??"

        mean = (
            (0.485, 0.456, 0.406)
            if config.is_dino
            else (0.48145466, 0.4578275, 0.40821073)
        )
        std = (
            (0.229, 0.224, 0.225)
            if config.is_dino
            else (0.26862954, 0.26130258, 0.27577711)
        )
        image_transform = Compose(
            [
                CenterCrop(config.input_image_size.hw()),
                Normalize(mean=mean, std=std),
            ]
        )
        mask_transform = CenterCrop(config.input_image_size.hw())
        super().__init__(
            name=name,
            transform=Transform(
                resize=config.image_resize,
                image_transform=image_transform,
                mask_transform=mask_transform,
            ),
        )

    def __call__(
        self, images: Float[torch.Tensor, "N C H W"], class_name: str
    ) -> DetectionResult:
        self.model.eval()
        with torch.no_grad(), torch.autocast("cuda"):
            if self.last_class_name != class_name and self.const_feature:
                self.last_class_name = class_name
                if self.train_data is not None:
                    train_tensor_dataset = self.train_data(class_name, self.transform)
                    indices = torch.randperm(len(train_tensor_dataset))[
                        : images.shape[0] - 1
                    ]
                    subset = torch.stack(
                        [train_tensor_dataset[int(i.item())] for i in indices]
                    )
                    self.model.set_ref_features(subset)
                else:
                    self.model.set_ref_features(images[: images.shape[0] - 1, ...])
            scores, maps, min_indices, max_indices, topmink_indices, topmink_scores = (
                self.model(images)
            )
        patch_distances = (
            self.model.compute_patch_offset_distance(
                self.model.config.input_image_size,
                min_indices.reshape(-1, min_indices.shape[-1]),
            )
            .view(min_indices.shape[0], -1)
            .mean(dim=-1)
        )
        print(f"Avg patch distance: {patch_distances.mean().item():.4f}")
        return DetectionResult(
            pred_scores=scores,
            anomaly_maps=maps,
            patch_distances=patch_distances,
            other=(min_indices, max_indices, topmink_indices, topmink_scores),
        )
