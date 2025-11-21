from dataclasses import dataclass, field
from typing import Callable
from torch import adaptive_avg_pool1d, cdist, equal, layer_norm, nn, tensor
from torch.nn.functional import unfold
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
    k_list: list[int] = field(default_factory=lambda: [1, 2, 3])
    topmin_min: float = 0.0
    topmin_max: float = 0.3

    patch_match: bool = False
    is_dino: bool = False
    borrow_indices: bool = False
    r1_with_r3_indice: bool = False


class MuSc(nn.Module):
    def __init__(self, config: MuScConfig2):
        super().__init__()
        self.r_list = config.r_list
        self.k_list = config.k_list
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

        self.dino_encoder = None
        if config.is_dino:
            self.dino_encoder = DINOv2VisionTransformer(model_name="dinov2_vitl14")
            self.feature_layers = [-1]
        self.vision_encoder = create_vision_transformer(
            image_size=ImageSize(h=self.input_H, w=self.input_W), device=config.device
        )
        if config.borrow_indices:
            self.r_list = [5, 3, 1]
        if config.r1_with_r3_indice:
            self.r_list = [5, 1]

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
        _, features = self.vision(pixel_values, self.feature_layers)
        features: Float[Tensor, "L N P D"]
        features = layer_norm(features, normalized_shape=features.shape[-2:])
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

        cls_tokens, features = self.vision(pixel_values, self.feature_layers)
        cls_tokens: Float[Tensor, "B J"]
        features: Float[Tensor, "L B P D"]
        scores_list = []
        features = layer_norm(features, normalized_shape=features.shape[-2:])
        min_indices_list = []
        for r_i, r in enumerate(self.r_list):
            r_features_list = []
            for l_features in features:
                r_l_features = self.LNAMD(l_features, r)
                r_features_list.append(r_l_features)
            r_features: Float[Tensor, "L B P D"] = torch.stack(r_features_list, dim=0)
            r_features /= r_features.norm(dim=-1, keepdim=True)
            ref_min_indices = None
            if self.config.borrow_indices and r == 1:
                ref_min_indices = min_indices_list[0]
            if self.config.r1_with_r3_indice and r == 1:
                ref_min_indices = min_indices_list[0]
            ref_features = None
            if self.ref_features_rlist is not None:
                ref_features = self.ref_features_rlist[r_i][
                    :, 0 : r_features.shape[1] - 1, ...
                ]
            r_scores, r_min_indices = self.MSM(
                r_features,
                ref_min_indices,
                ref_features,
            )
            r_min_indices: Int[Tensor, "L B P (B-1)"]
            if False:
                min_indices_flatten: Int[Tensor, "L*B*(B-1) P"] = r_min_indices.permute(
                    0, 1, 3, 2
                ).reshape(-1, self.patch_num)
                correct_indices: Int[Tensor, "P"] = torch.arange(
                    0, self.patch_num, device=self.device
                )
                correct_mask: Bool[Tensor, "L*B*(B-1) P"] = (
                    min_indices_flatten == correct_indices.unsqueeze(0)
                )
                acc = correct_mask.sum().item() / correct_mask.numel()
                print(f"r={r} MSM accuracy: {acc:.4f}")
            min_indices_list.append(r_min_indices)
            r_scores: Float[Tensor, "L B P"]
            r_scores: Float[Tensor, "B P"] = torch.mean(r_scores, dim=0)
            scores_list.append(r_scores)
        min_indices: Int[Tensor, "R L B P (B-1)"] = torch.stack(min_indices_list, dim=0)
        min_indices: Int[Tensor, "B L R (B-1) P"] = min_indices.permute(2, 1, 0, 4, 3)
        if self.config.r1_with_r3_indice:
            scores = scores_list[1]
        else:
            scores: Float[Tensor, "B P"] = torch.mean(
                torch.stack(scores_list, dim=0), dim=0
            )
        scores_image_level: Float[Tensor, "B"] = torch.max(scores, dim=1).values

        cls_tokens = cls_tokens / cls_tokens.norm(dim=-1, keepdim=True)
        cls_similarity: Float[Tensor, "B B"] = cls_tokens @ cls_tokens.t()

        if self.ref_features_rlist is not None:
            # 因为cls_similarity是 batch 内部图像间的，不涉及参考集
            final_scores: Float[Tensor, "B"] = scores_image_level
        else:
            final_scores_list = []
            for k in self.k_list:
                k_final_scores: Float[Tensor, "B"] = self.RsCIN(
                    cls_similarity, scores_image_level, k
                )
                final_scores_list.append(k_final_scores)
            final_scores: Float[Tensor, "B"] = torch.mean(
                torch.stack(final_scores_list, dim=0), dim=0
            )
        scores_patch: Float[Tensor, "B PH PW"] = scores.view(
            -1, self.patch_H, self.patch_W
        )
        scores_pixel: Float[Tensor, "B H W"] = torch.nn.functional.interpolate(
            scores_patch.unsqueeze(1),
            size=(self.input_H, self.input_W),
            mode="bilinear",
            align_corners=True,
        ).squeeze(1)
        return final_scores, scores_pixel, min_indices

    @generate_call_signature(forward)
    def __call__(self): ...

    def vision(
        self,
        pixel_values: Float[Tensor, "B 3 H W"],
        feature_layers: list[int],
    ) -> tuple[
        Float[Tensor, "B J"],
        Float[Tensor, "L B P D"],
    ]:
        if self.config.is_dino:
            assert self.dino_encoder is not None
            features = self.dino_encoder(pixel_values)
            features = [features]
            cls_tokens, _ = self.vision_encoder(pixel_values, [])
        else:
            cls_tokens, features = self.vision_encoder(pixel_values, feature_layers)
        return cls_tokens, torch.stack(features, dim=0)

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
            features: Float[Tensor, f"B D*{r*r} P"] = unfold(
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
    ]:
        scores_list = []
        min_indices_list = []
        for i in range(features.shape[1]):
            scores, indices = self.compute_score(
                features,
                i,
                ref_min_indices[:, i, ...] if ref_min_indices is not None else None,
                ref_features,
            )
            scores_list.append(scores)
            min_indices_list.append(indices)
        return torch.stack(scores_list, dim=1), torch.stack(min_indices_list, dim=1)

    def compute_score(
        self,
        features: Float[Tensor, "L B P D"],
        i: int,
        ref_match_indices: Int[Tensor, "L P (B-1)"] | None = None,
        ref_features: Float[Tensor, "L B-1 P D"] | None = None,
    ) -> tuple[
        Float[Tensor, "L P"],
        Int[Tensor, "L P (B-1)"],
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
        if self.config.patch_match:
            correct_indices: Int[Tensor, "P"] = torch.arange(
                0, self.patch_num, device=self.device
            )
            match_indices[:] = correct_indices.unsqueeze(0).unsqueeze(-1)
        if ref_match_indices is not None:
            match_indices = ref_match_indices
        scores1: Float[Tensor, "L P (B-1)"] = scores.gather(
            dim=-1, index=match_indices.unsqueeze(-1)
        ).squeeze(-1)
        # scores2: Float[Tensor, "L P (B-1)"] = torch.min(scores, dim=-1).values
        # assert equal(scores1, scores2)
        scores = scores1
        k_min = int(self.topmin_min * scores.shape[-1])
        k_max = int(self.topmin_max * scores.shape[-1])
        k = k_max - k_min
        scores_topkmax: Float[Tensor, f"L P {k_max}"] = torch.topk(
            scores, k=k_max, largest=False, dim=-1, sorted=True
        ).values
        scores_topminmax: Float[Tensor, f"L P {k}"] = scores_topkmax[:, :, k_min:k_max]
        scores_final: Float[Tensor, "L P"] = torch.mean(scores_topminmax, dim=-1)
        return scores_final, match_indices

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
        if config.k_list != default_config.k_list:
            inner = ""
            for k in config.k_list:
                inner += str(k)
            name += f"(k{inner})"
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
        if config.patch_match:
            name += "(match)"
        if config.is_dino:
            name += "(dino)"
        if config.borrow_indices:
            name += "(borrow)"
        if config.r1_with_r3_indice:
            name += "(r1fr3)"
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
                    subset = torch.stack(
                        [train_tensor_dataset[i] for i in range(images.shape[0] - 1)]
                    )
                    self.model.set_ref_features(subset)
                else:
                    self.model.set_ref_features(images[: images.shape[0] - 1, ...])
            scores, maps, min_indices = self.model(images)
        return DetectionResult(
            pred_scores=scores,
            anomaly_maps=maps,
            other=min_indices,
        )
