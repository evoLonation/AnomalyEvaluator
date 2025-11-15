from dataclasses import dataclass, field
from torch import adaptive_avg_pool1d, cdist, layer_norm, nn, tensor
from torch.nn.functional import unfold
from jaxtyping import Float, Bool, jaxtyped
import torch
from torch import Tensor
from torchvision.transforms import CenterCrop, Compose, Normalize

from data.utils import ImageResize, ImageSize
from evaluator.clip import generate_call_signature
from evaluator.detector import DetectionResult, TensorDetector
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

        self.vision_encoder, _ = create_vision_transformer(
            image_size=ImageSize(h=self.input_H, w=self.input_W), device=config.device
        )
        self.device = config.device

    @jaxtyped(typechecker=None)
    def forward(
        self,
        pixel_values: Float[Tensor, "B 3 {self.input_H} {self.input_W}"],
    ) -> tuple[
        Float[Tensor, "B"],
        Float[Tensor, "B {self.input_H} {self.input_W}"],
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
        for r in self.r_list:
            r_features_list = []
            for l_features in features:
                r_l_features = self.LNAMD(l_features, r)
                r_features_list.append(r_l_features)
            r_features: Float[Tensor, "L B P D"] = torch.stack(r_features_list, dim=0)
            r_features /= r_features.norm(dim=-1, keepdim=True)
            r_scores: Float[Tensor, "L B P"] = self.MSM(
                r_features, self.topmin_min, self.topmin_max
            )
            r_scores: Float[Tensor, "B P"] = torch.mean(r_scores, dim=0)
            scores_list.append(r_scores)
        scores: Float[Tensor, "B P"] = torch.mean(
            torch.stack(scores_list, dim=0), dim=0
        )
        scores_image_level: Float[Tensor, "B"] = torch.max(scores, dim=1).values

        cls_tokens = cls_tokens / cls_tokens.norm(dim=-1, keepdim=True)
        cls_similarity: Float[Tensor, "B B"] = cls_tokens @ cls_tokens.t()

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
        return final_scores, scores_pixel

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
            features: Float[Tensor, "B D PH PW"] = features.view(
                *features.shape[0:2], self.patch_H, self.patch_W
            )
            padding = r // 2
            features: Float[Tensor, f"B D*{r*r} P"] = unfold(
                features, kernel_size=(r, r), padding=padding, stride=1, dilation=1
            )
            features: Float[Tensor, f"B P D*{r*r}"] = features.permute(0, 2, 1)
            features: Float[Tensor, f"B*P D*{r*r}"] = features.view(
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
            features: Float[Tensor, "B P D"] = features.view(
                features_.shape[0], features_.shape[1], self.embed_dim
            )
            return features
        return features_

    def MSM(
        self,
        features: Float[Tensor, "L B P D"],
        topmin_min: float,
        topmin_max: float,
    ) -> Float[Tensor, "L B P"]:
        scores_list = []
        for i in range(features.shape[1]):
            scores: Float[Tensor, "L P"] = self.compute_score(
                features, i, topmin_min, topmin_max
            )
            scores_list.append(scores)
        return torch.stack(scores_list, dim=1)

    def compute_score(
        self,
        features: Float[Tensor, "L B P D"],
        i: int,
        topmin_min: float,
        topmin_max: float,
    ) -> Float[Tensor, "L P"]:
        feat: Float[Tensor, "L P D"] = features[:, i, :, :]
        refs: Float[Tensor, "L (B-1) P D"] = torch.cat(
            [features[:, :i, :, :], features[:, i + 1 :, :, :]], dim=1
        )
        refs: Float[Tensor, "L (B-1)*P D"] = refs.view(refs.shape[0], -1, refs.shape[3])
        scores: Float[Tensor, "L P (B-1)*P"] = cdist(feat, refs, p=2)
        scores: Float[Tensor, "L P (B-1) P"] = scores.view(
            *scores.shape[0:2], -1, self.patch_num
        )
        scores: Float[Tensor, "L P (B-1)"] = torch.min(scores, dim=-1).values
        k_min = int(topmin_min * scores.shape[-1])
        k_max = int(topmin_max * scores.shape[-1])
        k = k_max - k_min
        scores_topkmax: Float[Tensor, f"L P {k_max}"] = torch.topk(
            scores, k=k_max, largest=False, dim=-1, sorted=True
        ).values
        scores_topminmax: Float[Tensor, f"L P {k}"] = scores_topkmax[:, :, k_min:k_max]
        scores_final: Float[Tensor, "L P"] = torch.mean(scores_topminmax, dim=-1)
        return scores_final

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
    def __init__(self, config: MuScConfig2):
        self.model = MuSc(config)
        image_transform = Compose(
            [
                CenterCrop(config.input_image_size.hw()),
                Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        mask_transform = CenterCrop(config.input_image_size.hw())
        super().__init__(
            name="MuSc2",
            resize=config.image_resize,
            image_transform=image_transform,
            mask_transform=mask_transform,
        )

    def __call__(
        self, images: Float[torch.Tensor, "N C H W"], class_name: str
    ) -> DetectionResult:
        self.model.eval()
        with torch.no_grad(), torch.autocast("cuda"):
            scores, maps = self.model(images)
        return DetectionResult(
            pred_scores=scores.cpu(),
            anomaly_maps=maps.cpu(),
        )
