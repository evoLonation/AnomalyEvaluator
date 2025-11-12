"""
MuSc (Multi-scale CLIP) Anomaly Detection Implementation

This module implements the MuSc anomaly detection model using CLIP Vision Transformer.
The implementation follows the style of clip.py and integrates core MuSc components.

Code Sources:
- PatchMaker, MeanMapper, Preprocessing: Copied from models/modules/_LNAMD.py
- LNAMD: Copied from models/modules/_LNAMD.py
- MSM (compute_scores_fast, MSM): Copied from models/modules/_MSM.py
- RsCIN (MMO, RsCIN): Copied from models/modules/_RsCIN.py
- BatchMuSc.forward: Based on anomaly_detection.py MuScDetector.detect()

Key Components:
- LNAMD: Layer-wise Neighborhood Aggregation for Multi-scale Detection
- MSM: Mutual Scoring Mechanism for anomaly scoring
- RsCIN: Riemannian Similarity-aware Class-token Integration Network
- CLIPVisionTransformer: Image encoder from clip.py
"""

from dataclasses import dataclass
from pathlib import Path

from data.cached_dataset import MVTecAD, RealIAD, VisA
from data.utils import ImageSize
from evaluator.detector import DetectionResult, Detector, TensorDetector
from evaluator.evaluation import evaluation_detection
from evaluator.openclip import create_vision_transformer
from .clip import (
    CLIPModel,
    CLIPVisionTransformer,
    generate_call_signature,
)
from PIL import Image
from .loss import binary_dice_loss, focal_loss
from jaxtyping import Bool, Float, jaxtyped
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import cast
from torch.utils.data import RandomSampler, Sampler
import evaluator.reproducibility as repro


# PatchMaker - copied from models/modules/_LNAMD.py
class PatchMaker(nn.Module):
    """Extract local r×r neighborhoods for each spatial position"""

    def __init__(self, patchsize: int, stride: int = 1):
        super().__init__()
        self.patchsize = patchsize
        self.stride = stride

    def patchify(
        self,
        features: Float[torch.Tensor, "B C H W"],
        return_spatial_info: bool = False,
    ) -> (
        tuple[Float[torch.Tensor, "B H*W C r r"], tuple[int, int]]
        | Float[torch.Tensor, "B H*W C r r"]
    ):
        padding = int((self.patchsize - 1) / 2)
        unfolder = nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, tuple(number_of_total_patches)
        return unfolded_features


# MeanMapper - copied from models/modules/_LNAMD.py
class MeanMapper(nn.Module):
    """Aggregate r×r neighborhood features to fixed dimension using adaptive average pooling"""

    def __init__(self, preprocessing_dim: int):
        super().__init__()
        self.preprocessing_dim = preprocessing_dim

    @jaxtyped(typechecker=None)
    def forward(
        self, features: Float[torch.Tensor, "N C r r"]
    ) -> Float[torch.Tensor, "N {self.preprocessing_dim}"]:
        features = features.reshape(len(features), 1, -1)
        return F.adaptive_avg_pool1d(features, self.preprocessing_dim).squeeze(1)


# Preprocessing - copied from models/modules/_LNAMD.py
class Preprocessing(nn.Module):
    """Apply MeanMapper to each layer and stack results"""

    def __init__(self, input_layers: list[int], output_dim: int):
        super().__init__()
        self.output_dim = output_dim
        self.preprocessing_modules = nn.ModuleList()
        for _ in input_layers:
            module = MeanMapper(output_dim)
            self.preprocessing_modules.append(module)

    @jaxtyped(typechecker=None)
    def forward(
        self, features: list[Float[torch.Tensor, "N C r r"]]
    ) -> Float[torch.Tensor, "N num_layers {self.output_dim}"]:
        _features = []
        for module, feature in zip(self.preprocessing_modules, features):
            _features.append(module(feature))
        return torch.stack(_features, dim=1)


# LNAMD - copied from models/modules/_LNAMD.py
class LNAMD(nn.Module):
    """Layer-wise Neighborhood Aggregation for Multi-scale Detection"""

    def __init__(
        self,
        device: torch.device,
        feature_dim: int = 1024,
        feature_layer: list[int] = [1, 2, 3, 4],
        r: int = 3,
        patchstride: int = 1,
    ):
        super().__init__()
        self.device = device
        self.r = r
        self.patch_maker = PatchMaker(r, stride=patchstride)
        self.LNA = Preprocessing(feature_layer, feature_dim)

    @jaxtyped(typechecker=None)
    def _embed(
        self, features: list[Float[torch.Tensor, "B num_patches+1 {self.feature_dim}"]]
    ) -> Float[torch.Tensor, "B num_valid_patches num_layers {self.feature_dim}"]:
        B = features[0].shape[0]

        features_layers = []
        for feature in features:
            # Remove CLS token and reshape to spatial dimensions
            feature = feature[:, 1:, :]
            feature = feature.reshape(
                feature.shape[0],
                int(math.sqrt(feature.shape[1])),
                int(math.sqrt(feature.shape[1])),
                feature.shape[2],
            )
            feature = feature.permute(0, 3, 1, 2)
            # Layer normalization
            feature = nn.LayerNorm(
                [feature.shape[1], feature.shape[2], feature.shape[3]]
            ).to(self.device)(feature)
            features_layers.append(feature)

        if self.r != 1:
            # Divide into r×r patches
            features_layers_patchified = []
            patch_shapes = []
            for x in features_layers:
                unfolded, shape = self.patch_maker.patchify(x, return_spatial_info=True)
                features_layers_patchified.append(unfolded)
                patch_shapes.append(shape)
            features_layers = features_layers_patchified
        else:
            patch_shapes = [f.shape[-2:] for f in features_layers]
            features_layers = [
                f.reshape(f.shape[0], f.shape[1], -1, 1, 1).permute(0, 2, 1, 3, 4)
                for f in features_layers
            ]

        # Align all layers to reference spatial size
        ref_num_patches = patch_shapes[0]
        for i in range(1, len(features_layers)):
            patch_dims = patch_shapes[i]
            if (
                patch_dims[0] == ref_num_patches[0]
                and patch_dims[1] == ref_num_patches[1]
            ):
                continue
            _features = features_layers[i]
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features_layers[i] = _features
        features_layers = [x.reshape(-1, *x.shape[-3:]) for x in features_layers]

        # Aggregation
        features_layers = self.LNA(features_layers)
        features_layers = features_layers.reshape(B, -1, *features_layers.shape[-2:])

        return features_layers.detach().cpu()


# MSM helper functions - copied from models/modules/_MSM.py
def compute_scores_fast(
    Z: Float[torch.Tensor, "N L C"],
    i: int,
    device: torch.device,
    topmin_min: float = 0,
    topmin_max: float = 0.3,
) -> Float[torch.Tensor, "L"]:
    """Compute anomaly scores for image i using mutual scoring"""
    image_num, patch_num, c = Z.shape
    Z_ref = torch.cat((Z[:i], Z[i + 1 :]), dim=0)
    patch2image = torch.cdist(Z[i : i + 1], Z_ref.reshape(-1, c)).reshape(
        patch_num, image_num - 1, patch_num
    )
    patch2image = torch.min(patch2image, -1)[0]

    # Interval average
    k_max_val = topmin_max
    k_min_val = topmin_min
    if k_max_val < 1:
        k_max_val = int(patch2image.shape[1] * k_max_val)
    if k_min_val < 1:
        k_min_val = int(patch2image.shape[1] * k_min_val)
    if k_max_val < k_min_val:
        k_max_val, k_min_val = k_min_val, k_max_val
    vals, _ = torch.topk(
        patch2image.float(), int(k_max_val), largest=False, sorted=True
    )
    vals, _ = torch.topk(
        vals.float(), int(k_max_val - k_min_val), largest=True, sorted=True
    )
    return torch.mean(vals, dim=1)


def MSM(
    Z: Float[torch.Tensor, "N L C"],
    device: torch.device,
    topmin_min: float = 0,
    topmin_max: float = 0.3,
) -> Float[torch.Tensor, "N L"]:
    """Mutual Scoring Mechanism for anomaly detection"""
    anomaly_scores_matrix = torch.tensor([]).double().to(device)
    for i in range(Z.shape[0]):
        anomaly_scores_i = compute_scores_fast(
            Z, i, device, topmin_min, topmin_max
        ).unsqueeze(0)
        anomaly_scores_matrix = torch.cat(
            (anomaly_scores_matrix, anomaly_scores_i.double()), dim=0
        )
    return anomaly_scores_matrix


# RsCIN helper functions - copied from models/modules/_RsCIN.py
def MMO(
    W: Float[torch.Tensor, "N N"],
    score: Float[torch.Tensor, "N"],
    k_list: list[int] = [1, 2, 3],
) -> Float[torch.Tensor, "N"]:
    """Markov Matrix Optimization for score refinement"""
    S_list = []
    for k in k_list:
        _, topk_matrix = torch.topk(
            W.float(), W.shape[0] - k, largest=False, sorted=True
        )
        W_mask = W.clone()
        for i in range(W.shape[0]):
            W_mask[i, topk_matrix[i]] = 0
        n = W.shape[-1]
        D_ = torch.zeros_like(W).float()
        for i in range(n):
            D_[i, i] = 1 / (W_mask[i, :].sum())
        P = D_ @ W_mask
        S = score.clone().unsqueeze(-1)
        S = P @ S
        S_list.append(S)
    S = torch.concat(S_list, -1).mean(-1)
    return S


def RsCIN(
    scores_old: np.ndarray,
    cls_tokens: list[np.ndarray] | None = None,
    k_list: list[int] = [0],
) -> np.ndarray:
    """Riemannian Similarity-aware Class-token Integration Network"""
    if cls_tokens is None or 0 in k_list:
        return scores_old
    cls_tokens_array = np.array(cls_tokens)
    scores = (scores_old - scores_old.min()) / (scores_old.max() - scores_old.min())
    similarity_matrix = cls_tokens_array @ cls_tokens_array.T
    similarity_matrix = torch.tensor(similarity_matrix)
    scores_new = MMO(
        similarity_matrix.clone().float(),
        score=torch.tensor(scores).clone().float(),
        k_list=k_list,
    )
    scores_new = scores_new.numpy()
    return scores_new


@dataclass
class MuScConfig:
    model_name: str = "openai/clip-vit-large-patch14-336"
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_image_size: tuple[int, int] = (518, 518)
    batch_size: int = 32
    feature_layers: list[int] | None = None  # default: [5, 11, 17, 23]
    r_list: list[int] | None = None  # default: [1, 3, 5]
    k_score: list[int] | None = None  # default: [1, 2, 3]
    topmin_min: float = 0.0
    topmin_max: float = 0.3

    def __post_init__(self):
        if self.feature_layers is None:
            self.feature_layers = [5, 11, 17, 23]
        if self.r_list is None:
            self.r_list = [1, 3, 5]
        if self.k_score is None:
            self.k_score = [1, 2, 3]


class BatchMuSc(nn.Module):
    """
    MuSc Anomaly Detector using CLIP Vision Transformer
    Implementation follows anomaly_detection.py with batch processing support
    """

    def __init__(self, config: MuScConfig):
        super().__init__()
        self.config = config
        self.batch_size = config.batch_size
        self.input_H, self.input_W = config.input_image_size

        # Extract vision transformer
        self.vision, self.preprocessor = create_vision_transformer(
            image_size=config.input_image_size,
            device=config.device,
        )
        # Load CLIP model
        clip_model: CLIPModel = CLIPModel.from_pretrained(
            config.model_name, device_map=config.device
        )

        # Extract vision transformer
        self.vision = CLIPVisionTransformer(
            clip_model,
            config.input_image_size,
            enable_vvv=False,
            device=config.device,
        )

        # Ensure config fields are initialized
        if config.feature_layers is None:
            config.feature_layers = [5, 11, 17, 23]
        if config.r_list is None:
            config.r_list = [1, 3, 5]
        if config.k_score is None:
            config.k_score = [1, 2, 3]

        self.feature_layers: list[int] = config.feature_layers
        self.r_list: list[int] = config.r_list
        self.k_score: list[int] = config.k_score
        self.topmin_min = config.topmin_min
        self.topmin_max = config.topmin_max
        self.device = config.device

    @jaxtyped(typechecker=None)
    def forward(
        self,
        pixel_values: Float[
            torch.Tensor, "{self.batch_size} C=3 {self.input_H} {self.input_W}"
        ],
    ) -> tuple[
        Float[torch.Tensor, "{self.batch_size}"],
        Float[torch.Tensor, "{self.batch_size} {self.input_H} {self.input_W}"],
    ]:
        """
        Forward pass for MuSc anomaly detection

        Returns:
            anomaly_scores: (batch_size,) anomaly scores for each image
            anomaly_maps: (batch_size, H, W) anomaly heatmaps
        """
        B = pixel_values.shape[0]

        # Extract features from CLIP vision transformer
        with torch.no_grad(), torch.cuda.amp.autocast():
            cls_token, patch_tokens_list = self.vision(
                pixel_values=pixel_values,
                output_layers=self.feature_layers,
            )
            # Normalize class tokens
            cls_token = cls_token / cls_token.norm(dim=-1, keepdim=True)

        # Convert to list of tensors with CLS token prepended
        # patch_tokens_list contains patch tokens without CLS
        # Need to add dummy CLS token for LNAMD compatibility
        patch_tokens_with_cls = []
        for patch_tokens in patch_tokens_list:
            # Add dummy CLS token at position 0
            dummy_cls = torch.zeros_like(patch_tokens[:, 0:1, :])
            tokens_with_cls = torch.cat([dummy_cls, patch_tokens], dim=1)
            patch_tokens_with_cls.append(tokens_with_cls)

        # LNAMD - Layer-wise Neighborhood Aggregation
        feature_dim = patch_tokens_with_cls[0].shape[-1]
        anomaly_maps_r = torch.tensor([]).double()

        for r in self.r_list:
            LNAMD_r = LNAMD(
                device=self.device,
                r=r,
                feature_dim=feature_dim,
                feature_layer=self.feature_layers,
            )
            Z_layers = {}

            # Process features with LNAMD
            patch_tokens_device = [p.to(self.device) for p in patch_tokens_with_cls]
            with torch.no_grad(), torch.cuda.amp.autocast():
                features = LNAMD_r._embed(patch_tokens_device)
                features /= features.norm(dim=-1, keepdim=True)
                for l in range(len(self.feature_layers)):
                    if str(l) not in Z_layers.keys():
                        Z_layers[str(l)] = []
                    Z_layers[str(l)].append(features[:, :, l, :])

            # MSM - Mutual Scoring Mechanism
            anomaly_maps_l = torch.tensor([]).double()
            for l in Z_layers.keys():
                Z = torch.cat(Z_layers[l], dim=0).to(self.device)
                anomaly_maps_msm = MSM(
                    Z=Z,
                    device=self.device,
                    topmin_min=self.topmin_min,
                    topmin_max=self.topmin_max,
                )
                anomaly_maps_l = torch.cat(
                    (anomaly_maps_l, anomaly_maps_msm.unsqueeze(0).cpu()), dim=0
                )
                torch.cuda.empty_cache()

            anomaly_maps_l = torch.mean(anomaly_maps_l, 0)
            anomaly_maps_r = torch.cat(
                (anomaly_maps_r, anomaly_maps_l.unsqueeze(0)), dim=0
            )

        anomaly_maps_iter = torch.mean(anomaly_maps_r, 0).to(self.device)
        del anomaly_maps_r
        torch.cuda.empty_cache()

        # Interpolate to original image size
        B_actual, L = anomaly_maps_iter.shape
        H = int(np.sqrt(L))
        anomaly_maps = F.interpolate(
            anomaly_maps_iter.view(B_actual, 1, H, H),
            size=self.input_H,
            mode="bilinear",
            align_corners=True,
        )

        anomaly_maps = anomaly_maps.cpu().numpy()
        torch.cuda.empty_cache()

        # Compute anomaly scores
        ac_score = np.array(anomaly_maps).reshape(B_actual, -1).max(-1)

        # RsCIN - Riemannian Similarity-aware Class-token Integration
        cls_tokens_list = [
            cls_token[i].squeeze().cpu().numpy() for i in range(cls_token.shape[0])
        ]
        scores_cls = RsCIN(ac_score, cls_tokens_list, k_list=self.k_score)

        # Convert to tensors
        anomaly_scores = torch.tensor(scores_cls, dtype=torch.float32)
        anomaly_maps_out = torch.tensor(anomaly_maps.squeeze(1), dtype=torch.float32)

        return anomaly_scores, anomaly_maps_out

    @generate_call_signature(forward)
    def __call__(self): ...


class MuScDetector(Detector):
    def __init__(self, config: MuScConfig):
        super().__init__(name="MuSc")
        self.model = BatchMuSc(config)
        self.preprocessor = self.model.preprocessor

    @torch.no_grad()
    def __call__(self, images: list[str], class_name: str) -> DetectionResult:
        self.model.eval()
        image_tensors = []
        for image in images:
            image = self.preprocessor(Image.open(image))
            image_tensors.append(image)
        image_tensors = torch.stack(image_tensors, dim=0)
        image_tensors = image_tensors.to(self.model.device)
        anomaly_scores, anomaly_maps = self.model(image_tensors)
        return DetectionResult(
            pred_scores=anomaly_scores.cpu().numpy(),
            anomaly_maps=anomaly_maps.cpu().numpy(),
        )


if __name__ == "__main__":
    repro.init(42)
    detector = MuScDetector(MuScConfig())
    for dataset in [MVTecAD(), VisA(), RealIAD()]:
        category_samplers: dict[str, Sampler] = {
            k: RandomSampler(
                v, replacement=False, generator=torch.Generator().manual_seed(42)
            )
            for k, v in dataset.get_meta_dataset(
                # detector.image_size
            ).category_datas.items()
        }
        evaluation_detection(
            path=Path("results/musc_hf_pp_test"),
            detector=detector,
            dataset=dataset,
            batch_size=16,
            category_samplers=category_samplers,
        )
