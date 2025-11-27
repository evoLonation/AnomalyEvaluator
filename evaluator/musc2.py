from dataclasses import dataclass, field
import time
from typing import Callable, Literal
import cv2
import numpy as np
from torch import adaptive_avg_pool1d, cdist, equal, layer_norm, nn, tensor
import torch.nn.functional as F
from jaxtyping import Float, Bool, jaxtyped, Int
import torch
from torch import Tensor
from torchvision.transforms import CenterCrop, Compose, Normalize

from align.match_patch import compute_background_mask, get_match_patch_mat
from data.base import Dataset
from data.utils import (
    ImageResize,
    ImageSize,
    Transform,
    from_cv2_image,
    normalize_image,
    to_cv2_image,
)
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
    # 如果不为 None，则在检测时计算 patch 偏移距离, 为跨整张图片时的距离惩罚
    offset_distance: float | None = None
    # recompute: 计算匹配矩阵后，对图片进行旋转，重新计算特征
    # distonly: 只计算匹配矩阵，只计算旋转矩阵并对patch坐标进行变换，用于 offset_distance 计算
    match_patch: Literal[None, "recompute", "distonly"] = None
    detail_result: bool = False
    # 是否启用shift augmentation: 对每张图片生成3种平移增强（右移、上移、右上移各patch_size/2）
    shift_augmentation: bool = False


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
        self.device = config.device
        self.config = config

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

        self.detail_result = config.detail_result

        self.enable_shift_augmentation = config.shift_augmentation
        # shift augmentation: 3种平移方式（右移、上移、右上移）
        self.shift_offsets = [
            (0, self.patch_size // 2),  # 右移
            (self.patch_size // 2, 0),  # 上移
            (self.patch_size // 2, self.patch_size // 2),  # 右上移
        ]

        self.ref_features_rlist: list[Float[Tensor, "L B-1 P D"]] | None = None

        if self.config.offset_distance is not None:
            self.patch_origin_coords = self.get_patch_pixel_coords()
            if self.config.match_patch != "distonly":
                patch_distances = cdist(
                    self.patch_origin_coords,
                    self.patch_origin_coords,
                    p=2,
                )
                patch_distances: Float[Tensor, "1 P 1 P"] = patch_distances.unsqueeze(
                    0
                ).unsqueeze(2)
                patch_distances = (
                    patch_distances * self.config.offset_distance / self.patch_size
                )
                self.patch_offset_distances = patch_distances

    # patch 的像素级坐标
    def get_patch_pixel_coords(self) -> Float[Tensor, "P 2"]:
        x_coords: Int[Tensor, "PW"] = torch.arange(
            0,
            self.patch_W,
            device=self.device,
        )
        x_coords: Int[Tensor, "PH PW"] = x_coords.unsqueeze(0).repeat(self.patch_H, 1)
        y_coords: Int[Tensor, "PH"] = torch.arange(
            0,
            self.patch_H,
            device=self.device,
        )
        y_coords: Int[Tensor, "PH PW"] = y_coords.unsqueeze(1).repeat(1, self.patch_W)
        patch_coords: Int[Tensor, "PH PW 2"] = torch.stack([y_coords, x_coords], dim=-1)
        patch_coords: Int[Tensor, "P 2"] = patch_coords.reshape(-1, 2)
        patch_coords: Float[Tensor, "P 2"] = (
            patch_coords * self.patch_size + self.patch_size // 2
        ).float()
        return patch_coords

    @jaxtyped(typechecker=None)
    def set_ref_features(
        self,
        pixel_values: Float[Tensor, "N 3 {self.input_H} {self.input_W}"],
    ):
        self.ref_features_rlist = []
        pixel_values = pixel_values.to(self.device)

        if self.enable_shift_augmentation:
            augmented_images = []
            for img in pixel_values:
                augmented_images.append(img)  # 原图
                for dy, dx in self.shift_offsets:
                    shifted = self.shift_image(img.unsqueeze(0), dx, dy)
                    augmented_images.append(shifted.squeeze(0))
            pixel_values = torch.stack(augmented_images, dim=0)  # N*4 x 3 x H x W

        features = self.vision(pixel_values, self.feature_layers)
        # features: Float[Tensor, "L N P D"] or "L N*4 P D" when shift augmentation enabled
        for r in self.r_list:
            r_features = self.get_r_features(features, r)
            self.ref_features_rlist.append(r_features)

    def get_ref_features(self):
        return self.ref_features_rlist

    def clear_ref_features(self):
        self.ref_features_rlist = None

    @jaxtyped(typechecker=None)
    def shift_image(
        self,
        images: Float[Tensor, "B 3 H W"],
        dx: int,
        dy: int,
    ) -> Float[Tensor, "B 3 H W"]:
        """
        对图像进行平移，使用反射填充处理边界
        dx: 水平方向平移（正值为右移）
        dy: 垂直方向平移（正值为下移）
        """
        # 使用 torch.nn.functional.pad 进行反射填充
        # pad 格式: (left, right, top, bottom)
        pad_left = abs(dx)
        pad_right = abs(dx)
        pad_top = abs(dy)
        pad_bottom = abs(dy)
        # 反射填充
        padded = F.pad(
            images, (pad_left, pad_right, pad_top, pad_bottom), mode="reflect"
        )
        h, w = images.shape[2:4]

        shifted = padded[
            :, :, pad_top + dy : pad_top + h + dy, pad_left + dx : pad_left + w + dx
        ]
        return shifted

    def declare_context(self, axis: str, length: int):
        _: Bool[Tensor, axis] = torch.empty((length,), dtype=torch.bool)

    @dataclass
    class Result:
        scores: Float[Tensor, "B"]
        scores_pixel: Float[Tensor, "B H W"]

    @dataclass
    class ResultWithDetail(Result):
        # 每个图像对其他图像的 patch 级最近邻索引
        min_indices: Int[Tensor, "B L R (B-1) P"]
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
        self.declare_context("R", (len(self.r_list) if not self.config.r3indice else 1))
        self.declare_context("D", self.embed_dim)
        self.declare_context("J", self.proj_dim)

        pixel_values = pixel_values.to(self.device)
        original_batch_size = pixel_values.shape[0]

        if self.enable_shift_augmentation:
            augmented_images = []
            # 对每张图片生成增强版本
            for img in pixel_values:
                augmented_images.append(img)  # 原图
                for dy, dx in self.shift_offsets:
                    shifted = self.shift_image(img.unsqueeze(0), dx, dy)
                    augmented_images.append(shifted.squeeze(0))
            pixel_values = torch.stack(augmented_images, dim=0)  # B*4 x 3 x H x W

        features = self.vision(pixel_values, self.feature_layers)
        # features: Float[Tensor, "L B P D"] or "L B*4 P D" when shift augmentation enabled
        if self.config.match_patch != None:
            start_time = time.time()
            assert (
                len(features) == 1
            ), "match_patch_mat only supports single-layer features."
            features_m = features.squeeze(0)
            feat_ref = features_m[0]
            mask_ref = compute_background_mask(
                feat_ref,
                grid_size=(self.patch_H, self.patch_W),
            )
            matrixes = []
            for i, feat in enumerate(features_m[1:]):
                try:
                    start_time = time.time()
                    mask = compute_background_mask(
                        feat,
                        grid_size=(self.patch_H, self.patch_W),
                    )
                    # print("Computed background masks in %.2f seconds." % (time.time() - start_time))
                    start_time = time.time()
                    match_patch_mat, score = get_match_patch_mat(
                        feat_ref,
                        feat,
                        self.config.input_image_size,
                        patch_size=self.patch_size,
                        mask1=mask_ref,
                        mask2=mask,
                        topk=5,
                    )
                    # print("Computed match patch matrix in %.2f seconds." % (time.time() - start_time))
                except ValueError as e:
                    print(f"图像 {i+1} 匹配失败，跳过匹配变换: {e}")
                    match_patch_mat = np.eye(2, 3, dtype=np.float32)
                matrixes.append(match_patch_mat)
            # print("Computed match patch matrices in %.2f seconds." % (time.time() - start_time))
            if (
                self.config.match_patch == "distonly"
                and self.config.offset_distance is not None
            ):
                patch_coords_list = [self.patch_origin_coords]
                for mat in matrixes:
                    pixel_coords = cv2.transform(
                        self.patch_origin_coords.unsqueeze(1).cpu().numpy(), mat
                    ).squeeze(1)
                    patch_coords_list.append(
                        torch.from_numpy(pixel_coords).to(self.device)
                    )
                self.pixel_coords_tensor: Float[Tensor, "B P 2"] = torch.stack(
                    patch_coords_list
                )
                # print("Computed patch coordinates in %.2f seconds." % (time.time() - start_time))
            elif self.config.match_patch == "recompute":
                pixel_values_transformed = [pixel_values[0]]
                for img, mat in zip(pixel_values[1:], matrixes):
                    transformed_img = cv2.warpAffine(
                        img.cpu().numpy().transpose(1, 2, 0),
                        mat,
                        dsize=self.config.input_image_size.hw(),
                        flags=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REFLECT,
                    ).transpose(2, 0, 1)
                    pixel_values_transformed.append(
                        torch.from_numpy(transformed_img).to(self.device)
                    )
                pixel_values = torch.stack(pixel_values_transformed, dim=0)
                features = self.vision(pixel_values, self.feature_layers)
                features: Float[Tensor, "L B P D"]
                # print("Recomputed features in %.2f seconds." % (time.time() - start_time))
        scores_list = []
        min_indices_list = []
        topmink_indices_list = []
        topmink_scores_list = []
        ref_min_indices = None
        with jaxtyped("context"):
            for r_i, r in enumerate(self.r_list):
                r_features: Float[Tensor, "L B P D"] = self.get_r_features(features, r)
                ref_features = None
                if self.ref_features_rlist is not None:
                    ref_features = self.ref_features_rlist[r_i][
                        :, 0 : r_features.shape[1] - 1, ...
                    ]
                r_scores, r_min_indices, r_topmink_indices, r_topmink_scores = self.MSM(
                    r_features,
                    ref_features=ref_features,
                    ref_min_indices=ref_min_indices,
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
            min_indices: Int[Tensor, "R L B P (B-1)"] = torch.stack(
                min_indices_list, dim=0
            )
            min_indices: Int[Tensor, "B L R (B-1) P"] = min_indices.permute(
                2, 1, 0, 4, 3
            )
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
            scores = torch.mean(torch.stack(scores_list, dim=0), dim=0)

        if self.enable_shift_augmentation:
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
                min_indices,
                max_indices_image_level,
                topmink_indices,
                topmink_scores,
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
        feature_layers: list[int],
    ) -> Float[Tensor, "L B P D"]:
        if self.config.is_dino:
            features = self.vision_encoder(pixel_values)
            features = [features]
        else:
            cls_tokens, features = self.vision_encoder(pixel_values, feature_layers)
        return torch.stack(features, dim=0)

    @jaxtyped(typechecker=None)
    def get_r_features(
        self,
        features: Float[Tensor, "L XN P D"],
        r: int,
    ) -> Float[Tensor, "L XN P D"]:
        r_features_list = []
        for l_features in features:
            r_l_features = self.LNAMD(l_features, r=r)
            r_features_list.append(r_l_features)
        r_features: Float[Tensor, "L XN P D"] = torch.stack(r_features_list, dim=0)
        r_features /= r_features.norm(dim=-1, keepdim=True)
        return r_features

    @jaxtyped(typechecker=None)
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

    @jaxtyped(typechecker=None)
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

    @jaxtyped(typechecker=None)
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
            # 计算需要排除的索引（同一原图的其他增强版本）
            exclude_indices = None
            if self.enable_shift_augmentation:
                # i对应的原图索引
                original_idx = i // 4
                # 排除同一原图的4个增强版本
                exclude_indices = list(range(original_idx * 4, (original_idx + 1) * 4))
                exclude_indices.remove(i)  # 不排除自己

            scores, indices, topmink_indices, topmink_scores = self.compute_score(
                features,
                i,
                ref_min_indices[:, i, ...] if ref_min_indices is not None else None,
                ref_features,
                exclude_indices,
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
        exclude_indices: list[int] | None = None,
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
        distances: Float[Tensor, "L P (B-1)*P"] = cdist(feat, refs, p=2)
        distances: Float[Tensor, "L P (B-1) P"] = distances.view(
            *distances.shape[0:2], -1, self.patch_num
        )
        if self.config.offset_distance is not None:
            if self.config.match_patch == "distonly":
                patch_coords = self.pixel_coords_tensor[i]
                ref_patch_coords = torch.cat(
                    [
                        self.pixel_coords_tensor[:i],
                        self.pixel_coords_tensor[i + 1 :],
                    ],
                    dim=0,
                ).reshape(-1, 2)
                patch_distances: Float[Tensor, "P (B-1)*P"] = cdist(
                    patch_coords,
                    ref_patch_coords,
                    p=2,
                )
                patch_distances: Float[Tensor, "1 P (B-1) P"] = patch_distances.reshape(
                    1, self.patch_num, -1, self.patch_num
                )
                patch_distances = (
                    patch_distances * self.config.offset_distance / self.patch_size
                )
            else:
                patch_distances = self.patch_offset_distances
            distances += patch_distances

        # 如果有需要排除的索引，先将对应位置的距离设为无穷大
        if exclude_indices is not None:
            for exclude_idx in exclude_indices:
                # 计算 exclude_idx 在 refs 中的位置
                if exclude_idx < i:
                    ref_pos = exclude_idx
                elif exclude_idx > i:
                    ref_pos = exclude_idx - 1
                else:
                    continue  # 跳过自己（不应该发生）

                # 将该位置的所有 patch 距离设为无穷大
                distances[:, :, ref_pos, :] = 9999

        if ref_match_indices is not None:
            match_indices = ref_match_indices
        else:
            match_indices: Int[Tensor, "L P (B-1)"] = torch.argmin(distances, dim=-1)
        scores: Float[Tensor, "L P (B-1)"] = distances.gather(
            dim=-1, index=match_indices.unsqueeze(-1)
        ).squeeze(-1)
        k_max = max(1, int(self.topmin_max * scores.shape[-1]))
        k_min = min(k_max - 1, int(self.topmin_min * scores.shape[-1]))
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
        if config.offset_distance is not None:
            name += f"(od{config.offset_distance})"
        if config.match_patch is not None:
            name += f"({config.match_patch})"
        if config.shift_augmentation:
            name += "(shift)"
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
        self.train_indices = None

    def get_train_indices(self):
        return self.train_indices

    def __call__(
        self, images: Float[torch.Tensor, "N C H W"], class_name: str
    ) -> DetectionResult:
        self.model.eval()
        with torch.no_grad(), torch.autocast("cuda"):
            if self.last_class_name != class_name and self.const_feature:
                self.last_class_name = class_name
                if self.train_data is not None:
                    train_tensor_dataset = self.train_data(class_name, self.transform)
                    self.train_indices = torch.randperm(len(train_tensor_dataset))[
                        : images.shape[0] - 1
                    ].tolist()
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
                    self.model.compute_patch_offset_distance(
                        self.model.config.input_image_size,
                        result.min_indices.reshape(-1, result.min_indices.shape[-1]),
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
