from pathlib import Path
from typing import Any, List, Optional, Tuple
import torch
import torch.nn as nn
from jaxtyping import Float, jaxtyped

from common.utils import generate_call_signature
from evaluator.vit import VisionTransformerBase

"""
import torch

REPO_DIR = <PATH/TO/A/LOCAL/DIRECTORY/WHERE/THE/DINOV3/REPO/WAS/CLONED>

# DINOv3 ViT models pretrained on web images
dinov3_vits16 = torch.hub.load(REPO_DIR, 'dinov3_vits16', source='local', weights=<CHECKPOINT/URL/OR/PATH>)
dinov3_vits16plus = torch.hub.load(REPO_DIR, 'dinov3_vits16plus', source='local', weights=<CHECKPOINT/URL/OR/PATH>)
dinov3_vitb16 = torch.hub.load(REPO_DIR, 'dinov3_vitb16', source='local', weights=<CHECKPOINT/URL/OR/PATH>)
dinov3_vitl16 = torch.hub.load(REPO_DIR, 'dinov3_vitl16', source='local', weights=<CHECKPOINT/URL/OR/PATH>)
dinov3_vith16plus = torch.hub.load(REPO_DIR, 'dinov3_vith16plus', source='local', weights=<CHECKPOINT/URL/OR/PATH>)
dinov3_vit7b16 = torch.hub.load(REPO_DIR, 'dinov3_vit7b16', source='local', weights=<CHECKPOINT/URL/OR/PATH>)

# DINOv3 ConvNeXt models pretrained on web images
dinov3_convnext_tiny = torch.hub.load(REPO_DIR, 'dinov3_convnext_tiny', source='local', weights=<CHECKPOINT/URL/OR/PATH>)
dinov3_convnext_small = torch.hub.load(REPO_DIR, 'dinov3_convnext_small', source='local', weights=<CHECKPOINT/URL/OR/PATH>)
dinov3_convnext_base = torch.hub.load(REPO_DIR, 'dinov3_convnext_base', source='local', weights=<CHECKPOINT/URL/OR/PATH>)
dinov3_convnext_large = torch.hub.load(REPO_DIR, 'dinov3_convnext_large', source='local', weights=<CHECKPOINT/URL/OR/PATH>)

# DINOv3 ViT models pretrained on satellite imagery
dinov3_vitl16 = torch.hub.load(REPO_DIR, 'dinov3_vitl16', source='local', weights=<CHECKPOINT/URL/OR/PATH>)
dinov3_vit7b16 = torch.hub.load(REPO_DIR, 'dinov3_vit7b16', source='local', weights=<CHECKPOINT/URL/OR/PATH>)
"""

REPO_DIR = Path("~/dinov3").expanduser().resolve()

WEIGHTS_DIR = Path("~/hdd/dinov3_weights").expanduser()


class DINOv3VisionTransformer(VisionTransformerBase):
    def __init__(
        self,
        model_name: str = "dinov3_vitl16",
    ):
        super().__init__()
        self.model_name = model_name

        if model_name == "dinov3_vitl16":
            weight_name = "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
            self.patch_size = 16
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

        # 从 torch.hub 加载预训练的 DINOv3 模型
        self.model: Any = torch.hub.load(
            str(REPO_DIR),
            model_name,
            source="local",
            weights=str(WEIGHTS_DIR / weight_name),
        )

    @jaxtyped(typechecker=None)
    def forward(
        self,
        pixel_values: Float[torch.Tensor, "N C H W"],
        output_layers: list[int] | None = None,
    ) -> list[Float[torch.Tensor, "N P D"]]:
        # 如果未指定 output_layers，返回最后一层
        if output_layers is None:
            output_layers = [self.get_layer_num() - 1]

        # 使用 get_intermediate_layers 获取指定层的输出
        # output_layers: 层索引列表，0 表示第一个 transformer block
        features_list = self.model.get_intermediate_layers(
            pixel_values, n=output_layers, return_class_token=False
        )
        features_list = list(features_list)

        return features_list

    @generate_call_signature(forward)
    def __call__(self): ...

    def get_embed_dim(self) -> int:
        # 根据模型名称返回嵌入维度
        if self.model_name == "dinov3_vitl16":
            return 1024
        else:
            raise ValueError(f"Unsupported model_name: {self.model_name}")

    def get_patch_size(self) -> int:
        return self.patch_size

    def get_layer_num(self) -> int:
        return self.model.n_blocks
