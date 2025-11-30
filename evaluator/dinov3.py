from pathlib import Path
from typing import Any, List, Optional, Tuple
import torch
import torch.nn as nn
from jaxtyping import Float, jaxtyped

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


class DINOv3VisionTransformer(nn.Module):
    def __init__(
        self,
        model_name: str = "dinov3_vitl16",
        device: str = "cuda",
        patch_size: int = 16,
    ):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.patch_size = patch_size

        if model_name == "dinov3_vitl16":
            weight_name = "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

        # 从 torch.hub 加载预训练的 DINOv3 模型
        self.model: Any = torch.hub.load(
            str(REPO_DIR),
            model_name,
            source="local",
            weights=str(WEIGHTS_DIR / weight_name),
        )
        self.model.eval()
        self.model.to(device)

    @jaxtyped(typechecker=None)
    def forward(
        self,
        pixel_values: Float[torch.Tensor, "N C H W"],
    ) -> Float[torch.Tensor, "N P D"]:
        with torch.inference_mode():
            # 确保输入在正确的设备上
            pixel_values = pixel_values.to(self.device)

            # 使用 DINOv2 的 get_intermediate_layers 方法
            # n=1 表示获取最后 1 层的输出 (实际是倒数第二层)
            # return_class_token=False 表示只返回 patch tokens
            features_list = self.model.get_intermediate_layers(
                pixel_values, n=1, return_class_token=False
            )

            # features_list[0] 形状: [N, num_patches, embed_dim]
            return features_list[0]

    def get_embed_dim(self) -> int:
        # 根据模型名称返回嵌入维度
        if self.model_name == "dinov3_vitl16":
            return 1024
        else:
            raise ValueError(f"Unsupported model_name: {self.model_name}")

    def get_patch_size(self) -> int:
        return self.patch_size
