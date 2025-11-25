from typing import Any, List, Optional, Tuple
import torch
import torch.nn as nn
from jaxtyping import Float, jaxtyped


class DINOv2VisionTransformer(nn.Module):
    """
    DINOv2 Vision Transformer wrapper for anomaly detection.
    
    基于 AnomalyDINO 的实现，封装 DINOv2 模型用于提取图像的 patch-level 特征。
    """
    
    def __init__(
        self,
        model_name: str = "dinov2_vitl14",
        device: str = "cuda",
        patch_size: int = 14,
    ):
        """
        Args:
            model_name: DINOv2 模型名称，可选:
                - dinov2_vits14: ViT-Small, patch_size=14, embed_dim=384
                - dinov2_vitb14: ViT-Base, patch_size=14, embed_dim=768
                - dinov2_vitl14: ViT-Large, patch_size=14, embed_dim=1024
                - dinov2_vitg14: ViT-Giant, patch_size=14, embed_dim=1536
            device: 设备 (cuda 或 cpu)
            patch_size: Patch 大小，默认 14
        """
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.patch_size = patch_size
        
        # 从 torch.hub 加载预训练的 DINOv2 模型
        self.model: Any = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model.eval()
        self.model.to(device)
        
    @jaxtyped(typechecker=None)
    def forward(
        self,
        pixel_values: Float[torch.Tensor, "N C H W"],
    ) -> Float[torch.Tensor, "N P D"]:
        """
        提取图像的 patch-level 特征。
        
        参考 AnomalyDINO 中 DINOv2Wrapper.extract_features 的实现:
        - 使用 get_intermediate_layers 提取倒数第二层的特征
        - 返回所有 patch tokens (不包含 [CLS] token)
        
        Args:
            pixel_values: 输入图像张量 [N, C, H, W]
                假设已经过预处理 (Resize, Normalize 等)
                
        Returns:
            特征列表，每个元素形状为 [N, num_patches, embed_dim]
            其中 num_patches = (H // patch_size) * (W // patch_size)
        """
        with torch.inference_mode():
            # 确保输入在正确的设备上
            pixel_values = pixel_values.to(self.device)
            
            # 使用 DINOv2 的 get_intermediate_layers 方法
            # n=1 表示获取最后 1 层的输出 (实际是倒数第二层)
            # return_class_token=False 表示只返回 patch tokens
            features_list = self.model.get_intermediate_layers(
                pixel_values,
                n=1,
                return_class_token=False
            )
            
            # features_list[0] 形状: [N, num_patches, embed_dim]
            return features_list[0]
