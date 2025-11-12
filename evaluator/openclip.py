"""
基于 open_clip 库的 CLIP Vision Transformer 实现
精简版本 - 使用组合模式而非继承
"""

from typing import Any, List, Optional, Tuple
import torch
import torch.nn as nn
from jaxtyping import Float, jaxtyped
import open_clip
from open_clip.transformer import Transformer as OpenCLIPTransformer
from open_clip.transformer import VisionTransformer as OpenCLIPVisionTransformer
from torchvision.transforms import Compose


class CustomTransformer(nn.Module):
    """自定义 Transformer，支持返回指定层的输出"""

    def __init__(self, original_transformer: OpenCLIPTransformer):
        super().__init__()
        self.transformer = original_transformer
        # 暴露常用属性
        self.resblocks = original_transformer.resblocks

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        output_layers: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: 输入张量
            attn_mask: 注意力掩码
            output_layers: 要输出的层索引列表，从0开始

        Returns:
            final_output: 最后一层的输出
            intermediate_outputs: 指定层的输出列表
        """
        if output_layers is None:
            output_layers = [len(self.resblocks) - 1]

        intermediate_outputs = []

        for i, r in enumerate(self.resblocks):
            x = r(x, attn_mask=attn_mask)
            if i in output_layers:
                intermediate_outputs.append(x)

        return x, intermediate_outputs


class CLIPVisionTransformer(nn.Module):
    """基于 open_clip 的 Vision Transformer，支持返回中间层特征"""

    def __init__(self, original_vit: OpenCLIPVisionTransformer):
        super().__init__()
        self.vit = original_vit

        # 暴露常用属性
        self.ln_pre = original_vit.ln_pre
        self.ln_post = original_vit.ln_post
        self.proj = original_vit.proj

        # 替换 transformer 为自定义版本
        self.transformer = CustomTransformer(original_vit.transformer)

        # 计算投影维度
        if self.proj is not None:
            self.projection_dim = self.proj.shape[1]
        else:
            self.projection_dim = original_vit.embed_dim

    def get_layer_num(self) -> int:
        """获取 transformer 层数"""
        return len(self.transformer.resblocks)

    @jaxtyped(typechecker=None)
    def forward(
        self,
        pixel_values: Float[torch.Tensor, "N C H W"],
        output_layers: Optional[List[int]] = None,
    ) -> Tuple[
        Float[torch.Tensor, "N PD"],  # cls_token
        List[Float[torch.Tensor, "N P ED"]],  # patch_tokens_list
    ]:
        """
        Args:
            pixel_values: [batch_size, channels, height, width]
            output_layers: 要输出的层索引列表，从0开始。如果为 None，输出最后一层

        Returns:
            cls_token: [batch_size, projection_dim]
            patch_tokens_list: List of [batch_size, num_patches, embed_dim]
        """
        if output_layers is None:
            output_layers = [self.get_layer_num() - 1]

        # === 从原始 VisionTransformer.forward 复制的代码 ===
        # Embedding
        x = self.vit._embeds(pixel_values)

        # Transformer with intermediate outputs
        x, intermediate_outputs = self.transformer(
            x, attn_mask=None, output_layers=output_layers
        )

        # 提取 patch tokens (排除 cls token)
        patch_tokens_list = []
        for output in intermediate_outputs:
            # output shape: [batch_size, num_patches + 1, embed_dim]
            # 去掉第一个 token (cls token)
            patch_tokens = output[:, 1:, :]
            patch_tokens_list.append(patch_tokens)

        # Pooling - 从原始代码的 _pool 方法简化而来
        # 因为默认配置 pool_type='tok', final_ln_after_pool=False, output_tokens=False
        pooled = x[:, 0, :]  # cls token

        # Layer norm (ln_post)
        pooled = self.ln_post(pooled)

        # Projection
        if self.proj is not None:
            cls_token = pooled @ self.proj
        else:
            cls_token = pooled

        return cls_token, patch_tokens_list

    def project_patches(
        self,
        patch_tokens: Float[torch.Tensor, "N P D"],
    ) -> Float[torch.Tensor, "N P D_proj"]:
        """
        对 patch tokens 应用投影

        Args:
            patch_tokens: [batch_size, num_patches, embed_dim]

        Returns:
            projected_patches: [batch_size, num_patches, projection_dim]
        """
        # 应用 layer norm
        patch_tokens = self.ln_post(patch_tokens)

        # 应用投影
        if self.proj is not None:
            projected = patch_tokens @ self.proj
        else:
            projected = patch_tokens

        return projected


def create_vision_transformer(
    model_name: str = "ViT-L-14-336",
    pretrained: str = "openai",
    image_size: Tuple[int, int] = (336, 336),
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> tuple[CLIPVisionTransformer, Compose]:
    """
    创建基于 open_clip 的 Vision Transformer

    Args:
        model_name: open_clip 模型名称，如 "ViT-L-14", "ViT-B-16" 等
        pretrained: 预训练权重，如 "openai", "laion2b_s32b_b82k" 等
        image_size: 输入图像大小 (height, width)
        device: 运行设备

    Returns:
        CLIPVisionTransformer 实例
    """
    print(f"Loading open_clip model: {model_name} with pretrained: {pretrained}")

    # 创建 open_clip 模型
    model, _, preprocessor = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=device,
        force_image_size=image_size,
    )

    print(f"Model loaded successfully")

    # 获取原始 visual 模块
    visual = model.visual

    # 创建自定义的 Vision Transformer（将原始 visual 作为成员）
    vision_transformer = CLIPVisionTransformer(visual)

    vision_transformer.to(device)

    print(f"CLIPVisionTransformer created with image size: {image_size}")
    print(f"Number of transformer layers: {vision_transformer.get_layer_num()}")
    print(f"Projection dim: {vision_transformer.projection_dim}")

    return vision_transformer, preprocessor


# 测试代码
if __name__ == "__main__":
    # 测试创建模型
    vit, transforms = create_vision_transformer()

    # 测试前向传播
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 336, 336).cuda()

    # 测试输出最后一层
    cls_token, patch_tokens_list = vit(dummy_input)
    print(f"\nTest with default output_layers:")
    print(f"CLS token shape: {cls_token.shape}")
    print(f"Number of intermediate outputs: {len(patch_tokens_list)}")
    print(f"Patch tokens shape: {patch_tokens_list[0].shape}")

    # 测试输出多个层
    cls_token, patch_tokens_list = vit(dummy_input, output_layers=[0, 11, 23])
    print(f"\nTest with output_layers=[0, 11, 23]:")
    print(f"CLS token shape: {cls_token.shape}")
    print(f"Number of intermediate outputs: {len(patch_tokens_list)}")
    for i, pt in enumerate(patch_tokens_list):
        print(f"Layer {[0, 11, 23][i]} patch tokens shape: {pt.shape}")

    # 测试 project_patches
    projected = vit.project_patches(patch_tokens_list[0])
    print(f"\nProjected patch tokens shape: {projected.shape}")

    print("\n✅ All tests passed!")
