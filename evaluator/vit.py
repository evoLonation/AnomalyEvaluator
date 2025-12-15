from torch import nn, Tensor
import torch
from jaxtyping import Float, jaxtyped
from abc import abstractmethod

from common.utils import generate_call_signature


class VisionTransformerBase(nn.Module):

    @abstractmethod
    @jaxtyped(typechecker=None)
    def forward(
        self,
        pixel_values: Float[torch.Tensor, "N C H W"],
        output_layers: list[int] | None = None,
    ) -> list[Float[torch.Tensor, "N P D"]]: ...  # patch_tokens_list

    @generate_call_signature(forward)
    def __call__(self): ...

    @abstractmethod
    def get_embed_dim(self) -> int:
        """获取视觉Transformer的嵌入维度。"""
        ...

    @abstractmethod
    def get_patch_size(self) -> int:
        """获取视觉Transformer的Patch大小。"""
        ...

    @abstractmethod
    def get_layer_num(self) -> int:
        """获取视觉Transformer的层数。"""
        ...
