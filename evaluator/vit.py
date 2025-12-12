from torch import nn, Tensor
import torch
from jaxtyping import Float
from abc import abstractmethod

from common.utils import generate_call_signature


class VisionTransformerBase(nn.Module):

    @abstractmethod
    def forward(self, pixel_values: Float[Tensor, "N 3 H W"]) -> Float[Tensor, "N P D"]: ...

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
