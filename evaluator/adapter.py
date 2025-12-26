import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, jaxtyped

from common.utils import generate_call_signature


class SimpleAdapter(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super(SimpleAdapter, self).__init__()
        self.fc = nn.Sequential(nn.Linear(c_in, c_out, bias=False), nn.LeakyReLU())

    def forward(self, x):
        x = self.fc(x)
        return x


class VisionAdapter(nn.Module):
    def __init__(self, embed_dim: int, bottleneck_dim: int = 768):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(embed_dim, bottleneck_dim, bias=False),  # 降维
            nn.LeakyReLU(inplace=False),  # 激活函数
        )
        self.fc2 = nn.Sequential(
            nn.Linear(bottleneck_dim, embed_dim, bias=False),  # 升维还原
            nn.LeakyReLU(inplace=False),  # 激活函数
        )

    def forward(self, x) -> Tensor:
        x = self.fc1(x)
        y = self.fc2(x)
        return y

    @generate_call_signature(forward)
    def __call__(self): ...


class VisionConvAdapter(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        bottleneck_dim: int = 768,
        kernel_size: int = 5,
        version: int = 1,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        padding = kernel_size // 2
        if version == 1:
            self.net = nn.Sequential(
                nn.Conv2d(
                    embed_dim, bottleneck_dim, kernel_size, padding=padding, bias=False
                ),
                nn.LeakyReLU(inplace=False),
                nn.Conv2d(
                    bottleneck_dim, embed_dim, kernel_size, padding=padding, bias=False
                ),
                # 移除最后一层激活，保留特征完整性
            )
        elif version == 2:
            # 添加最后一层 LeakyReLU
            self.net = nn.Sequential(
                nn.Conv2d(
                    embed_dim, bottleneck_dim, kernel_size, padding=padding, bias=False
                ),
                nn.LeakyReLU(inplace=False),
                nn.Conv2d(
                    bottleneck_dim, embed_dim, kernel_size, padding=padding, bias=False
                ),
                nn.LeakyReLU(inplace=False),
            )
        elif version == 3:
            # 先过一层线性层，再过卷积层
            self.linear = nn.Sequential(
                nn.Linear(embed_dim, bottleneck_dim, bias=False),
                nn.LeakyReLU(inplace=False),
            )
            self.conv = nn.Sequential(
                nn.Conv2d(
                    bottleneck_dim, embed_dim, kernel_size, padding=padding, bias=False
                ),
                nn.LeakyReLU(inplace=False),
            )
        else:
            raise ValueError("version not supported")
        self.version = version            

    @jaxtyped(typechecker=None)
    def forward(
        self, x: Float[Tensor, "N P D"], grid_size: tuple[int, int]
    ) -> Float[Tensor, "N P D"]:
        # x: [N, P, D] -> [Batch, Patches, Dim]
        if self.version == 1:
            # 变换维度以适应 Conv2d: [N, D, H, W]
            x = x.transpose(-2, -1).reshape(*x.shape[:-1], *grid_size)
            x = self.net(x)
            x = x.reshape(*x.shape[:-2], -1).transpose(-2, -1)
            return x
        elif self.version == 2:
            x = x.transpose(-2, -1).reshape(*x.shape[:-1], *grid_size)
            x = self.net(x)
            x = x.reshape(*x.shape[:-2], -1).transpose(-2, -1)
            return x
        elif self.version == 3:
            x = self.linear(x)
            x = x.transpose(-2, -1).reshape(*x.shape[:-1], *grid_size)
            x = self.conv(x)
            x = x.reshape(*x.shape[:-2], -1).transpose(-2, -1)
            return x
        else:
            raise ValueError("version not supported")

    @generate_call_signature(forward)
    def __call__(self): ...