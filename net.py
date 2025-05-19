from typing import Tuple

import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        groups: int,
    ) -> None:
        super(CausalConv1d, self).__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.pad,
            dilation=dilation,
            groups=groups,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = x[..., : -self.pad]
        return x


class WaveNetResidualBlock(nn.Module):
    def __init__(
        self,
        n_channel: int,
        n_mul: int,
        frames: int,
        kernel_size: int,
        dilation_rate: int,
        n_groups: int,
    ) -> None:
        super(WaveNetResidualBlock, self).__init__()
        self.sigmoid_conv = nn.Sequential(
            nn.LayerNorm([n_channel * n_mul, frames]),
            CausalConv1d(
                n_channel * n_mul,
                n_channel * n_mul,
                kernel_size,
                dilation_rate,
                n_groups,
            ),
            nn.Sigmoid(),
        )
        self.tanh_conv = nn.Sequential(
            nn.LayerNorm([n_channel * n_mul, frames]),
            CausalConv1d(
                n_channel * n_mul,
                n_channel * n_mul,
                kernel_size,
                dilation_rate,
                n_groups,
            ),
            nn.Tanh(),
        )
        self.skip_connection = nn.Sequential(
            nn.LayerNorm([n_channel * n_mul, frames]),
            nn.Conv1d(n_channel * n_mul, n_channel, (1,), groups=n_groups),
        )
        self.residual = nn.Sequential(
            nn.LayerNorm([n_channel * n_mul, frames]),
            nn.Conv1d(n_channel * n_mul, n_channel * n_mul, (1,), groups=n_groups),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sigmoid_conv = self.sigmoid_conv(x)
        tanh_conv = self.tanh_conv(x)
        mul = torch.mul(sigmoid_conv, tanh_conv)
        skip = self.skip_connection(mul)
        residual = self.residual(mul)
        return skip, residual + x


class WaveNet(nn.Module):
    def __init__(
        self,
        n_blocks: int,
        n_channel: int,
        n_mul: int,
        frames: int,
        kernel_size: int,
        n_groups: int,
    ) -> None:
        super(WaveNet, self).__init__()
        self.n_blocks = n_blocks
        self.kernel_size = kernel_size
        self.feature_layer = nn.Sequential(
            nn.LayerNorm([n_channel, frames]),
            nn.Conv1d(n_channel, n_channel * n_mul, (1,), groups=n_groups),
        )
        self.blocks = nn.ModuleList(
            [
                WaveNetResidualBlock(
                    n_channel, n_mul, frames, kernel_size, 2**i, n_groups
                )
                for i in range(n_blocks)
            ]
        )
        self.skip_connection = nn.Sequential(
            nn.ReLU(),
            nn.LayerNorm([n_channel, frames]),
            nn.Conv1d(n_channel, n_channel, (1,), groups=n_groups),
            nn.ReLU(),
            nn.LayerNorm([n_channel, frames]),
            nn.Conv1d(n_channel, n_channel, (1,), groups=n_groups),
        )

    def get_receptive_field(self) -> int:
        rf = 1
        for _ in range(self.n_blocks):
            rf = rf * 2 + self.kernel_size - 2
        return rf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, F, T = x.shape
        x = torch.reshape(x, (B, C * F, T))
        x = self.feature_layer(x)
        skips = []
        for _, block in enumerate(self.blocks):
            skip, x = block(x)
            skips.append(skip)
        skips = torch.stack(skips).sum(0)
        output = self.skip_connection(skips)
        output = output.reshape(B, C, F, T)
        return output[..., self.get_receptive_field() - 1 : -1]


def WaveNetModel() -> nn.Module:
    n_blocks = 3
    n_channel = 10
    n_mul = 4
    frames = 63
    kernel_size = 3
    n_groups = 1
    return WaveNet(n_blocks, n_channel, n_mul, frames, kernel_size, n_groups)
