"""Fast-SCNN model."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvNormAct(nn.Module):
    """Convolution followed by batch normalization and ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DepthwiseSeparableDownsample(nn.Module):
    """Depthwise separable block used in the early downsampling path."""

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.depthwise_norm = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pointwise_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.depthwise_norm(x)
        x = self.pointwise(x)
        x = self.pointwise_norm(x)
        return self.activation(x)


class BottleneckBlock(nn.Module):
    """Mobile inverted bottleneck block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: int,
    ) -> None:
        super().__init__()
        if stride not in (1, 2):
            raise ValueError(f"Unsupported stride for bottleneck block: {stride}")

        expanded_channels = int(round(in_channels * expand_ratio))
        use_skip = stride == 1 and in_channels == out_channels

        layers: list[nn.Module] = []
        if expand_ratio != 1:
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels, expanded_channels, kernel_size=1, bias=False
                    ),
                    nn.BatchNorm2d(expanded_channels),
                    nn.ReLU(inplace=True),
                ]
            )
        else:
            expanded_channels = in_channels

        layers.extend(
            [
                nn.Conv2d(
                    expanded_channels,
                    expanded_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=expanded_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(expanded_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(expanded_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            ]
        )

        self.use_skip = use_skip
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        transformed = self.block(x)
        if self.use_skip:
            return x + transformed
        return transformed


class PyramidPooling(nn.Module):
    """Multi-scale context aggregation by adaptive pooling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_sizes: tuple[int, ...] = (1, 2, 3, 6),
    ) -> None:
        super().__init__()
        self.pool_branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size)),
                    nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
                )
                for pool_size in pool_sizes
            ]
        )
        total_channels = in_channels * (len(pool_sizes) + 1)
        self.project = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_size = x.shape[-2:]
        pooled_features = [x]
        for branch in self.pool_branches:
            pooled = branch(x)
            pooled = F.interpolate(
                pooled,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
            pooled_features.append(pooled)
        return self.project(torch.cat(pooled_features, dim=1))


class DownsamplingStem(nn.Module):
    """Initial layers that reduce the spatial resolution quickly."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.initial_conv = ConvNormAct(in_channels, 32, stride=2)
        self.separable_block1 = DepthwiseSeparableDownsample(32, 48, stride=2)
        self.separable_block2 = DepthwiseSeparableDownsample(48, 64, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial_conv(x)
        x = self.separable_block1(x)
        return self.separable_block2(x)


class ContextEncoder(nn.Module):
    """Low-resolution feature extractor with global context pooling."""

    def __init__(self) -> None:
        super().__init__()
        self.stage1 = nn.Sequential(
            BottleneckBlock(64, 64, stride=2, expand_ratio=6),
            BottleneckBlock(64, 64, stride=1, expand_ratio=6),
            BottleneckBlock(64, 64, stride=1, expand_ratio=6),
        )
        self.stage2 = nn.Sequential(
            BottleneckBlock(64, 96, stride=2, expand_ratio=6),
            BottleneckBlock(96, 96, stride=1, expand_ratio=6),
            BottleneckBlock(96, 96, stride=1, expand_ratio=6),
        )
        self.stage3 = nn.Sequential(
            BottleneckBlock(96, 128, stride=1, expand_ratio=6),
            BottleneckBlock(128, 128, stride=1, expand_ratio=6),
            BottleneckBlock(128, 128, stride=1, expand_ratio=6),
        )
        self.pyramid_pooling = PyramidPooling(128, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.pyramid_pooling(x)


class FeatureFusion(nn.Module):
    """Fuse high-resolution detail with low-resolution context."""

    def __init__(self) -> None:
        super().__init__()
        self.low_resolution_refine = ConvNormAct(
            128,
            128,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=128,
        )
        self.low_resolution_project = nn.Conv2d(128, 128, kernel_size=1, bias=True)
        self.high_resolution_project = nn.Conv2d(64, 128, kernel_size=1, bias=True)
        self.activation = nn.ReLU(inplace=True)

    def forward(
        self,
        high_resolution_features: torch.Tensor,
        low_resolution_features: torch.Tensor,
    ) -> torch.Tensor:
        target_size = high_resolution_features.shape[-2:]
        low_resolution_features = F.interpolate(
            low_resolution_features,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        low_resolution_features = self.low_resolution_refine(low_resolution_features)
        low_resolution_features = self.low_resolution_project(low_resolution_features)
        high_resolution_features = self.high_resolution_project(
            high_resolution_features
        )
        return self.activation(high_resolution_features + low_resolution_features)


class SegmentationHead(nn.Module):
    """Prediction head for the fused features."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.block1 = ConvNormAct(
            128,
            128,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=128,
        )
        self.block2 = ConvNormAct(
            128,
            128,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=128,
        )
        self.classifier = nn.Conv2d(128, num_classes, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        return self.classifier(x)


class FastSCNN(nn.Module):
    """Fast-SCNN segmentation model."""

    def __init__(self, in_channels: int = 3, n_classes: int = 2) -> None:
        super().__init__()
        self.downsampling_stem = DownsamplingStem(in_channels)
        self.context_encoder = ContextEncoder()
        self.feature_fusion = FeatureFusion()
        self.segmentation_head = SegmentationHead(n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        high_resolution_features = self.downsampling_stem(x)
        low_resolution_features = self.context_encoder(high_resolution_features)
        fused_features = self.feature_fusion(
            high_resolution_features,
            low_resolution_features,
        )
        logits = self.segmentation_head(fused_features)
        return F.interpolate(
            logits,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )
