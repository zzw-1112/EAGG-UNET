# -*- coding: utf-8 -*-
"""EAGG-UNet: UNet backbone + Gaussian-adjacency GCN + edge enhancement + GRRM.

Components
----------
GraphConvolution        Standard graph convolution (``bmm(adj, xW)``).
GCN                     Two graph-conv layers with LayerNorm / GELU / residual.
GenAdj_Gaussian         Learnable Gaussian adjacency built from spatial distance
                        and feature similarity, fused by a learnable weight.
EdgeEnhance             Channel + spatial attention on a fixed edge kernel.
GraphResidualRefinement (GRRM) channel re-calibration + depthwise local
                        refinement with a learnable residual scale.
Encoder / Decoder       U-Net stage blocks.
EAGG                    The full segmentation network.
"""

import logging
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============ 1. Basic graph convolution ============
class GraphConvolution(nn.Module):
    """Standard graph convolution layer: ``A @ (X W) + b``."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, mode='fan_out', nonlinearity='relu')
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            b, c, h, w = x.size()
            x = x.view(b, c, -1).permute(0, 2, 1)

        support = torch.matmul(x, self.weight)
        output = torch.bmm(adj, support)

        if self.bias is not None:
            output = output + self.bias

        return output


# ============ 2. GCN block ============
class GCN(nn.Module):
    def __init__(self, in_channels: int, hidden: int, out_channels: int,
                 dropout: float = 0.2):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(in_channels, hidden)
        self.gc2 = GraphConvolution(hidden, out_channels)

        self.layer_norm1 = nn.LayerNorm(hidden)
        self.layer_norm2 = nn.LayerNorm(out_channels)

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        self.residual_proj = None
        if in_channels != out_channels:
            self.residual_proj = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        original_shape = None
        if x.dim() == 4:
            b, c, h, w = x.size()
            original_shape = (b, c, h, w)
            x = x.view(b, c, -1).permute(0, 2, 1)

        residual = x

        x = self.gc1(x, adj)
        x = self.layer_norm1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.gc2(x, adj)
        x = self.layer_norm2(x)
        x = self.activation(x)
        x = self.dropout(x)

        if self.residual_proj is not None:
            residual = self.residual_proj(residual)
        x = x + residual

        if original_shape is not None:
            b, c, h, w = original_shape
            x = x.permute(0, 2, 1).view(b, -1, h, w)

        return x


# ============ 3. Gaussian adjacency generator ============
class GenAdj_Gaussian(nn.Module):
    """Build a learnable adjacency matrix A in [B, N, N], N = H * W.

    Two sources are fused:
      * spatial: Gaussian kernel over normalised pixel coordinates
      * feature: Gaussian kernel over cosine similarity of learned features
    """

    def __init__(self, in_channels: int,
                 sigma_spatial: float = 1.0,
                 sigma_feature: float = 0.5,
                 use_spatial: bool = True,
                 use_feature: bool = True,
                 spatial_weight: float = 0.5,
                 feature_weight: float = 0.5):
        super(GenAdj_Gaussian, self).__init__()

        self.use_spatial = use_spatial
        self.use_feature = use_feature
        self.spatial_weight = spatial_weight
        self.feature_weight = feature_weight

        self.feature_encoder = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.log_sigma_spatial = nn.Parameter(torch.log(torch.tensor(sigma_spatial)))
        self.log_sigma_feature = nn.Parameter(torch.log(torch.tensor(sigma_feature)))
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        n = h * w

        if self.use_spatial:
            # Build coordinates on the same device as x.
            coords_y, coords_x = torch.meshgrid(
                torch.arange(h, device=x.device),
                torch.arange(w, device=x.device),
                indexing='ij'
            )
            coords = torch.stack([coords_x, coords_y], dim=0).float()
            coords = coords.view(2, -1).permute(1, 0)

            coords[:, 0] = coords[:, 0] / (w - 1) if w > 1 else 0
            coords[:, 1] = coords[:, 1] / (h - 1) if h > 1 else 0

            dist_spatial = torch.cdist(coords, coords, p=2)
            dist_spatial = dist_spatial.unsqueeze(0).expand(b, -1, -1)

            sigma_spatial = torch.exp(self.log_sigma_spatial)
            adj_spatial = torch.exp(-dist_spatial / (2 * sigma_spatial ** 2))
        else:
            adj_spatial = torch.ones(b, n, n, device=x.device) / n

        if self.use_feature:
            feat = self.feature_encoder(x)
            feat = feat.view(b, 64, n)
            feat = F.normalize(feat, p=2, dim=1)

            sim_feature = torch.matmul(feat.transpose(1, 2), feat)
            dist_feature = 1 - sim_feature
            dist_feature = torch.clamp(dist_feature, min=0)

            sigma_feature = torch.exp(self.log_sigma_feature)
            adj_feature = torch.exp(-dist_feature / (2 * sigma_feature ** 2))
        else:
            adj_feature = torch.ones(b, n, n, device=x.device) / n

        if self.use_spatial and self.use_feature:
            weight = torch.sigmoid(self.fusion_weight)
            adj = weight * adj_spatial + (1 - weight) * adj_feature
        elif self.use_spatial:
            adj = adj_spatial
        else:
            adj = adj_feature

        adj = adj + 1e-8
        adj = F.softmax(adj, dim=-1)

        return adj


# ============ 4. Edge enhancement ============
class EdgeEnhance(nn.Module):
    """Edge-aware enhancement: fixed edge kernel + channel/spatial attention."""

    def __init__(self, channels, reduction=4):
        super(EdgeEnhance, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.edge_conv = nn.Conv2d(channels, channels, kernel_size=3,
                                   padding=1, groups=channels, bias=False)

        # Initialised with a Sobel-x kernel (a Laplacian variant was also
        # evaluated). Kept trainable so the edge response can adapt.
        with torch.no_grad():
            kernel = torch.tensor([
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ], dtype=torch.float32).view(1, 1, 3, 3)
            self.edge_conv.weight.data = kernel.expand(channels, 1, 3, 3).clone()
            self.edge_conv.weight.requires_grad = True

        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        edge = self.edge_conv(x)
        edge_abs = torch.abs(edge)

        ca = self.channel_attention(x)
        x_ca = x * ca

        sa = self.spatial_attention(edge_abs)
        x_sa = x * sa

        edge_enhanced = x + self.gamma * edge_abs

        combined = torch.cat([x_ca, x_sa], dim=1)
        out = self.fusion(combined)
        out = out + edge_enhanced

        return out


# ============ 5. Graph Residual Refinement Module (GRRM) ============
class GraphResidualRefinement(nn.Module):
    """Graph Residual Refinement Module.

    Post-processing for the GCN output:
      1. channel re-calibration,
      2. local depthwise-separable convolution to restore spatial structure,
      3. learnable residual connection against GCN over-smoothing,
      4. small parameter overhead.
    """

    def __init__(self, channels, reduction=16):
        super(GraphResidualRefinement, self).__init__()

        hidden_channels = max(channels // reduction, 32)

        # ===== Channel attention =====
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # ===== Local spatial refinement =====
        self.local_refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                      groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels)
        )

        # ===== Feature fusion =====
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # ===== Learnable residual scaling =====
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        identity = x

        channel_weight = self.channel_attention(x)
        x_channel = x * channel_weight

        x_local = self.local_refine(x)

        x_fused = torch.cat([x_channel, x_local], dim=1)
        x_refined = self.fusion(x_fused)

        out = identity + self.gamma * x_refined

        return out


# ============ 6. Encoder / Decoder ============
class Encoder(nn.Module):
    """Two conv-BN-ReLU blocks + EdgeEnhance, returns (features, pooled)."""

    def __init__(self, in_channels: int, out_channels: int):
        super(Encoder, self).__init__()

        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.edge_enhance = EdgeEnhance(channels=out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, ceil_mode=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.down_conv(x)
        out = self.edge_enhance(out)
        out_pool = self.pool(out)
        return out, out_pool


class Decoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels,
                                     kernel_size=2, stride=2)
        self.up_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_copy: torch.Tensor, x: torch.Tensor,
                interpolate: bool = True) -> torch.Tensor:
        out = self.up(x)
        if interpolate:
            out = F.interpolate(out, size=(x_copy.size(2), x_copy.size(3)),
                                mode="bilinear", align_corners=True)
        else:
            diffY = x_copy.size()[2] - out.size()[2]
            diffX = x_copy.size()[3] - out.size()[3]
            out = F.pad(out, (diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffY - diffY // 2))

        out = torch.cat([x_copy, out], dim=1)
        out_conv = self.up_conv(out)
        return out_conv


# ============ 7. Base model ============
class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self):
        raise NotImplementedError

    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nbr_params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info(f'Nbr of trainable parameters: {nbr_params}')

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nbr_params = sum([np.prod(p.size()) for p in model_parameters])
        return super(BaseModel, self).__str__() + f"\nNbr of trainable parameters: {nbr_params}"

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()


# ============ 8. EAGG-UNet ============
class EAGG(BaseModel):
    """EAGG-UNet.

    U-Net encoder/decoder with edge enhancement, a Gaussian-adjacency graph
    bottleneck and graph residual refinement.

    Args:
        num_classes: number of output classes.
        in_channels: number of input bands (5 for the LCMA dataset).
        freeze_bn: freeze BatchNorm statistics at construction time.
        sigma_spatial / sigma_feature: initial Gaussian bandwidths.
        use_spatial / use_feature: which adjacency sources to combine.
        gcn_spatial_size: side length of the square graph node grid.
        **kwargs: accepted and ignored so that older configs (which may carry
            keys such as ``top_k``) keep loading without errors.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int = 5,
                 freeze_bn: bool = False,
                 sigma_spatial: float = 0.3,
                 sigma_feature: float = 0.8,
                 use_spatial: bool = True,
                 use_feature: bool = True,
                 gcn_spatial_size: int = 32,
                 **kwargs):
        super(EAGG, self).__init__()

        # ===== U-Net encoder =====
        self.down1 = Encoder(in_channels, 64)
        self.down2 = Encoder(64, 128)
        self.down3 = Encoder(128, 256)
        self.down4 = Encoder(256, 512)

        # ===== Bottleneck =====
        self.middle_conv = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # ===== GCN down-sampling =====
        self.gcn_spatial_size = gcn_spatial_size
        self.gcn_downsample = nn.AdaptiveAvgPool2d((gcn_spatial_size, gcn_spatial_size))

        # ===== Graph branch =====
        self.adj_encoder = GenAdj_Gaussian(
            in_channels=1024,
            sigma_spatial=sigma_spatial,
            sigma_feature=sigma_feature,
            use_spatial=use_spatial,
            use_feature=use_feature
        )
        self.gcn_encoder = GCN(1024, 2048, 1024, dropout=0.2)

        # ===== Post-GCN refinement =====
        self.post_gcn_feature_select = GraphResidualRefinement(
            channels=1024
        )

        # ===== U-Net decoder =====
        self.up1 = Decoder(1024, 512)
        self.up2 = Decoder(512, 256)
        self.up3 = Decoder(256, 128)
        self.up4 = Decoder(128, 64)

        # ===== Output head =====
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

        self._initialize_weights()

        if freeze_bn:
            self.freeze_bn()

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f'EAGG initialized with {total_params:,} parameters')

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ===== Encode =====
        x1, x = self.down1(x)
        x2, x = self.down2(x)
        x3, x = self.down3(x)
        x4, x = self.down4(x)

        # ===== Bottleneck =====
        x = self.middle_conv(x)

        # ===== Graph branch =====
        b, c, h, w = x.size()

        x_small = self.gcn_downsample(x)
        adj = self.adj_encoder(x_small)
        x_flat = x_small.view(b, c, -1).permute(0, 2, 1)
        x_gcn = self.gcn_encoder(x_flat, adj)
        x_gcn = x_gcn.permute(0, 2, 1).view(
            b, c, self.gcn_spatial_size, self.gcn_spatial_size
        )

        x_gcn = self.post_gcn_feature_select(x_gcn)

        x = F.interpolate(x_gcn, size=(h, w), mode='bilinear', align_corners=True)

        # ===== Decode =====
        x = self.up1(x4, x)
        x = self.up2(x3, x)
        x = self.up3(x2, x)
        x = self.up4(x1, x)

        # ===== Head =====
        x = self.final_conv(x)
        return x


# Backward-compatible alias: earlier code referred to the network as
# ``UNetWithFeatureSelect``.
UNetWithFeatureSelect = EAGG


if __name__ == '__main__':
    # Quick shape / parameter sanity check.
    model = EAGG(num_classes=20, in_channels=5)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'trainable parameters: {total:,}')
    dummy = torch.randn(1, 5, 128, 128)
    with torch.no_grad():
        out = model(dummy)
    print('input :', tuple(dummy.shape))
    print('output:', tuple(out.shape))
