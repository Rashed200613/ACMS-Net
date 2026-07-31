import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Primitive blocks
# ---------------------------------------------------------------------------

class DepthwiseSeparableConv5x5(nn.Module):
    """5×5 Depthwise Separable Convolution (Large Scale branch)."""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels,
                kernel_size=5, padding=2,
                groups=in_channels, bias=False
            ),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class DilatedDepthwiseConv7x7(nn.Module):
    """7×7 Dilated Depthwise Convolution (d=2, Wider Context branch)."""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels,
                kernel_size=7, padding=6,
                dilation=2, groups=in_channels, bias=False
            ),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# ---------------------------------------------------------------------------
# A. Adaptive Multi-Scale Feature Extraction (AMSFE)
# ---------------------------------------------------------------------------

class AMSFEBlock(nn.Module):
    """
    4-branch multi-scale feature extraction:
        branch1 : 1×1 Conv  (Point-wise / Local)
        branch3 : 3×3 Conv  (Medium Scale)
        branch5 : 5×5 DWConv (Large Scale)
        branch7 : 7×7 Dilated DWConv, d=2 (Wider Context)

    Output channels = 4 × out_channels  (concatenation)
    """

    def __init__(self, in_channels, out_channels=42):
        super().__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.branch5 = DepthwiseSeparableConv5x5(in_channels, out_channels)
        self.branch7 = DilatedDepthwiseConv7x7(in_channels, out_channels)

    def forward(self, x):
        b1 = self.branch1(x)
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        b7 = self.branch7(x)
        return torch.cat([b1, b3, b5, b7], dim=1)   # → 4 × out_channels


# ---------------------------------------------------------------------------
# B. Cross-Scale Interaction Module (CSIM)
#    Diagram: each scale attends to all other scales via cross-attention (Q,K,V)
# ---------------------------------------------------------------------------

class CrossScaleInteraction(nn.Module):
    """
    Cross-scale attention where each scale's features act as Query,
    while the concatenated multi-scale feature map provides Key and Value.
    This matches the diagram's Q,K,V cross-scale information exchange.
    """

    def __init__(self, channels_per_scale, num_scales=4, reduction=4):
        super().__init__()

        self.num_scales = num_scales
        C = channels_per_scale
        total = C * num_scales

        # Per-scale Q projection (each scale projects its own features as query)
        self.q_proj = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(C, C // reduction),
                nn.ReLU(inplace=True)
            )
            for _ in range(num_scales)
        ])

        # Shared K, V projections from full concatenated feature map
        self.k_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(total, C // reduction),
            nn.ReLU(inplace=True)
        )
        self.v_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(total, C),
            nn.Sigmoid()
        )

    def forward(self, x, channels_per_scale):
        splits = torch.split(x, channels_per_scale, dim=1)  # list of (B, C, H, W)

        # Key and Value from full multi-scale feature map
        k = self.k_proj(x)   # (B, C // reduction)
        v = self.v_proj(x)   # (B, C)

        refined = []
        for i, q_layer in enumerate(self.q_proj):
            q = q_layer(splits[i])                              # (B, C // reduction)
            # Attention score: dot product of Q and K, normalised
            attn = torch.sigmoid((q * k).sum(dim=1, keepdim=True))  # (B, 1)
            # Gate the value with attention score and apply to this scale
            gate = (v * attn).unsqueeze(-1).unsqueeze(-1)      # (B, C, 1, 1)
            refined.append(splits[i] * gate)

        return torch.cat(refined, dim=1)                        # same shape as input


# ---------------------------------------------------------------------------
# C. Scale Attention – Adaptive Weighting
#    Diagram: w1 + w3 + w5 = 1  (branch5 and branch7 merged as one group)
# ---------------------------------------------------------------------------

class ScaleAttention(nn.Module):
    """
    Learns 3 soft weights (w1, w3, w5/7) over scale groups via
    GAP → FC → ReLU → FC → Softmax, matching the diagram's notation
    where w1 + w3 + w5 = 1.

    branch5 and branch7 are merged into a single scale group (w5)
    before weighting, then split back and each receives the same weight.
    """

    def __init__(self, channels_per_scale, num_scales=4):
        super().__init__()

        C = channels_per_scale
        total = C * num_scales

        # 3 groups: [b1], [b3], [b5 + b7]
        self.num_groups = 3

        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.fc1  = nn.Linear(total, total // 4)
        self.relu = nn.ReLU(inplace=True)
        self.fc2  = nn.Linear(total // 4, self.num_groups)

    def forward(self, x, channels_per_scale):
        B = x.size(0)
        C = channels_per_scale

        gap = self.gap(x).view(B, -1)                        # (B, total)
        w   = self.fc2(self.relu(self.fc1(gap)))              # (B, 3)
        w   = F.softmax(w, dim=1)                            # w1+w3+w5=1

        # Split into 4 per-branch feature maps
        b1, b3, b5, b7 = torch.split(x, C, dim=1)

        w1 = w[:, 0].view(B, 1, 1, 1)
        w3 = w[:, 1].view(B, 1, 1, 1)
        w5 = w[:, 2].view(B, 1, 1, 1)   # shared between branch5 and branch7

        out = torch.cat([b1 * w1, b3 * w3, b5 * w5, b7 * w5], dim=1)
        return out                                            # same shape as input


# ---------------------------------------------------------------------------
# D. SE Attention – Channel Recalibration
# ---------------------------------------------------------------------------

class SEAttention(nn.Module):
    """
    Squeeze-and-Excitation channel recalibration.
    GAP → FC → ReLU → FC → Sigmoid → channel-wise multiply.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, _, _ = x.shape
        scale = self.fc(self.gap(x)).view(B, C, 1, 1)
        return x * scale


# ---------------------------------------------------------------------------
# Full AMSFE Stage (AMSFE + CSIM + ScaleAttn + SE + Residual)
# ---------------------------------------------------------------------------

class AMSFEStage(nn.Module):
    """
    One complete AMSFE stage as shown in diagram:
        Input → AMSFE (4-branch concat)
                → CSIM
                → Scale Attention
                → SE Attention
                → (+) Residual (1×1 projection if needed)
        Output: 168 channels
    """

    def __init__(self, in_channels, branch_channels=42):
        super().__init__()

        total = branch_channels * 4   # 168

        self.amsfe      = AMSFEBlock(in_channels, branch_channels)
        self.csim       = CrossScaleInteraction(branch_channels, num_scales=4)
        self.scale_attn = ScaleAttention(branch_channels, num_scales=4)
        self.se         = SEAttention(total)

        self.residual_proj = (
            nn.Conv2d(in_channels, total, kernel_size=1, bias=False)
            if in_channels != total else nn.Identity()
        )

        self.bn_out = nn.BatchNorm2d(total)

    def forward(self, x):
        residual = self.residual_proj(x)

        out = self.amsfe(x)
        out = self.csim(out, channels_per_scale=out.size(1) // 4)
        out = self.scale_attn(out, channels_per_scale=out.size(1) // 4)
        out = self.se(out)

        out = self.bn_out(out + residual)
        return out


# ---------------------------------------------------------------------------
# Main Model – ACMS-Net
# ---------------------------------------------------------------------------

class CustomModel(nn.Module):
    """
    ACMS-Net: Adaptive Cross-Scale Multi-Scale Attention Network.

    Architecture (diagram-aligned):
        Conv Stem  (Conv3×3,32 → BN,ReLU → Conv3×3,32 → BN,ReLU → MaxPool2×2)
        AMSFE Stage-1 → output 168ch → MaxPool2×2
        Bridge: MaxPool2×2 only                          (1×1 conv removed —
            it was just channel remixing; the two MaxPools already do the
            real spatial-reduction work, per senior's note)
        AMSFE Stage-2 → output 168ch → MaxPool2×2
        Conv Block  (Conv3×3,192 → BN,ReLU → Conv3×3,192 → BN,ReLU → MaxPool2×2)
        Global Average Pooling → 192-d vector
        Classifier  (Drop0.4 → FC192 → ReLU → Drop0.4 → FC96 → ReLU → FC N)
    """

    def __init__(self, num_classes=9):
        super().__init__()

        # ── Conv Stem ────────────────────────────────────────────────────────
        self.conv_stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        # ── AMSFE Stage-1 ─────────────────────────────────────────────────
        self.amsfe_stage1 = AMSFEStage(in_channels=32, branch_channels=42)
        self.pool1 = nn.MaxPool2d(2)

        # ── Inter-stage bridge (1×1 conv removed — pure channel remixing,
        #    cut per senior's note; the two MaxPools do the real work) ──────
        self.bridge = nn.Sequential(
            nn.MaxPool2d(2)
        )

        # ── AMSFE Stage-2 ─────────────────────────────────────────────────
        self.amsfe_stage2 = AMSFEStage(in_channels=168, branch_channels=42)
        self.pool2 = nn.MaxPool2d(2)

        # ── Conv Block (reduced: 168→192→192 instead of 168→224→224) ────────
        self.conv_block = nn.Sequential(
            nn.Conv2d(168, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        # ── Global Average Pooling ───────────────────────────────────────────
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # ── Classifier (diagram: Dropout → FC → ReLU, repeated) ─────────────
        self.drop1 = nn.Dropout(0.4)
        self.fc1   = nn.Linear(192, 192)

        self.drop2 = nn.Dropout(0.4)
        self.fc2   = nn.Linear(192, 96)

        self.fc3   = nn.Linear(96, num_classes)

    def forward(self, x):

        x = self.conv_stem(x)          # (B, 32,  H/2,  W/2)

        x = self.amsfe_stage1(x)       # (B, 168, H/2,  W/2)
        x = self.pool1(x)              # (B, 168, H/4,  W/4)

        x = self.bridge(x)             # (B, 168, H/8,  W/8)

        x = self.amsfe_stage2(x)       # (B, 168, H/8,  W/8)
        x = self.pool2(x)              # (B, 168, H/16, W/16)

        x = self.conv_block(x)         # (B, 192, H/32, W/32)

        x = self.global_pool(x)        # (B, 192, 1, 1)
        x = torch.flatten(x, 1)        # (B, 192)

        # Classifier: Dropout → FC → ReLU (diagram-aligned)
        x = self.drop1(x)
        x = F.relu(self.fc1(x))

        x = self.drop2(x)
        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        return x


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":

    model = CustomModel()

    dummy = torch.zeros(1, 3, 224, 224)
    out   = model(dummy)

    print(f"Output shape    : {out.shape}")
    print(f"Total Parameters: {count_params(model):,}")