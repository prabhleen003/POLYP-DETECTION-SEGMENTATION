"""
SD-DeepLab: Structural DeepLab for Polyp Segmentation
Full implementation matching the trained checkpoint
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import torchvision.models as tvm


class ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1,
                 padding=1, dilation=1, bias=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
                      padding=padding * dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SobelEdge(nn.Module):
    """Fixed (non-learnable) Sobel edge detector."""
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def forward(self, x):
        gx  = F.conv2d(x, self.kx, padding=1)
        gy  = F.conv2d(x, self.ky, padding=1)
        mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
        mag = mag / (mag.amax(dim=(2, 3), keepdim=True) + 1e-6)
        return mag


class SCCR(nn.Module):
    """Structural-Conditioned Channel Routing."""
    def __init__(self, feat_ch: int, struct_ch: int = 4, bottleneck: int = 16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.router = nn.Sequential(
            nn.Linear(struct_ch, bottleneck, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, feat_ch, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, feat: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        s_vec  = self.gap(S).squeeze(-1).squeeze(-1)
        gates  = self.router(s_vec)
        gates  = gates.unsqueeze(-1).unsqueeze(-1)
        return feat * gates


class ASA(nn.Module):
    """Anisotropic Strip Attention."""
    def __init__(self, feat_ch: int, strip_k: int = 7):
        super().__init__()
        pad = strip_k // 2
        self.h_strip = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(strip_k, 1),
                      padding=(pad, 0), bias=False),
            nn.BatchNorm2d(1),
        )
        self.w_strip = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(1, strip_k),
                      padding=(0, pad), bias=False),
            nn.BatchNorm2d(1),
        )
        self.sigmoid = nn.Sigmoid()
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        avg  = feat.mean(dim=1, keepdim=True)
        a_h  = self.sigmoid(self.h_strip(avg))
        a_w  = self.sigmoid(self.w_strip(avg))
        mask = a_h * a_w
        return feat + torch.tanh(self.gate) * feat * mask


class SDAA(nn.Module):
    """Structural Dual-Axis Attention."""
    def __init__(self, feat_ch: int, struct_ch: int = 4,
                 bottleneck: int = 16, strip_k: int = 7):
        super().__init__()
        self.sccr = SCCR(feat_ch, struct_ch, bottleneck)
        self.asa  = ASA(feat_ch, strip_k)

    def forward(self, feat: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        feat = self.sccr(feat, S)
        feat = self.asa(feat)
        return feat


class StructuralInitHead(nn.Module):
    """Produces initial structural state S5 = (M, B, D, U) @ H/16."""
    def __init__(self, in_ch=256):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBnRelu(in_ch, 128, kernel=3, padding=1),
            ConvBnRelu(128,    64, kernel=3, padding=1),
        )
        self.head_M  = nn.Conv2d(64, 1, kernel_size=1)
        self.head_U  = nn.Sequential(
            ConvBnRelu(64, 32, kernel=1, padding=0),
            nn.Conv2d(32, 1, kernel_size=1),
        )
        self.sobel   = SobelEdge()
        self.d_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        feat  = self.encoder(x)
        M     = self.head_M(feat)
        M_prob = torch.sigmoid(M)
        B     = self.sobel(M_prob)
        D     = torch.tanh(self.d_scale * M)
        U_raw = self.head_U(feat)
        U     = torch.clamp(torch.sigmoid(U_raw), 0.05, 0.95)
        return torch.cat([M, B, D, U], dim=1)


class BoundaryRefinement(nn.Module):
    """Edge-gated boundary delta predictor inside each STB."""
    def __init__(self, feat_ch):
        super().__init__()
        self.sobel         = SobelEdge()
        self.edge_sharpen  = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1, bias=False),
            nn.Sigmoid(),
        )
        self.feat_boundary = nn.Sequential(
            ConvBnRelu(feat_ch, 32, kernel=3, padding=1),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.delta_head = nn.Sequential(
            ConvBnRelu(3, 16, kernel=1, padding=0),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Tanh(),
        )

    def forward(self, B_curr, M_curr, feat):
        sobel_e  = self.sobel(torch.sigmoid(M_curr))
        sharp_e  = self.edge_sharpen(sobel_e)
        feat_cue = self.feat_boundary(feat)
        gated    = feat_cue * sharp_e
        inp      = torch.cat([B_curr, sharp_e, gated], dim=1)
        delta_B  = self.delta_head(inp)
        return delta_B, sharp_e


class StructuralTransitionBlock(nn.Module):
    """Multi-scale structural evolution step."""
    def __init__(self, feat_ch, mid_ch=128):
        super().__init__()
        self.feat_proj   = ConvBnRelu(feat_ch, mid_ch, kernel=1, padding=0)
        self.state_embed = ConvBnRelu(4,       mid_ch, kernel=1, padding=0)
        self.fusion      = nn.Sequential(
            ConvBnRelu(mid_ch * 2, mid_ch,      kernel=3, padding=1),
            ConvBnRelu(mid_ch,     mid_ch // 2, kernel=3, padding=1),
        )
        self.sdaa        = SDAA(feat_ch=mid_ch // 2, struct_ch=4,
                                bottleneck=16, strip_k=7)

        half             = mid_ch // 2
        self.delta_M     = nn.Sequential(
            ConvBnRelu(half, 32, kernel=1, padding=0),
            nn.Conv2d(32, 1, 1),
        )
        self.delta_B     = BoundaryRefinement(feat_ch=mid_ch)
        self.delta_D     = nn.Sequential(
            ConvBnRelu(half, 32, kernel=1, padding=0),
            nn.Conv2d(32, 1, 1),
            nn.Tanh(),
        )
        self.delta_U     = nn.Sequential(
            ConvBnRelu(half, 16, kernel=1, padding=0),
            nn.Conv2d(16, 1, 1),
            nn.Tanh(),
        )
        self.log_step_M  = nn.Parameter(torch.zeros(1))
        self.log_step_D  = nn.Parameter(torch.zeros(1))
        self.log_step_U  = nn.Parameter(torch.full((1,), -2.3))

    def forward(self, S_i, F_i):
        M_curr, B_curr = S_i[:, 0:1], S_i[:, 1:2]
        D_curr, U_curr = S_i[:, 2:3], S_i[:, 3:4]

        f_proj  = self.feat_proj(F_i)
        s_embed = self.state_embed(S_i)
        fused   = self.fusion(torch.cat([f_proj, s_embed], dim=1))

        fused   = self.sdaa(fused, S_i)

        step_M  = torch.exp(self.log_step_M)
        step_D  = torch.exp(self.log_step_D)
        step_U  = torch.exp(self.log_step_U)

        dM      = self.delta_M(fused) * step_M
        dD      = self.delta_D(fused) * step_D
        dU      = self.delta_U(fused) * step_U
        dB, B_aux = self.delta_B(B_curr, M_curr, f_proj)

        M_new  = M_curr + dM
        B_new  = torch.clamp(B_curr + dB, 0.0, 1.0)
        D_new  = torch.clamp(D_curr + dD, -1.0, 1.0)
        U_new  = torch.clamp(U_curr + dU, 0.05, 0.95)

        return torch.cat([M_new, B_new, D_new, U_new], dim=1), B_aux


class StructuralEnergyLayer(nn.Module):
    """One gradient step of: E(S) = λ_c ||ΔM||² + λ_s ||∇D||²"""
    def __init__(self):
        super().__init__()
        self.log_eta      = nn.Parameter(torch.tensor(-1.0))
        self.log_lambda_c = nn.Parameter(torch.tensor(-1.0))
        self.log_lambda_s = nn.Parameter(torch.tensor(-1.0))

        lap   = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                              dtype=torch.float32).view(1, 1, 3, 3)
        kx    = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                              dtype=torch.float32).view(1, 1, 3, 3)
        ky    = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                              dtype=torch.float32).view(1, 1, 3, 3)
        gauss = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]],
                              dtype=torch.float32)
        gauss = (gauss / gauss.sum()).view(1, 1, 3, 3)
        self.register_buffer("lap_kernel", lap)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)
        self.register_buffer("gauss", gauss)

    def forward(self, S):
        M, B, D, U = S[:, 0:1], S[:, 1:2], S[:, 2:3], S[:, 3:4]
        eta   = torch.exp(self.log_eta)
        lam_c = torch.exp(self.log_lambda_c)
        lam_s = torch.exp(self.log_lambda_s)

        M_star = M - eta * lam_c * F.conv2d(M, self.lap_kernel, padding=1)
        D_star = torch.clamp(D - eta * lam_s * F.conv2d(D, self.lap_kernel, padding=1),
                             -1.0, 1.0)
        B_star = torch.clamp(F.conv2d(B, self.gauss, padding=1), 0.0, 1.0)
        U_star = torch.clamp(F.conv2d(U, self.gauss, padding=1), 0.05, 0.95)

        return torch.cat([M_star, B_star, D_star, U_star], dim=1)


class ASPP(nn.Module):
    def __init__(self, in_ch=2048, out_ch=256):
        super().__init__()
        self.branch1    = ConvBnRelu(in_ch, out_ch, kernel=1, padding=0)
        self.branch2    = ConvBnRelu(in_ch, out_ch, kernel=3, dilation=6,  padding=1)
        self.branch3    = ConvBnRelu(in_ch, out_ch, kernel=3, dilation=12, padding=1)
        self.branch4    = ConvBnRelu(in_ch, out_ch, kernel=3, dilation=18, padding=1)
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            ConvBnRelu(out_ch * 5, out_ch, kernel=1, padding=0),
            nn.Dropout2d(0.1),
        )

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        b1   = self.branch1(x)
        b2   = self.branch2(x)
        b3   = self.branch3(x)
        b4   = self.branch4(x)
        gp   = F.interpolate(self.global_pool(x), size=(h, w),
                             mode="bilinear", align_corners=False)
        return self.project(torch.cat([b1, b2, b3, b4, gp], dim=1))


class ResNet50Backbone(nn.Module):
    """
    ResNet-50, OS=16:
      layer1  → H/4,  256ch   (F2)
      layer2  → H/8,  512ch   (F3)
      layer3  → H/16, 1024ch  (F4)
      layer4  → H/16, 2048ch  (F5)   — stride killed, dilation=2
    """
    def __init__(self, pretrained=True):
        super().__init__()
        weights = tvm.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        base    = tvm.resnet50(weights=weights)

        self.layer0 = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self._make_dilated(self.layer4, dilation=2)

    @staticmethod
    def _make_dilated(layer, dilation):
        b0 = layer[0]
        b0.conv2.stride   = (1, 1)
        b0.conv2.dilation = (dilation, dilation)
        b0.conv2.padding  = (dilation, dilation)
        if b0.downsample is not None:
            b0.downsample[0].stride = (1, 1)
        for block in layer[1:]:
            block.conv2.dilation = (dilation, dilation)
            block.conv2.padding  = (dilation, dilation)

    def forward(self, x):
        x  = self.layer0(x)
        F2 = self.layer1(x)
        F3 = self.layer2(F2)
        F4 = self.layer3(F3)
        F5 = self.layer4(F4)
        return F2, F3, F4, F5


class StructuralProjectionHead(nn.Module):
    """S2 (4ch) → mask logit @ 512×512 using all structural channels."""
    def __init__(self, out_size=512):
        super().__init__()
        self.out_size = out_size
        self.head = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, S):
        logit = self.head(S)
        return F.interpolate(logit, size=(self.out_size, self.out_size),
                             mode="bilinear", align_corners=False)


class SDDeepLab(nn.Module):
    """Full SD-DeepLab pipeline."""
    def __init__(self, pretrained=True, out_size=512):
        super().__init__()
        self.backbone    = ResNet50Backbone(pretrained=pretrained)
        self.aspp        = ASPP(in_ch=2048, out_ch=256)
        self.struct_init = StructuralInitHead(in_ch=256)
        self.stb4        = StructuralTransitionBlock(feat_ch=1024, mid_ch=128)
        self.stb3        = StructuralTransitionBlock(feat_ch=512,  mid_ch=128)
        self.stb2        = StructuralTransitionBlock(feat_ch=256,  mid_ch=128)
        self.energy      = StructuralEnergyLayer()
        self.proj        = StructuralProjectionHead(out_size=out_size)

    def forward(self, x, training=False):
        F2, F3, F4, F5  = self.backbone(x)
        F5_aspp         = self.aspp(F5)

        S5              = self.struct_init(F5_aspp)

        S4_up           = F.interpolate(S5, size=(F4.shape[2], F4.shape[3]),
                                        mode="bilinear", align_corners=False)
        S4, B_aux4      = self.stb4(S4_up, F4)

        S3_up           = F.interpolate(S4, size=(F3.shape[2], F3.shape[3]),
                                        mode="bilinear", align_corners=False)
        S3, B_aux3      = self.stb3(S3_up, F3)

        S2_up           = F.interpolate(S3, size=(F2.shape[2], F2.shape[3]),
                                        mode="bilinear", align_corners=False)
        S2, B_aux2      = self.stb2(S2_up, F2)

        S2_star         = self.energy(S2)
        mask_logit      = self.proj(S2_star)

        if training:
            return dict(mask=mask_logit, S2=S2_star,
                        B_aux2=B_aux2, B_aux3=B_aux3, B_aux4=B_aux4)
        return dict(mask=torch.sigmoid(mask_logit))
