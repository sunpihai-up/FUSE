import torch
import torch.nn.functional as F
from torch.fft import fft2, ifft2
import torch.nn as nn
from torch import Tensor

from PIL import Image
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

import os
import math
from typing import Tuple, List

# from util.vis import vis_voxelgrid
from util.vis import vis_depth_map

from .utils import token2feature, feature2token


class DecoupleImage(nn.Module):
    def __init__(
        self,
        alpha=0.8,
        beta=0.05,
        kappa=1.5,
        tau=1.0,
        iter_max=500,
        lambda_max=1e2,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.tau = tau
        self.iter_max = iter_max
        self.lambda_max = lambda_max

    def psf2otf(self, psf, shape):
        rotated_psf = torch.flip(psf, dims=(-2, -1))
        h, w = psf.shape[-2], psf.shape[-1]
        target_h, target_w = shape
        pad_h = target_h - h
        pad_w = target_w - w
        padded_psf = F.pad(rotated_psf, (0, pad_w, 0, pad_h), mode="constant", value=0)
        return torch.fft.fft2(padded_psf)

    def ssf_filter_batch(self, I):
        device = I.device
        B, C, H, W = I.shape

        # ================== 滤波器定义 ==================
        # 一阶差分滤波器
        Dx = torch.tensor([[1.0, -1.0]], device=device) / 2.0
        Dy = torch.tensor([[1.0], [-1.0]], device=device) / 2.0

        # 二阶差分滤波器
        fxx = torch.tensor([[1.0, -2.0, 1.0]], device=device) / 4.0
        fyy = torch.tensor([[1.0], [-2.0], [1.0]], device=device) / 4.0
        fuu = (
            torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, -2.0, 0.0], [0.0, 0.0, 1.0]], device=device
            )
            / 4.0
        )
        fvv = (
            torch.tensor(
                [[0.0, 0.0, 1.0], [0.0, -2.0, 0.0], [1.0, 0.0, 0.0]], device=device
            )
            / 4.0
        )

        # =============== 预处理输入图像的梯度 ===============
        # 计算输入图像的一阶梯度
        Dx_kernel = Dx.view(1, 1, 1, 2).expand(C, 1, 1, 2).contiguous()
        Dy_kernel = Dy.view(1, 1, 2, 1).expand(C, 1, 2, 1).contiguous()

        # 计算Dx_I
        padded_I = F.pad(I, (0, 1, 0, 0), mode="circular")
        Dx_I = F.conv2d(padded_I, Dx_kernel, groups=C, padding=0).view(B, C, H, W)

        # 计算Dy_I
        padded_I = F.pad(I, (0, 0, 0, 1), mode="circular")
        Dy_I = F.conv2d(padded_I, Dy_kernel, groups=C, padding=0).view(B, C, H, W)

        # =============== 计算频域分母项 ===============
        otfDx = self.psf2otf(Dx, (H, W))
        otfDy = self.psf2otf(Dy, (H, W))
        otfFxx = self.psf2otf(fxx, (H, W))
        otfFyy = self.psf2otf(fyy, (H, W))
        otfFuu = self.psf2otf(fuu, (H, W))
        otfFvv = self.psf2otf(fvv, (H, W))

        Denormin1 = torch.abs(otfDx) ** 2 + torch.abs(otfDy) ** 2
        Denormin2 = (
            torch.abs(otfFxx) ** 2
            + torch.abs(otfFyy) ** 2
            + torch.abs(otfFuu) ** 2
            + torch.abs(otfFvv) ** 2
        )

        Denormin1 = Denormin1.view(1, 1, H, W).expand(B, C, H, W).to(torch.complex64)
        Denormin2 = Denormin2.view(1, 1, H, W).expand(B, C, H, W).to(torch.complex64)

        # =============== 初始化迭代变量 ===============
        S = I.clone()
        Normin0 = torch.fft.fft2(S)
        lambda_val = 10 * self.beta  # 初始lambda值
        current_alpha = self.alpha

        # =============== 主迭代循环 ===============
        for iter in range(1, self.iter_max + 1):
            if lambda_val > self.lambda_max:
                break

            Denormin = 1.0 + current_alpha * Denormin1 + lambda_val * Denormin2

            # ----------------- 一阶梯度计算 -----------------
            # 计算当前S的一阶梯度
            padded_S = F.pad(S, (0, 1, 0, 0), mode="circular")
            gx = F.conv2d(padded_S, Dx_kernel, groups=C, padding=0).view(B, C, H, W)
            padded_S = F.pad(S, (0, 0, 0, 1), mode="circular")
            gy = F.conv2d(padded_S, Dy_kernel, groups=C, padding=0).view(B, C, H, W)

            # 计算梯度差异
            gx_diff = gx - Dx_I
            gy_diff = gy - Dy_I

            # ----------------- 二阶梯度计算 -----------------
            fxx_kernel = fxx.view(1, 1, 1, 3).expand(C, 1, 1, 3).contiguous()
            padded_S = F.pad(S, (1, 1, 0, 0), mode="circular")
            gxx = F.conv2d(padded_S, fxx_kernel, groups=C, padding=0).view(B, C, H, W)

            fyy_kernel = fyy.view(1, 1, 3, 1).expand(C, 1, 3, 1).contiguous()
            padded_S = F.pad(S, (0, 0, 1, 1), mode="circular")
            gyy = F.conv2d(padded_S, fyy_kernel, groups=C, padding=0).view(B, C, H, W)

            fuu_kernel = fuu.view(1, 1, 3, 3).expand(C, 1, 3, 3).contiguous()
            padded_S = F.pad(S, (1, 1, 1, 1), mode="circular")
            guu = F.conv2d(padded_S, fuu_kernel, groups=C, padding=0).view(B, C, H, W)

            fvv_kernel = fvv.view(1, 1, 3, 3).expand(C, 1, 3, 3).contiguous()
            padded_S = F.pad(S, (1, 1, 1, 1), mode="circular")
            gvv = F.conv2d(padded_S, fvv_kernel, groups=C, padding=0).view(B, C, H, W)

            # ----------------- 联合阈值处理 -----------------
            sum_sq = gxx**2 + gyy**2 + guu**2 + gvv**2
            if C > 1:
                sum_sq = sum_sq.sum(dim=1, keepdim=True)

            t = sum_sq < (self.beta / lambda_val)
            t = t.expand(B, C, H, W)

            gxx = gxx.masked_fill(t, 0.0)
            gyy = gyy.masked_fill(t, 0.0)
            guu = guu.masked_fill(t, 0.0)
            gvv = gvv.masked_fill(t, 0.0)

            # ----------------- 反向投影项计算 -----------------
            # 一阶差异项的反向投影
            reversed_Dx = torch.flip(Dx_kernel, dims=[-1])
            padded_gx_diff = F.pad(gx_diff, (0, 1, 0, 0), mode="circular")
            Normin_x = F.conv2d(padded_gx_diff, reversed_Dx, groups=C, padding=0).view(
                B, C, H, W
            )
            Normin_x = torch.roll(Normin_x, shifts=1, dims=-1)

            reversed_Dy = torch.flip(Dy_kernel, dims=[-2])
            padded_gy_diff = F.pad(gy_diff, (0, 0, 0, 1), mode="circular")
            Normin_y = F.conv2d(padded_gy_diff, reversed_Dy, groups=C, padding=0).view(
                B, C, H, W
            )
            Normin_y = torch.roll(Normin_y, shifts=1, dims=-2)

            Normin1 = Normin_x + Normin_y

            # 二阶项的反向投影
            reversed_fxx = torch.flip(fxx_kernel, dims=[-1])
            padded_gxx = F.pad(gxx, (1, 1, 0, 0), mode="circular")
            Normin_xx = F.conv2d(padded_gxx, reversed_fxx, groups=C, padding=0).view(
                B, C, H, W
            )

            reversed_fyy = torch.flip(fyy_kernel, dims=[-2])
            padded_gyy = F.pad(gyy, (0, 0, 1, 1), mode="circular")
            Normin_yy = F.conv2d(padded_gyy, reversed_fyy, groups=C, padding=0).view(
                B, C, H, W
            )

            reversed_fuu = torch.flip(fuu_kernel, dims=(-2, -1))
            padded_guu = F.pad(guu, (1, 1, 1, 1), mode="circular")
            Normin_uu = F.conv2d(padded_guu, reversed_fuu, groups=C, padding=0).view(
                B, C, H, W
            )

            reversed_fvv = torch.flip(fvv_kernel, dims=(-2, -1))
            padded_gvv = F.pad(gvv, (1, 1, 1, 1), mode="circular")
            Normin_vv = F.conv2d(padded_gvv, reversed_fvv, groups=C, padding=0).view(
                B, C, H, W
            )

            Normin2 = Normin_xx + Normin_yy + Normin_uu + Normin_vv

            # ----------------- 频域更新 -----------------
            FS = (
                Normin0
                + current_alpha * torch.fft.fft2(Normin1)
                + lambda_val * torch.fft.fft2(Normin2)
            ) / Denormin
            S = torch.real(torch.fft.ifft2(FS))

            # ----------------- 参数更新 -----------------
            # current_alpha *= tau  # 保持alpha不变
            lambda_val *= self.kappa

            if iter % 50 == 0:
                print(f"Iteration {iter}, lambda: {lambda_val:.1f}")

        return S.clamp(0, 1)

    def forward(self, imgs):
        low_freq = self.ssf_filter_batch(imgs)
        high_freq = imgs - low_freq
        return low_freq, high_freq


class DirectionVarEntropy(nn.Module):
    def __init__(
        self,
        patch_size: Tuple[int, int] = (14, 14),
        dct_block_size: Tuple[int, int] = (3, 3),
        bins: int = 256,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.dct_block_size = dct_block_size
        self.bins = bins

        # Pre-generate DCT basis matrix (for acceleration)
        self.register_buffer("dct_matrix", self.generate_dct_matrix(dct_block_size[0]))

    def generate_dct_matrix(self, size: int) -> Tensor:
        """Generate an orthogonal DCT-II basis matrix."""
        matrix = torch.zeros(size, size)
        for i in range(size):
            for j in range(size):
                if i == 0:
                    matrix[i, j] = math.sqrt(1 / size)
                else:
                    matrix[i, j] = math.sqrt(2 / size) * math.cos(
                        math.pi * (2 * j + 1) * i / (2 * size)
                    )
        return matrix

    def dct2d(self, blocks: Tensor) -> Tensor:
        """Batch 2D DCT transformation."""
        return torch.matmul(self.dct_matrix, torch.matmul(blocks, self.dct_matrix.T))

    def extract_patches(self, x: Tensor) -> Tensor:
        """Extract patches from image (without padding)."""
        B, C, H, W = x.shape
        ph, pw = self.patch_size
        dh, dw = self.dct_block_size

        # Main patch extraction (14x14 or given patch_size)
        x = x.unfold(2, ph, ph).unfold(3, pw, pw)  # (B, C, H/ph, W/pw, ph, pw)
        x = x.permute(0, 2, 3, 1, 4, 5)  # (B, H_p, W_p, C, ph, pw)

        return x.contiguous()

    def compute_direction_variance(self, dct_blocks: Tensor) -> Tensor:
        """
        Compute the variance of standard deviations across four directions (rows, columns, main diagonal, and anti-diagonal).
        The result is normalized and the variance is calculated.
        """

        """ Calculate standard deviations for each direction """
        # Row-wise standard deviation
        S1 = torch.std(dct_blocks, dim=-1, unbiased=False).mean(dim=-1)

        # Column-wise standard deviation
        S2 = torch.std(dct_blocks, dim=-2, unbiased=False).mean(dim=-1)

        # Main diagonal standard deviation
        diag_main = torch.diagonal(dct_blocks, dim1=-2, dim2=-1)
        S3 = torch.std(diag_main, dim=-1, unbiased=False)

        # Anti-diagonal standard deviation (flip the tensor horizontally and then take the main diagonal)
        flipped = torch.flip(dct_blocks, dims=[-1])
        diag_anti = torch.diagonal(flipped, dim1=-2, dim2=-1)
        S4 = torch.std(diag_anti, dim=-1, unbiased=False)

        """ Normalize the standard deviations to avoid division by zero """
        S = (S1 + S2 + S3 + S4) / 4 + 1e-8
        S1_norm = S1 / S
        S2_norm = S2 / S
        S3_norm = S3 / S
        S4_norm = S4 / S

        """ Calculate the variance of the normalized standard deviations """
        return torch.var(
            torch.stack([S1_norm, S2_norm, S3_norm, S4_norm], dim=-1),
            dim=-1,
            unbiased=True,
        )

    def compute_entropy(self, patches):
        """
        Compute the entropy for each channel of the patches, preserving the channel dimension.
        """
        # Convert pixel values to integers in the range [0, 255]
        patches_int = (patches * 255).round().long().clamp(0, 255)
        patches_flat = patches_int.flatten(start_dim=4)  # [B, H_p, W_p, C, ph*pw]

        # Get tensor dimensions and device
        B, H_p, W_p, C, N = patches_flat.shape
        device = patches_flat.device

        # Expand dimensions for broadcasting comparison
        patches_expanded = patches_flat.unsqueeze(-1)  # [B, H_p, W_p, C, N, 1]
        # Shape: [1, 1, 1, 1, 1, 256]
        values = torch.arange(self.bins, device=device).view(1, 1, 1, 1, 1, self.bins)

        # Count the occurrences of each pixel value
        counts = (patches_expanded == values).sum(dim=4)  # [B, H_p, W_p, C, 256]

        # Compute probabilities for each pixel value
        prob = counts.float() / N + 1e-10
        entropy = -(prob * torch.log2(prob)).sum(dim=-1)  # [B, H_p, W_p, C]
        return entropy

    def forward(self, x: Tensor) -> Tensor:
        """Calculate information richness for each patch."""
        B, C, H, W = x.shape

        # Extract patches
        patches = self.extract_patches(x)  # (B, H_p, W_p, C, ph, pw)

        # Compute entropy
        entropy = self.compute_entropy(patches)  # (B, H_p, W_p, C)

        # Sub-patch extraction (7x7 or given dct_block_size)
        dh, dw = self.dct_block_size
        patches = patches.unfold(4, dh, 1).unfold(5, dw, 1)  # 步长1
        # patches = patches.unfold(4, dh, dh).unfold(5, dw, dw)  # (B, H_p, W_p, C, ph/dh, pw/dw, dh, dw)
        patches = patches.contiguous()
        B_p, H_p, W_p, C_p, n_dh, n_dw, dh, dw = patches.shape

        # Flatten for DCT processing
        dct_input = patches.view(-1, dh, dw)  # (dct_block_number, dct_h, dct_w)

        # Apply DCT transformation
        dct_blocks = self.dct2d(dct_input)  # (B*H_p*W_p*C*n_dh*n_dw, dh, dw)

        # Compute multidirectional variance feature
        psi_m = self.compute_direction_variance(dct_blocks)
        psi_m = psi_m.view(B, H_p, W_p, C, n_dh, n_dw).mean(
            dim=[-1, -2]
        )  # (B, H_p, W_p, C)

        # Combine features (variance * entropy)
        richness = psi_m * entropy  # (B, H_p, W_p, C)

        # print(psi_m.min(), psi_m.max(), psi_m.mean())
        # print(entropy.min(), entropy.max(), entropy.mean())

        # Return average information richness per patch
        return richness.mean(dim=-1)  # (B, H_p, W_p)


class PatchGra(nn.Module):
    """
    Compute patch-wise gradient magnitude metric for information richness measurement.
    Processes input in patch format and outputs scalar values per patch.
    """

    def __init__(
        self,
        patch_size: Tuple[int, int] = (14, 14),
    ):
        super().__init__()
        self.p = 1  # Original MATLAB parameter
        self.patch_size = patch_size

    def extract_patches(self, x: Tensor) -> Tensor:
        """Extract patches from image (without padding)."""
        B, C, H, W = x.shape
        ph, pw = self.patch_size

        # Main patch extraction (14x14 or given patch_size)
        x = x.unfold(2, ph, ph).unfold(3, pw, pw)  # (B, C, H/ph, W/pw, ph, pw)
        x = x.permute(0, 2, 3, 1, 4, 5)  # (B, H_p, W_p, C, ph, pw)

        return x.contiguous()

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [B, H_p, W_p, C, h, w]
               B: batch size
               H_p, W_p: number of patches in height/width dimensions
               C: channels
               h, w: spatial dimensions of each patch

        Returns:
            Tensor of shape [B, H_p, W_p] containing scalar values
            representing information richness per patch
        """
        x = self.extract_patches(x)  # (B, H_p, W_p, C, ph, pw)

        # Save original shape and reshape for patch processing
        B, H_p, W_p, C, h, w = x.shape
        x = x.contiguous().view(-1, C, h, w)  # [B*H_p*W_p, C, h, w]

        # Compute horizontal differences
        diff_h = torch.diff(x, dim=-1)  # Horizontal diff
        pad_h = (x[..., 0] - x[..., -1]).unsqueeze(-1)  # Circular padding
        u_h = torch.abs(torch.cat([diff_h, pad_h], dim=-1))

        # Compute vertical differences
        diff_v = torch.diff(x, dim=-2)  # Vertical diff
        pad_v = (x[:, :, 0, :] - x[:, :, -1, :]).unsqueeze(-2)  # Circular padding
        u_v = torch.abs(torch.cat([diff_v, pad_v], dim=-2))

        # Compute constants
        gamma = 0.5 * self.p - 1
        e = torch.tensor(torch.e, dtype=x.dtype)
        c = self.p * (e**gamma)

        # Compute magnitude components
        mu_h = c * u_h - torch.pow(2 * u_h + 0.01, self.p)
        mu_v = c * u_v - torch.pow(2 * u_v + 0.01, self.p)

        # Combine magnitudes and reduce spatial dimensions
        S = torch.sqrt(mu_h**2 + mu_v**2)
        S_channel_mean = S.mean(dim=[-2, -1])  # Average over h,w per channel

        # Reshape back to original patch structure
        S_patch = S_channel_mean.view(B, H_p, W_p, C)

        # Final reduction: average across channels
        return S_patch.mean(dim=-1)


class ModalConfidence(nn.Module):
    def __init__(
        self,
        alpha: float = 0.8,
        beta: float = 0.05,
        kappa: float = 1.5,
        tau: float = 1,
        iter_max: int = 500,
        lambda_max: float = 100,
    ):
        super().__init__()
        self.img_decouple = DecoupleImage(
            alpha=alpha,
            beta=beta,
            kappa=kappa,
            tau=tau,
            iter_max=iter_max,
            lambda_max=lambda_max,
        )
        self.low_weight = DirectionVarEntropy()
        self.high_weight = PatchGra()

    def forward(self, x):
        B, c, w, h = x.shape
        low, high = self.img_decouple(x)
        low_weight = self.low_weight(low).contiguous().view(B, -1)
        high_weight = self.high_weight(high).contiguous().view(B, -1)
        del low, high
        return low_weight, high_weight


class FFTLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, patch_grid_size):
        # The shape of x: [B, N, C]
        x_fea = token2feature(x, patch_grid_size)
        x_fft = torch.fft.fft2(x_fea)
        x_magnitude = torch.abs(x_fft)
        x_phase = torch.angle(x_fft)
        x_fft = torch.cat([x_magnitude, x_phase], dim=1)
        x_fft = feature2token(x_fft)
        return x_fft


class DualDomainFeatureDecoupler(nn.Module):
    def __init__(
        self,
        embed_dim,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Frequency-Domain Guided Branch
        self.fft_layer = FFTLayer()
        self.freq_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim // 4),
            nn.LayerNorm(embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 2),
            nn.Softmax(dim=-1),
        )

        # Spatial Refinement Branch
        self.spatial_split = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 4, 3, padding=1, groups=4),
            nn.BatchNorm2d(embed_dim // 4),
            nn.GELU(),
            nn.Conv2d(embed_dim // 4, 2, 1),
        )

    def forward(self, x, patch_grid_size):
        B, N, C = x.shape

        # Frequency Domain Guidance
        x_fft = self.fft_layer(x, patch_grid_size)  # [B, N, C*2]
        freq_gates = self.freq_gate(x_fft.mean(dim=1))  # [B, 2]

        # Spatial domain refinement
        x_fea = token2feature(x, patch_grid_size)
        spatial_mask = self.spatial_split(x_fea)  # [B, 2, H, W]

        # Dual domain fusion
        combined_mask = F.softmax(spatial_mask * freq_gates.view(B, 2, 1, 1), dim=1)

        low = x_fea * combined_mask[:, 0:1]
        high = x_fea * combined_mask[:, 1:2]

        low = feature2token(low)
        high = feature2token(high)

        return low, high


class RelativeConfidenceCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, reduction=1, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear Projection Layers
        self.W_q = nn.Linear(embed_dim, embed_dim)  # x1 -> Query
        self.W_k = nn.Linear(embed_dim, embed_dim)  # x2 -> Key
        self.W_v = nn.Linear(embed_dim, embed_dim)  # x2 -> Value

        self.temperature = nn.Parameter(torch.tensor(1.0))

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x1, x2, mask_1, mask_2):
        B, N, _ = x1.shape
        """
        x1/x2: [B, N, C]
        masks: [B, N]
        """
        q = self.W_q(x1)
        k = self.W_k(x2)
        v = self.W_v(x2)

        # [B, H, N, D]
        q = q.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Step 3: Calculate attention score (incorporating relative credibility)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)

        # [B, N, 1] - [B, 1, N] → [B, N, N]
        relative_mask = (mask_1.unsqueeze(-1) - mask_2.unsqueeze(1)) / self.temperature
        attn_scores = attn_scores + relative_mask.unsqueeze(1)  # [B, H, N, N]

        # Calculating attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Step 4: Aggregate Value
        attn_output = torch.matmul(attn_weights, v)  # [B, H, N, D]

        # Merge multiple heads and project
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(B, N, -1)
        attn_output = self.out_proj(attn_output)  # [B, N, C]
        attn_output = self.norm(0.5 * x1 + self.dropout(attn_output))
        return attn_output


class FeatureDecoupleFusion(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
    ):
        super().__init__()
        self.img_feature_decouple = DualDomainFeatureDecoupler(embed_dim=embed_dim)
        self.vox_feature_decouple = DualDomainFeatureDecoupler(embed_dim=embed_dim)
        self.low_cross_att = RelativeConfidenceCrossAttention(
            embed_dim=embed_dim, num_heads=num_heads
        )
        self.high_cross_att = RelativeConfidenceCrossAttention(
            embed_dim=embed_dim, num_heads=num_heads
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x1,
        x2,
        x1_low_cof,
        x2_low_cof,
        x1_high_cof,
        x2_high_cof,
        patch_grid_size,
    ):
        """
        Shape of x1/x2: [B, N, C]
        Shape of cofs:  [B, N]
        """

        x1_low, x1_high = self.img_feature_decouple(x1, patch_grid_size)
        x2_low, x2_high = self.vox_feature_decouple(x2, patch_grid_size)

        low_fused = self.low_cross_att(x1_low, x2_low, x1_low_cof, x2_low_cof)
        high_fused = self.high_cross_att(x2_high, x1_high, x2_high_cof, x1_high_cof)

        fused = low_fused + high_fused

        return self.layer_norm(fused)


class GaussianLaplacianPyramid(nn.Module):
    def __init__(self, nlev=3, scale=0.5, sigma=1.0, kernel_size=3):
        super().__init__()
        self.nlev = nlev
        self.scale = scale
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.gaussian_filter = self._create_gaussian_kernel(self.kernel_size, sigma)

    def _create_gaussian_kernel(self, kernel_size, sigma):
        """创建高斯卷积核"""
        ax = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        return kernel.view(1, 1, kernel_size, kernel_size) / kernel.sum()

    def _gaussian_blur(self, x):
        """应用高斯模糊（带反射填充）"""
        channels = x.size(1)
        device = x.device
        gaussian_filter = self.gaussian_filter.repeat(channels, 1, 1, 1).to(device)

        pad = (
            self.gaussian_filter.shape[-1] // 2,
            self.gaussian_filter.shape[-1] // 2,
            self.gaussian_filter.shape[-2] // 2,
            self.gaussian_filter.shape[-2] // 2,
        )
        x_pad = F.pad(x, pad, mode="reflect")
        # print(x_pad.device, gaussian_filter.device)
        return F.conv2d(x_pad, gaussian_filter, groups=channels)

    def _downsample(self, x):
        """高斯模糊+下采样"""
        x_blur = self._gaussian_blur(x)
        _, _, h, w = x_blur.shape
        new_size = (int(h * self.scale), int(w * self.scale))
        return F.interpolate(x_blur, new_size, mode="bilinear", align_corners=False)

    def _upsample(self, x, target_size):
        """上采样+高斯模糊"""
        x_up = F.interpolate(x, target_size, mode="bilinear", align_corners=False)
        return self._gaussian_blur(x_up)

    def forward(self, x):
        # 构建高斯金字塔和拉普拉斯金字塔
        G = [x]
        L = []
        current = x

        for _ in range(self.nlev - 1):
            down = self._downsample(current)
            up = self._upsample(down, current.shape[-2:])
            L.append(current - up)
            G.append(down)
            current = down

        # 低频信息：最底层高斯层上采样回原尺寸
        # low_freq = self._upsample(G[-1], x.shape[-2:])
        low_freq = torch.zeros_like(x)
        for i, g in enumerate(G):
            up_g = self._upsample(g, x.shape[-2:])
            weight = 0.5**i  # 高层级权重衰减
            low_freq += weight * up_g

        # 高频信息：所有拉普拉斯层上采样求和
        high_freq = torch.zeros_like(x)
        for laplacian in L:
            up_lap = self._upsample(laplacian, x.shape[-2:])
            high_freq += up_lap

        return low_freq, high_freq


class SpatialFrequency(nn.Module):
    """空间频率特征提取模块"""

    def __init__(self, kernel_size=3):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        # 镜像填充1像素
        x_pad = F.pad(x, (1, 1, 1, 1), mode="reflect")

        # 计算水平和垂直梯度
        kernel_h = torch.tensor([0, 1, -1], dtype=torch.float32).view(1, 1, 1, 3)
        kernel_v = torch.tensor([0, 1, -1], dtype=torch.float32).view(1, 1, 3, 1)

        grad_h = F.conv2d(x_pad, kernel_h.to(x.device), padding=0)
        grad_v = F.conv2d(x_pad, kernel_v.to(x.device), padding=0)

        # 裁剪到原始尺寸
        grad_h = grad_h[:, :, 1:-1, 1:-1]
        grad_v = grad_v[:, :, 1:-1, 1:-1]

        # 计算空间频率
        rf = self.avg_pool(grad_v.pow(2))
        cf = self.avg_pool(grad_h.pow(2))
        return torch.sqrt(rf + cf)


class LowFrequencyExtractor(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.depthwise = nn.Conv2d(
            dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim
        )
        self.pointwise = nn.Conv2d(dim, dim, kernel_size=1)

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        low_pass = self.depthwise(x)
        low_pass = self.pointwise(low_pass)
        gate = self.gate(low_pass)
        return low_pass * gate


class FrequencyDecoupler(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.low_extractor = LowFrequencyExtractor(dim)

    def forward(self, x):
        low_freq = self.low_extractor(x)
        high_freq = x - low_freq

        return low_freq, high_freq


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"

        # Linear layers for Q, K, V
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        # Output linear layer
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x1, x2):
        # Ensure x1 and x2 have the same batch size and embedding dimension
        assert x1.size(0) == x2.size(0), "Batch sizes of x1 and x2 must match"
        assert x1.size(2) == x2.size(2), "Embedding dimensions of x1 and x2 must match"

        batch_size, seq_len1, embed_dim = x1.shape
        seq_len2 = x2.size(1)

        # Project inputs to Q, K, V
        Q = self.query(x1)  # (batch_size, seq_len1, embed_dim)
        K = self.key(x2)  # (batch_size, seq_len2, embed_dim)
        V = self.value(x2)  # (batch_size, seq_len2, embed_dim)

        # Reshape Q, K, V for multi-head attention
        Q = Q.view(batch_size, seq_len1, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # (batch_size, num_heads, seq_len1, head_dim)
        K = K.view(batch_size, seq_len2, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # (batch_size, num_heads, seq_len2, head_dim)
        V = V.view(batch_size, seq_len2, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # (batch_size, num_heads, seq_len2, head_dim)

        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32)
        )  # (batch_size, num_heads, seq_len1, seq_len2)
        attn_weights = F.softmax(
            scores, dim=-1
        )  # (batch_size, num_heads, seq_len1, seq_len2)

        # Apply attention weights to V
        output = torch.matmul(
            attn_weights, V
        )  # (batch_size, num_heads, seq_len1, head_dim)
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len1, self.embed_dim)
        )  # (batch_size, seq_len1, embed_dim)

        # Final linear layer
        output = self.out(output)
        return output


# class Feature_Decouple_Fusion(nn.Module):
#     def __init__(
#         self,
#         embed_dim: int,
#         num_heads: int,
#     ):
#         super().__init__()

#         self.feature_decouple = FrequencyDecoupler(dim=embed_dim)
#         self.cross_atten = CrossAttention(embed_dim=embed_dim, num_heads=num_heads)
#         self.layer_norm_low = nn.LayerNorm(embed_dim)
#         self.layer_norm_high = nn.LayerNorm(embed_dim)
#         self.layer_norm = nn.LayerNorm(embed_dim)

#     def forward(self, img, vox, patch_grid_size):
#         img_fea = token2feature(img, patch_grid_size)
#         # vox_fea = token2feature(vox, patch_grid_size)

#         img_low_freq, img_high_freq = self.feature_decouple(img_fea)
#         img_low_freq = feature2token(img_low_freq)
#         img_high_freq = feature2token(img_high_freq)

#         img_low_freq = self.layer_norm_low(img_low_freq)
#         img_high_freq = self.layer_norm_high(img_high_freq)

#         fused_high_freq = self.cross_atten(img_high_freq, vox)
#         fused = self.layer_norm(img_low_freq + fused_high_freq)
#         return fused


# class Feature_Decouple_Fusion(nn.Module):
#     def __init__(
#         self,
#         embed_dim: int,
#     ):
#         super().__init__()
#         self.feature_decouple = GaussianLaplacianPyramid()
#         self.layer_norm = nn.LayerNorm(embed_dim)

#     def forward(
#         self,
#         x1,
#         x2,
#         x1_low_weight,
#         x2_low_weight,
#         x1_high_weight,
#         x2_high_weight,
#         patch_grid_size,
#     ):
#         x1_fea = token2feature(x1, patch_grid_size)
#         x2_fea = token2feature(x2, patch_grid_size)

#         x1_low, x1_high = self.feature_decouple(x1_fea)
#         x2_low, x2_high = self.feature_decouple(x2_fea)

#         x1_low_weight = x1_low_weight.unsqueeze(1)
#         x2_low_weight = x2_low_weight.unsqueeze(1)
#         sum_low = x1_low_weight + x2_low_weight + 1e-6
#         fused_low = (x1_low * x1_low_weight + x2_low * x2_low_weight) / sum_low

#         x1_high_weight = x1_high_weight.unsqueeze(1)
#         x2_high_weight = x2_high_weight.unsqueeze(1)
#         sum_high = x1_high_weight + x2_high_weight + 1e-6
#         fused_high = (x1_high * x1_high_weight + x2_high * x2_high_weight) / sum_high

#         fused = feature2token((fused_high + fused_low) / 2)
#         # fused = fused + x1 + x2
#         # fused = x1
#         # Layer Normalization
#         fused_tokens = self.layer_norm(fused)  # [B, N, D]
#         # fused_tokens = F.gelu(fused_tokens)

#         return fused_tokens
