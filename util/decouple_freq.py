import torch
import torch.nn.functional as F
from torch.fft import fft2, ifft2
import torch.fft 

def psf2otf(psf, shape):
    rotated_psf = torch.flip(psf, dims=(-2, -1))
    h, w = psf.shape[-2], psf.shape[-1]
    target_h, target_w = shape
    pad_h = target_h - h
    pad_w = target_w - w
    padded_psf = F.pad(rotated_psf, (0, pad_w, 0, pad_h), mode='constant', value=0)
    return torch.fft.fft2(padded_psf)

def ssf_filter_batch(I, alpha=0.8, beta=0.05, kappa=1.5, tau=1.0, iter_max=500, lambda_max=1e2):
    device = I.device
    B, C, H, W = I.shape
    
    # ================== 滤波器定义 ==================
    # 一阶差分滤波器
    Dx = torch.tensor([[1.0, -1.0]], device=device) / 2.0
    Dy = torch.tensor([[1.0], [-1.0]], device=device) / 2.0
    
    # 二阶差分滤波器
    fxx = torch.tensor([[1.0, -2.0, 1.0]], device=device) / 4.0
    fyy = torch.tensor([[1.0], [-2.0], [1.0]], device=device) / 4.0
    fuu = torch.tensor([[1.0, 0.0, 0.0],
                       [0.0, -2.0, 0.0],
                       [0.0, 0.0, 1.0]], device=device) / 4.0
    fvv = torch.tensor([[0.0, 0.0, 1.0],
                       [0.0, -2.0, 0.0],
                       [1.0, 0.0, 0.0]], device=device) / 4.0

    # =============== 预处理输入图像的梯度 ===============
    # 计算输入图像的一阶梯度
    Dx_kernel = Dx.view(1, 1, 1, 2).expand(C, 1, 1, 2).contiguous()
    Dy_kernel = Dy.view(1, 1, 2, 1).expand(C, 1, 2, 1).contiguous()
    
    # 计算Dx_I
    padded_I = F.pad(I, (0, 1, 0, 0), mode='circular')
    Dx_I = F.conv2d(padded_I, Dx_kernel, groups=C, padding=0).view(B, C, H, W)
    
    # 计算Dy_I
    padded_I = F.pad(I, (0, 0, 0, 1), mode='circular')
    Dy_I = F.conv2d(padded_I, Dy_kernel, groups=C, padding=0).view(B, C, H, W)

    # =============== 计算频域分母项 ===============
    otfDx = psf2otf(Dx, (H, W))
    otfDy = psf2otf(Dy, (H, W))
    otfFxx = psf2otf(fxx, (H, W))
    otfFyy = psf2otf(fyy, (H, W))
    otfFuu = psf2otf(fuu, (H, W))
    otfFvv = psf2otf(fvv, (H, W))
    
    Denormin1 = torch.abs(otfDx)**2 + torch.abs(otfDy)**2
    Denormin2 = (torch.abs(otfFxx)**2 + torch.abs(otfFyy)**2 
                + torch.abs(otfFuu)**2 + torch.abs(otfFvv)**2)
    
    Denormin1 = Denormin1.view(1, 1, H, W).expand(B, C, H, W).to(torch.complex64)
    Denormin2 = Denormin2.view(1, 1, H, W).expand(B, C, H, W).to(torch.complex64)

    # =============== 初始化迭代变量 ===============
    S = I.clone()
    Normin0 = torch.fft.fft2(S)
    lambda_val = 10 * beta  # 初始lambda值
    current_alpha = alpha

    # =============== 主迭代循环 ===============
    for iter in range(1, iter_max + 1):
        if lambda_val > lambda_max:
            break

        Denormin = 1.0 + current_alpha * Denormin1 + lambda_val * Denormin2

        # ----------------- 一阶梯度计算 -----------------
        # 计算当前S的一阶梯度
        padded_S = F.pad(S, (0, 1, 0, 0), mode='circular')
        gx = F.conv2d(padded_S, Dx_kernel, groups=C, padding=0).view(B, C, H, W)
        padded_S = F.pad(S, (0, 0, 0, 1), mode='circular')
        gy = F.conv2d(padded_S, Dy_kernel, groups=C, padding=0).view(B, C, H, W)
        
        # 计算梯度差异
        gx_diff = gx - Dx_I
        gy_diff = gy - Dy_I

        # ----------------- 二阶梯度计算 -----------------
        fxx_kernel = fxx.view(1, 1, 1, 3).expand(C, 1, 1, 3).contiguous()
        padded_S = F.pad(S, (1, 1, 0, 0), mode='circular')
        gxx = F.conv2d(padded_S, fxx_kernel, groups=C, padding=0).view(B, C, H, W)
        
        fyy_kernel = fyy.view(1, 1, 3, 1).expand(C, 1, 3, 1).contiguous()
        padded_S = F.pad(S, (0, 0, 1, 1), mode='circular')
        gyy = F.conv2d(padded_S, fyy_kernel, groups=C, padding=0).view(B, C, H, W)
        
        fuu_kernel = fuu.view(1, 1, 3, 3).expand(C, 1, 3, 3).contiguous()
        padded_S = F.pad(S, (1, 1, 1, 1), mode='circular')
        guu = F.conv2d(padded_S, fuu_kernel, groups=C, padding=0).view(B, C, H, W)
        
        fvv_kernel = fvv.view(1, 1, 3, 3).expand(C, 1, 3, 3).contiguous()
        padded_S = F.pad(S, (1, 1, 1, 1), mode='circular')
        gvv = F.conv2d(padded_S, fvv_kernel, groups=C, padding=0).view(B, C, H, W)

        # ----------------- 联合阈值处理 -----------------
        sum_sq = gxx**2 + gyy**2 + guu**2 + gvv**2
        if C > 1:
            sum_sq = sum_sq.sum(dim=1, keepdim=True)
            
        t = sum_sq < (beta / lambda_val)
        t = t.expand(B, C, H, W)
        
        gxx = gxx.masked_fill(t, 0.0)
        gyy = gyy.masked_fill(t, 0.0)
        guu = guu.masked_fill(t, 0.0)
        gvv = gvv.masked_fill(t, 0.0)

        # ----------------- 反向投影项计算 -----------------
        # 一阶差异项的反向投影
        reversed_Dx = torch.flip(Dx_kernel, dims=[-1])
        padded_gx_diff = F.pad(gx_diff, (0, 1, 0, 0), mode='circular')
        Normin_x = F.conv2d(padded_gx_diff, reversed_Dx, groups=C, padding=0).view(B, C, H, W)
        Normin_x = torch.roll(Normin_x, shifts=1, dims=-1)
        
        reversed_Dy = torch.flip(Dy_kernel, dims=[-2])
        padded_gy_diff = F.pad(gy_diff, (0, 0, 0, 1), mode='circular')
        Normin_y = F.conv2d(padded_gy_diff, reversed_Dy, groups=C, padding=0).view(B, C, H, W)
        Normin_y = torch.roll(Normin_y, shifts=1, dims=-2)
        
        Normin1 = Normin_x + Normin_y

        # 二阶项的反向投影
        reversed_fxx = torch.flip(fxx_kernel, dims=[-1])
        padded_gxx = F.pad(gxx, (1, 1, 0, 0), mode='circular')
        Normin_xx = F.conv2d(padded_gxx, reversed_fxx, groups=C, padding=0).view(B, C, H, W)
        
        reversed_fyy = torch.flip(fyy_kernel, dims=[-2])
        padded_gyy = F.pad(gyy, (0, 0, 1, 1), mode='circular')
        Normin_yy = F.conv2d(padded_gyy, reversed_fyy, groups=C, padding=0).view(B, C, H, W)
        
        reversed_fuu = torch.flip(fuu_kernel, dims=(-2, -1))
        padded_guu = F.pad(guu, (1, 1, 1, 1), mode='circular')
        Normin_uu = F.conv2d(padded_guu, reversed_fuu, groups=C, padding=0).view(B, C, H, W)
        
        reversed_fvv = torch.flip(fvv_kernel, dims=(-2, -1))
        padded_gvv = F.pad(gvv, (1, 1, 1, 1), mode='circular')
        Normin_vv = F.conv2d(padded_gvv, reversed_fvv, groups=C, padding=0).view(B, C, H, W)
        
        Normin2 = Normin_xx + Normin_yy + Normin_uu + Normin_vv

        # ----------------- 频域更新 -----------------
        FS = (Normin0 + current_alpha * torch.fft.fft2(Normin1) + lambda_val * torch.fft.fft2(Normin2)) / Denormin
        S = torch.real(torch.fft.ifft2(FS))
        
        # ----------------- 参数更新 -----------------
        # current_alpha *= tau  # 保持alpha不变
        lambda_val *= kappa

        if iter % 50 == 0:
            print(f'Iteration {iter}, lambda: {lambda_val:.1f}')

    return S.clamp(0, 1)