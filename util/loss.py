import torch
from torch import nn
import torch.nn.functional as F
from kornia.filters.sobel import spatial_gradient, sobel


class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target, valid_mask, eps=1e-8):
        valid_mask = valid_mask.detach()
        diff_log = torch.log(target[valid_mask] + eps) - torch.log(
            pred[valid_mask] + eps
        )
        loss = torch.sqrt(
            torch.pow(diff_log, 2).mean() - self.lambd * torch.pow(diff_log.mean(), 2)
        )
        return loss


class SiLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target, valid_mask):
        valid_mask = valid_mask.detach()
        pred = pred[valid_mask]
        target = target[valid_mask]
        
        diff = target - pred
        loss = torch.sqrt(
            torch.pow(diff, 2).mean() - self.lambd * torch.pow(diff.mean(), 2)
        )
        return loss


class F1_Loss(nn.Module):
    def __init__(self, log_normalized=False):
        super().__init__()
        self.log_normalized = log_normalized

    def log_normalize_fun(self, depth_map, max_val=None):        
        max_value = torch.max(depth_map) if max_val is None else max_val
        return torch.log1p(depth_map) / torch.log1p(max_value)
    
    def forward(self, pred, target, valid_mask):
        pred = pred[valid_mask]
        target = target[valid_mask]
        
        if self.log_normalized:
            pred = self.log_normalize_fun(pred)
            target = self.log_normalize_fun(target)
        diff = torch.abs(pred - target)
        return diff.mean()

class MSGIL_NORM_Loss(nn.Module):
    """
    Our proposed GT normalized Multi-scale Gradient Loss Function.
    """
    def __init__(self, scale=4, valid_threshold=-1e-8, max_threshold=1e8):
        super(MSGIL_NORM_Loss, self).__init__()
        self.scales_num = scale
        self.valid_threshold = valid_threshold
        self.max_threshold = max_threshold
        self.EPSILON = 1e-6

    def one_scale_gradient_loss(self, pred_scale, gt, mask):
        mask_float = mask.to(dtype=pred_scale.dtype, device=pred_scale.device)

        d_diff = pred_scale - gt

        v_mask = torch.mul(mask_float[:, :, :-2, :], mask_float[:, :, 2:, :])
        v_gradient = torch.abs(d_diff[:, :, :-2, :] - d_diff[:, :, 2:, :])
        v_gradient = torch.mul(v_gradient, v_mask)

        h_gradient = torch.abs(d_diff[:, :, :, :-2] - d_diff[:, :, :, 2:])
        h_mask = torch.mul(mask_float[:, :, :, :-2], mask_float[:, :, :, 2:])
        h_gradient = torch.mul(h_gradient, h_mask)

        valid_num = torch.sum(h_mask) + torch.sum(v_mask)

        gradient_loss = torch.sum(h_gradient) + torch.sum(v_gradient)
        gradient_loss = gradient_loss / (valid_num + 1e-8)

        return gradient_loss

    def transform(self, gt):
        # Get mean and standard deviation
        data_mean = []
        data_std_dev = []
        for i in range(gt.shape[0]):
            gt_i = gt[i]
            mask = gt_i > 0
            depth_valid = gt_i[mask]
            if depth_valid.shape[0] < 10:
                data_mean.append(torch.tensor(0).cuda())
                data_std_dev.append(torch.tensor(1).cuda())
                continue
            size = depth_valid.shape[0]
            depth_valid_sort, _ = torch.sort(depth_valid, 0)
            depth_valid_mask = depth_valid_sort[int(size*0.1): -int(size*0.1)]
            data_mean.append(depth_valid_mask.mean())
            data_std_dev.append(depth_valid_mask.std())
        data_mean = torch.stack(data_mean, dim=0).cuda()
        data_std_dev = torch.stack(data_std_dev, dim=0).cuda()

        return data_mean, data_std_dev

    def forward(self, pred, gt):
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
            gt = gt.unsqueeze(1)
            
        mask = gt > self.valid_threshold
        grad_term = 0.0
        gt_mean, gt_std = self.transform(gt)
        gt_trans = (gt - gt_mean[:, None, None, None]) / (gt_std[:, None, None, None] + 1e-8)
        for i in range(self.scales_num):
            step = pow(2, i)
            d_gt = gt_trans[:, :, ::step, ::step]
            d_pred = pred[:, :, ::step, ::step]
            d_mask = mask[:, :, ::step, ::step]
            grad_term += self.one_scale_gradient_loss(d_pred, d_gt, d_mask)
        return grad_term

class MultiScaleGradient(torch.nn.Module):
    def __init__(self, start_scale=1, num_scales=4):
        super(MultiScaleGradient, self).__init__()
        print("Setting up Multi Scale Gradient loss...")

        self.start_scale = start_scale
        self.num_scales = num_scales

        self.multi_scales = [
            torch.nn.AvgPool2d(
                self.start_scale * (2**scale), self.start_scale * (2**scale)
            )
            for scale in range(self.num_scales)
        ]
        print("Done")

    def forward(self, prediction, target):
        # helper to remove potential nan in labels
        def nan_helper(y):
            return torch.isnan(y), lambda z: z.nonzero()[0]

        prediction = prediction.unsqueeze(1)
        target = target.unsqueeze(1)
        
        loss_value = 0
        loss_value_2 = 0
        diff = prediction - target
        _, _, H, W = target.shape
        upsample = torch.nn.Upsample(
            size=(2 * H, 2 * W), mode="bicubic", align_corners=True
        )
        record = []

        for m in self.multi_scales:
            # input and type are of the type [B x C x H x W]
            # Use kornia spatial gradient computation
            delta_diff = spatial_gradient(m(diff))
            is_nan = torch.isnan(delta_diff)
            is_not_nan_sum = (~is_nan).sum()
            # output of kornia spatial gradient is [B x C x 2 x H x W]
            loss_value += (
                torch.abs(delta_diff[~is_nan]).sum()
                / is_not_nan_sum
                * target.shape[0]
                * 2
            )
            # * batch size * 2 (because kornia spatial product has two outputs).
            # replaces the following line to be able to deal with nan's.
            # loss_value += torch.abs(delta_diff).mean(dim=(3,4)).sum()

        return loss_value / self.num_scales


class MixedLoss(nn.Module):
    def __init__(self, siloss_lambd=0.5, grad_loss_weight=0.5, log_normalize=False):
        super().__init__()
        self.siloss_lambd = siloss_lambd
        self.grad_loss_weight = grad_loss_weight
        self.log_normalize = log_normalize
        
        # self.si_loss = SiLoss(lambd=siloss_lambd)
        # self.si_loss = SiLogLoss(lambd=siloss_lambd)
        self.si_loss = F1_Loss()
        self.grad_loss = MultiScaleGradient()
        # self.grad_loss = MSGIL_NORM_Loss()

    def log_normalize_fun(self, depth_map, max_val=None):        
        # Find the maximum value
        max_value = torch.max(depth_map) if max_val is None else max_val
        return torch.log1p(depth_map) / torch.log1p(max_value)
    
    def forward(self, pred, target, valid_mask, eps=1e-8):
        if self.log_normalize:
            pred = self.log_normalize_fun(pred)
            target = self.log_normalize_fun(target)
        
        si_loss_value = self.si_loss(pred, target, valid_mask)
        grad_loss_value = self.grad_loss(pred, target)

        total_loss = si_loss_value + self.grad_loss_weight * grad_loss_value
        
        return total_loss, si_loss_value, grad_loss_value


class FeatureCosLoss(nn.Module):
    def __init__(self, alpha=8.5, beta=0.4):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, student_features, teacher_features):
        total_loss = 0.0
        count = 0

        for s_feat, t_feat in zip(student_features, teacher_features):
            # Ensure features have the same shape
            if s_feat.shape != t_feat.shape:
                raise ValueError(f"Shape mismatch: {s_feat.shape} vs {t_feat.shape}")
            
            # Normalize features to compute cosine similarity
            s_feat_norm = F.normalize(s_feat, dim=-1)
            t_feat_norm = F.normalize(t_feat, dim=-1)

            # Compute cosine similarity
            cosine_sim = torch.sum(s_feat_norm * t_feat_norm, dim=-1)  # Shape: (batch_size, num_tokens)

            # Mask tokens with cosine similarity > alpha
            mask = (cosine_sim <= self.alpha) & (cosine_sim >= self.beta)

            # Compute cosine distance for the valid tokens
            valid_diff = 1.0 - cosine_sim[mask]  # Cosine distance is 1 - cosine similarity

            # Accumulate loss and count valid tokens
            if valid_diff.numel() > 0:
                total_loss += valid_diff.mean()
                count += 1

        # Average loss across all layers
        if count > 0:
            return total_loss / count
        else:
            return torch.tensor(0.0, requires_grad=True).cuda()