import torch
from torch import nn
import torch.nn.functional as F
from kornia.filters.sobel import spatial_gradient, sobel


class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5, do_sqrt=False):
        super().__init__()
        self.lambd = lambd
        self.sqrt = do_sqrt

    def forward(self, pred, target, valid_mask, eps=1e-8):
        valid_mask = valid_mask.detach()

        target = target[valid_mask] + eps
        pred = pred[valid_mask] + eps

        diff_log = torch.log(target) - torch.log(pred)
        d_square_mean = torch.pow(diff_log, 2).mean()
        d_mean_square = torch.pow(diff_log.mean(), 2)

        loss = d_square_mean - self.lambd * d_mean_square
        if self.sqrt:
            loss = torch.sqrt(loss)

        if torch.isnan(loss).item() or torch.isinf(loss).item():
            raise RuntimeError(
                f"Silog error, {loss}, d_square_mean: {d_square_mean}, d_mean: {d_mean_square}"
            )
        return loss


class SiLoss(nn.Module):
    def __init__(self, lambd=0.5, do_sqrt=False):
        super().__init__()
        self.lambd = lambd
        self.sqrt = do_sqrt

    def forward(self, pred, target, valid_mask):
        valid_mask = valid_mask.detach()
        pred = pred[valid_mask]
        target = target[valid_mask]

        diff = target - pred
        d_square_mean = torch.pow(diff, 2).mean()
        d_mean_square = torch.pow(diff.mean(), 2)

        loss = d_square_mean - self.lambd * d_mean_square
        if self.sqrt:
            loss = torch.sqrt(loss)

        if torch.isnan(loss).item() or torch.isinf(loss).item():
            raise RuntimeError(
                f"Siloss error, {loss}, d_square_mean: {d_square_mean}, d_mean: {d_mean_square}"
            )
        return loss


class L1_Loss(nn.Module):
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
        loss = diff.mean()

        if torch.isnan(loss).item() or torch.isinf(loss).item():
            raise RuntimeError(f"L1 NAN error, {loss}")
        return loss


class MultiScaleGradient(torch.nn.Module):
    def __init__(self, start_scale=1, num_scales=4, log_space=False, eps=1e-6):
        super(MultiScaleGradient, self).__init__()

        self.start_scale = start_scale
        self.num_scales = num_scales
        self.log_space = log_space
        self.eps = eps

        self.multi_scales = [
            torch.nn.AvgPool2d(
                self.start_scale * (2**scale), self.start_scale * (2**scale)
            )
            for scale in range(self.num_scales)
        ]

    def forward(self, prediction, target, mask=None):
        if prediction.ndim == 3:
            prediction = prediction.unsqueeze(1)
            target = target.unsqueeze(1)
            if mask != None:
                mask = mask.unsqueeze(1)

        target = target + (~mask) * 1000

        loss = 0
        if self.log_space:
            prediction = torch.log(prediction)
            target = torch.log(target)

        diff = prediction - target
        _, _, H, W = target.shape

        for m in self.multi_scales:
            # input and type are of the type [B x C x H x W]
            diff_pooled = m(diff)
            if mask != None:
                mask_pooled = m(mask.float())
                # Determine whether all the original windows of the pooled mask are valid
                # (with an average value of 1.0)
                mask_scale = torch.isclose(mask_pooled, torch.ones_like(mask_pooled))

            # Use kornia spatial gradient computation
            # output of kornia spatial gradient is [B x C x 2 x H x W]
            delta_diff = spatial_gradient(diff_pooled)
            is_nan = torch.isnan(delta_diff)

            if mask != None:
                # Expand mask_scale to match the dimension of delta_diff
                B, C, _, H_p, W_p = delta_diff.shape
                mask_scale = mask_scale.view(B, 1, 1, H_p, W_p).expand_as(delta_diff)

            valid_mask = ~is_nan
            if mask != None:
                # Combine the valid regions of non-NaN and mask
                valid_mask = valid_mask & mask_scale
            valid_count = valid_mask.sum()

            loss += torch.abs(delta_diff[valid_mask]).sum() / valid_count

        if torch.isnan(loss).item() | torch.isinf(loss).item():
            raise RuntimeError(f"MultiScaleGradient error, {loss}")
        return loss / self.num_scales


class GradientLoss_Li(nn.Module):
    def __init__(
        self,
        scale_num=4,
        loss_weight=1,
        step=2,
        data_type=["lidar", "stereo"],
    ):
        super(GradientLoss_Li, self).__init__()
        self.__scales = scale_num
        self.loss_weight = loss_weight
        self.data_type = data_type
        self.step = step
        self.eps = 1e-6

    def gradient_log_loss(self, log_prediction_d, log_gt, mask, step=2):
        log_d_diff = log_prediction_d - log_gt

        v_gradient = torch.abs(log_d_diff[:, :, :-step, :] - log_d_diff[:, :, step:, :])
        v_mask = torch.mul(mask[:, :, :-step, :], mask[:, :, step:, :])
        v_gradient = torch.mul(v_gradient, v_mask)

        h_gradient = torch.abs(log_d_diff[:, :, :, :-step] - log_d_diff[:, :, :, step:])
        h_mask = torch.mul(mask[:, :, :, :-step], mask[:, :, :, step:])
        h_gradient = torch.mul(h_gradient, h_mask)

        N = torch.sum(h_mask) + torch.sum(v_mask) + self.eps

        gradient_loss = torch.sum(h_gradient) + torch.sum(v_gradient)
        gradient_loss = gradient_loss / N

        return gradient_loss

    def forward(self, prediction, target, mask):
        if prediction.ndim == 3:
            prediction = prediction.unsqueeze(1)
            target = target.unsqueeze(1)
            mask = mask.unsqueeze(1)

        total = 0
        target_trans = target + (~mask) * 1000
        pred_log = torch.log(prediction)
        gt_log = torch.log(target_trans)
        for scale in range(self.__scales):
            step = pow(2, scale)

            total += self.gradient_log_loss(
                pred_log[:, ::step, ::step],
                gt_log[:, ::step, ::step],
                mask[:, ::step, ::step],
                step=self.step,
            )
        loss = total / self.__scales
        if torch.isnan(loss).item() | torch.isinf(loss).item():
            raise RuntimeError(f"GradientLoss_Li error, {loss}")
        return loss * self.loss_weight


class MixedLoss(nn.Module):
    def __init__(
        self,
        siloss_lambd=0.5,
        grad_loss_weight=0.25,
        do_sqrt=False,
        scale_num=4,
        log_normalize=False,
    ):
        super().__init__()
        self.siloss_lambd = siloss_lambd
        self.grad_loss_weight = grad_loss_weight
        self.do_sqrt = do_sqrt
        self.scale_num = scale_num
        self.log_normalize = log_normalize

        self.si_loss = SiLogLoss(lambd=siloss_lambd, do_sqrt=do_sqrt)
        # self.si_loss = SiLoss(lambd=siloss_lambd)
        # self.si_loss = L1_Loss()

        # self.grad_loss = MultiScaleGradient(log_space=False)
        self.grad_loss = GradientLoss_Li(scale_num=scale_num, step=2)

    def log_normalize_fun(self, depth_map, max_val=None):
        # Find the maximum value
        max_value = torch.max(depth_map) if max_val is None else max_val
        return torch.log1p(depth_map) / torch.log1p(max_value)

    def forward(self, pred, target, valid_mask, eps=1e-8):
        if self.log_normalize:
            pred = self.log_normalize_fun(pred)
            target = self.log_normalize_fun(target)

        si_loss_value = self.si_loss(pred, target, valid_mask)
        grad_loss_value = self.grad_loss(pred, target, valid_mask)

        total_loss = si_loss_value + self.grad_loss_weight * grad_loss_value

        return total_loss, si_loss_value, grad_loss_value


class FeatureCosLoss(nn.Module):
    def __init__(self, alpha=0.85, beta=0.4):
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
            cosine_sim = torch.sum(
                s_feat_norm * t_feat_norm, dim=-1
            )  # Shape: (batch_size, num_tokens)

            # Mask tokens with cosine similarity > alpha
            mask = (cosine_sim <= self.alpha) & (cosine_sim >= self.beta)

            # Compute cosine distance for the valid tokens
            valid_diff = (
                1.0 - cosine_sim[mask]
            )  # Cosine distance is 1 - cosine similarity

            # Accumulate loss and count valid tokens
            if valid_diff.numel() > 0:
                total_loss += valid_diff.mean()
                count += 1

        # Average loss across all layers
        if count > 0:
            return total_loss / count
        else:
            return torch.tensor(0.0, requires_grad=True).cuda()
