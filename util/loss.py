import torch
from torch import nn
import torch.nn.functional as F
import torch
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
    def __init__(self, siloss_lambd=0.5, grad_loss_weight=0.5):
        super().__init__()
        self.siloss_lambd = siloss_lambd
        self.grad_loss_weight = grad_loss_weight

        self.si_loss = SiLoss(lambd=siloss_lambd)
        self.grad_loss = MultiScaleGradient()

    def forward(self, pred, target, valid_mask, eps=1e-8):
        si_loss_value = self.si_loss(pred, target, valid_mask)
        grad_loss_value = self.grad_loss(pred, target)

        total_loss = si_loss_value + self.grad_loss_weight * grad_loss_value
        
        return total_loss, si_loss_value, grad_loss_value
