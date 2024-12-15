import torch
from torch import nn


class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target, valid_mask, eps=1e-8):
        valid_mask = valid_mask.detach()
        diff_log = torch.log(target[valid_mask] + eps) - torch.log(pred[valid_mask] + eps)
        loss = torch.sqrt(torch.pow(diff_log, 2).mean() -
                          self.lambd * torch.pow(diff_log.mean(), 2))
        return loss
