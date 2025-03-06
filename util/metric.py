import torch
import numpy as np

dataset2params = {
    "dense": {"clip_distance": 1000.0, "reg_factor": 5.7},
    "mvsec": {"clip_distance": 80.0, "reg_factor": 3.70378},
    "mvsec_2": {"clip_distance": 80.0, "reg_factor": 3.70378},
    "mvsec_3": {"clip_distance": 80.0, "reg_factor": 3.70378},
    "eventscape": {"clip_distance": 1000.0, "reg_factor": 5.7},
}
    # "dense": {"clip_distance": 80.0, "reg_factor": 3.70378},


def convert_nl2abs_depth(depth, clip_distance, reg_factor):
    depth = np.exp(reg_factor * (depth - 1.0))
    depth *= clip_distance
    # depth = np.clip(depth, np.exp(-1 * reg_factor) * clip_distance, clip_distance)
    return depth


def convert_nl2abs_depth_tensor(depth, clip_distance, reg_factor):
    device = depth.device
    clip_distance = torch.tensor(clip_distance).to(device)
    reg_factor = torch.tensor(reg_factor).to(device)

    depth = torch.exp(reg_factor * (depth - 1.0)) * clip_distance
    # min_depth = torch.exp(-1 * reg_factor) * clip_distance
    # depth = torch.clamp(depth, min=min_depth, max=clip_distance)
    return depth


def prepare_depth_data(target, prediction, clip_distance, reg_factor=3.70378):
    """
    Prepare depth data by normalizing and clipping the target and prediction tensors.

    Args:
        target (torch.Tensor): Ground truth depth tensor (Normalized Log Depth).
        prediction (torch.Tensor): Predicted depth tensor (Normalized Log Depth).
        clip_distance (float): Maximum allowable depth value (Absolute Scale).
        reg_factor (float, optional): Regularization factor for prediction normalization. Defaults to 3.70378.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Processed target, prediction tensors, and valid mask.
    """
    # Ensure input tensors are in float32 for calculations
    target = target.float()
    prediction = prediction.float()

    # Normalize prediction (0 - 1)
    prediction = torch.exp(reg_factor * (prediction - torch.ones_like(prediction)))
    target = torch.exp(reg_factor * (target - torch.ones_like(target)))

    # Get back to the absolute values
    target *= clip_distance
    prediction *= clip_distance

    min_depth = torch.exp(-1 * torch.tensor(reg_factor)) * torch.tensor(clip_distance)
    max_depth = clip_distance

    # Handle invalid values in prediction
    prediction[torch.isinf(prediction)] = max_depth
    prediction[torch.isnan(prediction)] = min_depth

    # Create valid mask
    depth_mask = torch.ones_like(target, dtype=torch.bool)
    valid_mask = torch.logical_and(target > min_depth, target < max_depth)
    valid_mask = torch.logical_and(depth_mask, valid_mask)

    return target, prediction, valid_mask


def eval_depth(pred, target, dataset, eps=1e-6):
    assert pred.shape == target.shape
    reg_factor = dataset2params[dataset]["reg_factor"]
    max_depth = dataset2params[dataset]["clip_distance"]
    min_depth = torch.exp(-1 * torch.tensor(reg_factor)) * torch.tensor(max_depth)

    mask = torch.logical_and(target <= max_depth, target >= min_depth)
    target = target[mask]
    pred = pred[mask]

    target = torch.clamp(target, min=min_depth, max=max_depth)
    pred = torch.clamp(pred, min=min_depth, max=max_depth)

    thresh = torch.max((target / pred), (pred / target))

    d1 = torch.sum(thresh < 1.25).float() / len(thresh)
    d2 = torch.sum(thresh < 1.25**2).float() / len(thresh)
    d3 = torch.sum(thresh < 1.25**3).float() / len(thresh)

    diff = pred - target
    diff_log = torch.log(pred) - torch.log(target)

    abs_rel = torch.mean(torch.abs(diff) / target)
    sq_rel = torch.mean(torch.pow(diff, 2) / target)

    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    rmse_log = torch.sqrt(torch.mean(torch.pow(diff_log, 2)))

    log10 = torch.mean(torch.abs(torch.log10(pred) - torch.log10(target)))
    silog = torch.sqrt(
        torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2)
    )

    return {
        "d1": d1.item(),
        "d2": d2.item(),
        "d3": d3.item(),
        "abs_rel": abs_rel.item(),
        "sq_rel": sq_rel.item(),
        "rmse": rmse.item(),
        "rmse_log": rmse_log.item(),
        "log10": log10.item(),
        "silog": silog.item(),
    }


# def eval_depth_ori(pred, target, dataset, eps=1e-6):
#     assert pred.shape == target.shape

#     reg_factor = dataset2params[dataset]["reg_factor"]
#     max_depth = dataset2params[dataset]["clip_distance"]
#     min_depth = torch.exp(-1 * torch.tensor(reg_factor)) * torch.tensor(max_depth)

#     # Create valid mask
#     pred.
#     depth_mask = torch.ones_like(target, dtype=torch.bool)
#     valid_mask = torch.logical_and(target > min_depth, target < max_depth)
#     valid_mask = torch.logical_and(depth_mask, valid_mask)

#     pred = pred[valid_mask] + eps
#     target = target[valid_mask] + eps

#     thresh = torch.max((target / pred), (pred / target))

#     d1 = torch.sum(thresh < 1.25).float() / len(thresh)
#     d2 = torch.sum(thresh < 1.25 ** 2).float() / len(thresh)
#     d3 = torch.sum(thresh < 1.25 ** 3).float() / len(thresh)

#     diff = pred - target
#     diff_log = torch.log(pred) - torch.log(target)

#     abs_rel = torch.mean(torch.abs(diff) / target)
#     sq_rel = torch.mean(torch.pow(diff, 2) / target)

#     rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
#     rmse_log = torch.sqrt(torch.mean(torch.pow(diff_log , 2)))

#     log10 = torch.mean(torch.abs(torch.log10(pred) - torch.log10(target)))
#     silog = torch.sqrt(torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2))

#     return {'d1': d1.item(), 'd2': d2.item(), 'd3': d3.item(), 'abs_rel': abs_rel.item(), 'sq_rel': sq_rel.item(),
#             'rmse': rmse.item(), 'rmse_log': rmse_log.item(), 'log10':log10.item(), 'silog':silog.item()}


def eval_disparity(pred, target, eps=1e-4):
    assert pred.shape == target.shape
    pred = pred + eps
    target = target + eps

    pred = 1.0 / pred
    target = 1.0 / target

    thresh = torch.max((target / pred), (pred / target))

    d1 = torch.sum(thresh < 1.25).float() / len(thresh)
    d2 = torch.sum(thresh < 1.25**2).float() / len(thresh)
    d3 = torch.sum(thresh < 1.25**3).float() / len(thresh)

    diff = pred - target
    diff_log = torch.log(pred) - torch.log(target)

    abs_rel = torch.mean(torch.abs(diff) / target)
    sq_rel = torch.mean(torch.pow(diff, 2) / target)

    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    rmse_log = torch.sqrt(torch.mean(torch.pow(diff_log, 2)))

    log10 = torch.mean(torch.abs(torch.log10(pred) - torch.log10(target)))
    silog = torch.sqrt(
        torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2)
    )

    return {
        "d1": d1.item(),
        "d2": d2.item(),
        "d3": d3.item(),
        "abs_rel": abs_rel.item(),
        "sq_rel": sq_rel.item(),
        "rmse": rmse.item(),
        "rmse_log": rmse_log.item(),
        "log10": log10.item(),
        "silog": silog.item(),
    }
