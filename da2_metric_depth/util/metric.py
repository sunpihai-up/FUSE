import torch
import numpy as np

dataset2params = {
    "dense": {"clip_distance": 1000.0, "reg_factor": 6.2044},
    "mvsec": {"clip_distance": 80.0, "reg_factor": 3.70378},
    "eventscape": {"clip_distance": 1000.0, "reg_factor": 5.7},
    "eventscape_voxel": {"clip_distance": 1000.0, "reg_factor": 5.7},
    "mvsec_voxel": {"clip_distance": 80.0, "reg_factor": 3.70378},
}

def convert_nl2abs_depth(depth, clip_distance, reg_factor):
    """Converts normalized logarithmic depth values to absolute depth values.

    Args:
        depth (numpy.ndarray): Input depth map in normalized logarithmic scale.
        clip_distance (float): Maximum depth value (used for scaling and clipping).
        reg_factor (float):  Regularization factor used in the logarithmic transformation.

    Returns:
        numpy.ndarray: Absolute depth map
    """
    depth = np.exp(reg_factor * (depth - 1.0))
    depth *= clip_distance
    depth = np.clip(depth, np.exp(-1 * reg_factor) * clip_distance, clip_distance)
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

def eval_depth(pred, target, dataset='dense', eps=1e-6):
    assert pred.shape == target.shape

    reg_factor = dataset2params[dataset]["reg_factor"]
    max_depth = dataset2params[dataset]["clip_distance"]
    
    # Convert to the correct scale
    pred, target, valid_mask = prepare_depth_data(
        target, pred, clip_distance=max_depth, reg_factor=reg_factor
    )
    
    pred = pred[valid_mask] + eps
    target = target[valid_mask] + eps

    thresh = torch.max((target / pred), (pred / target))

    d1 = torch.sum(thresh < 1.25).float() / len(thresh)
    d2 = torch.sum(thresh < 1.25 ** 2).float() / len(thresh)
    d3 = torch.sum(thresh < 1.25 ** 3).float() / len(thresh)

    diff = pred - target
    diff_log = torch.log(pred) - torch.log(target)

    abs_rel = torch.mean(torch.abs(diff) / target)
    sq_rel = torch.mean(torch.pow(diff, 2) / target)

    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    rmse_log = torch.sqrt(torch.mean(torch.pow(diff_log , 2)))

    log10 = torch.mean(torch.abs(torch.log10(pred) - torch.log10(target)))
    silog = torch.sqrt(torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2))

    return {'d1': d1.item(), 'd2': d2.item(), 'd3': d3.item(), 'abs_rel': abs_rel.item(), 'sq_rel': sq_rel.item(), 
            'rmse': rmse.item(), 'rmse_log': rmse_log.item(), 'log10':log10.item(), 'silog':silog.item()}

def eval_depth_ori(pred, target, dataset, eps=1e-6):
    assert pred.shape == target.shape
    
    reg_factor = dataset2params[dataset]["reg_factor"]
    max_depth = dataset2params[dataset]["clip_distance"]
    min_depth = torch.exp(-1 * torch.tensor(reg_factor)) * torch.tensor(max_depth)
    
    # Create valid mask
    depth_mask = torch.ones_like(target, dtype=torch.bool)
    valid_mask = torch.logical_and(target > min_depth, target < max_depth)
    valid_mask = torch.logical_and(depth_mask, valid_mask)
    
    pred = pred[valid_mask] + eps
    target = target[valid_mask] + eps
    
    thresh = torch.max((target / pred), (pred / target))

    d1 = torch.sum(thresh < 1.25).float() / len(thresh)
    d2 = torch.sum(thresh < 1.25 ** 2).float() / len(thresh)
    d3 = torch.sum(thresh < 1.25 ** 3).float() / len(thresh)

    diff = pred - target
    diff_log = torch.log(pred) - torch.log(target)

    abs_rel = torch.mean(torch.abs(diff) / target)
    sq_rel = torch.mean(torch.pow(diff, 2) / target)

    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    rmse_log = torch.sqrt(torch.mean(torch.pow(diff_log , 2)))

    log10 = torch.mean(torch.abs(torch.log10(pred) - torch.log10(target)))
    silog = torch.sqrt(torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2))

    return {'d1': d1.item(), 'd2': d2.item(), 'd3': d3.item(), 'abs_rel': abs_rel.item(), 'sq_rel': sq_rel.item(), 
            'rmse': rmse.item(), 'rmse_log': rmse_log.item(), 'log10':log10.item(), 'silog':silog.item()}
    