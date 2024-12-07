import numpy as np
import glob
import argparse
import tqdm
from os.path import join



def paras_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--target_dataset", default="gt")
    parser.add_argument("--predictions_dataset", default="pred_depth")
    parser.add_argument("--split-file", default=None, type=str)
    parser.add_argument("--clip_distance", default=1000.0, type=float)
    
    args = parser.parse_args()
    return args

def initialize_results():
    return {
        "d1": np.array([0.0]),
        "d2": np.array([0.0]),
        "d3": np.array([0.0]),
        "abs_rel": np.array([0.0]),
        "sq_rel": np.array([0.0]),
        "rmse": np.array([0.0]),
        "rmse_log": np.array([0.0]),
        "log10": np.array([0.0]),
        "silog": np.array([0.0]),
    }

def log_results(results):
    print("=" * 90)
    print("{:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}".format(
        *tuple(results.keys())
    ))
    print("{:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}".format(
        *tuple([v for v in results.values()])
    ))
    print("=" * 90)
    print()

def eval_depth(pred, target, dataset='dense', eps=1e-6):
    assert pred.shape == target.shape

    # reg_factor = dataset2params[dataset]["reg_factor"]
    # max_depth = dataset2params[dataset]["clip_distance"]

    # # Convert to the correct scale
    # pred, target, valid_mask = prepare_depth_data(
    #     target, pred, clip_distance=max_depth, reg_factor=reg_factor
    # )

    # pred = pred[valid_mask]
    # target = target[valid_mask]

    pred = pred + eps
    target = target + eps
    
    thresh = np.maximum(target / pred, pred / target)

    d1 = np.sum(thresh < 1.25) / len(thresh)
    d2 = np.sum(thresh < 1.25 ** 2) / len(thresh)
    d3 = np.sum(thresh < 1.25 ** 3) / len(thresh)

    diff = pred - target
    diff_log = np.log(pred) - np.log(target)

    abs_rel = np.mean(np.abs(diff) / target)
    sq_rel = np.mean(diff ** 2 / target)

    rmse = np.sqrt(np.mean(diff ** 2))
    rmse_log = np.sqrt(np.mean(diff_log ** 2))

    log10 = np.mean(np.abs(np.log10(pred) - np.log10(target)))
    silog = np.sqrt(np.mean(diff_log ** 2) - 0.5 * (np.mean(diff_log) ** 2))

    return {
        'd1': d1, 'd2': d2, 'd3': d3,
        'abs_rel': abs_rel, 'sq_rel': sq_rel,
        'rmse': rmse, 'rmse_log': rmse_log,
        'log10': log10, 'silog': silog
    }


if __name__ == "__main__":
    args = paras_args()
    
    if args.split_file is None:
        prediction_files = sorted(glob.glob(join(args.predictions_dataset, "*.npy")))
        target_files = sorted(glob.glob(join(args.target_dataset, "*.npy")))
    else:
        with open(args.split_file, 'r') as f:
            lines = f.readlines()
            # TODO
    
    # Information about the dataset length
    print("len of prediction files", len(prediction_files))
    print("len of target files", len(target_files))
    
    assert len(target_files) == len(prediction_files)
    
    results = initialize_results()
    sample_num = 0
    
    for idx in tqdm.tqdm(range(len(target_files))):
        p_file, t_file = prediction_files[idx], target_files[idx]
        
        # Read absolute scale ground truth
        target_depth = np.load(t_file)
        # Read absolute scale predicted depth data
        predicted_depth = np.load(p_file)
        
        assert target_depth.ndim <= 2
        assert predicted_depth.ndim <= 2
        
        if target_depth.ndim == 2:
            target_depth = target_depth.ravel()
            predicted_depth = predicted_depth.ravel()
        
        target_depth = np.clip(target_depth, 0, args.clip_distance)
        predicted_depth = np.clip(predicted_depth, 0, args.clip_distance)
        
        cur_results = eval_depth(target_depth, predicted_depth)
        
        for k in results.keys():
            results[k] += cur_results[k]
        sample_num += 1
    
    for k in results.keys():
        results[k] = (results[k] / sample_num).item()
    
    log_results(results)