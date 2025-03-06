from hmac import new
import numpy as np
import torch
import argparse
import glob
from os.path import join
import tqdm
import cv2
import json
from pprint import pprint
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def FLAGS():
    parser = argparse.ArgumentParser("""Event Depth Data Evaluation.""")

    # training / validation dataset
    parser.add_argument("--dataset", choices=["dense", "mvsec", "eventscape"])
    parser.add_argument("--target_dataset", default="gt")
    parser.add_argument("--predictions_dataset", default="pred_depth")
    parser.add_argument("--event_masks", default="")
    parser.add_argument("--crop_ymax", default=260, type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--idx", type=int, default=-1)
    parser.add_argument("--start_idx", type=int, default=-1)
    parser.add_argument("--prediction_offset", type=int, default=0)
    parser.add_argument("--target_offset", type=int, default=0)
    parser.add_argument("--clip_distance", type=float, default=80.0)
    parser.add_argument("--output_folder", type=str, default=None)
    parser.add_argument("--disparity", action="store_true")
    parser.add_argument("--alignment", action="store_true")
    parser.add_argument("--metric", action="store_true")
    parser.add_argument("--nan_mask", action="store_true")
    parser.add_argument("--json_path", default=None)

    flags = parser.parse_args()

    return flags


metrics_keywords = [
    f"_abs_rel_diff",
    f"_squ_rel_diff",
    f"_RMS_linear",
    f"_RMS_log",
    f"_SILog",
    f"_mean_target_depth",
    f"_median_target_depth",
    f"_mean_prediction_depth",
    f"_median_prediction_depth",
    f"_mean_depth_error",
    f"_median_diff",
    f"_threshold_delta_1.25",
    f"_threshold_delta_1.25^2",
    f"_threshold_delta_1.25^3",
    f"_10_mean_target_depth",
    f"_10_median_target_depth",
    f"_10_mean_prediction_depth",
    f"_10_median_prediction_depth",
    f"_10_abs_rel_diff",
    f"_10_squ_rel_diff",
    f"_10_RMS_linear",
    f"_10_RMS_log",
    f"_10_SILog",
    f"_10_mean_depth_error",
    f"_10_median_diff",
    f"_10_threshold_delta_1.25",
    f"_10_threshold_delta_1.25^2",
    f"_10_threshold_delta_1.25^3",
    f"_20_abs_rel_diff",
    f"_20_squ_rel_diff",
    f"_20_RMS_linear",
    f"_20_RMS_log",
    f"_20_SILog",
    f"_20_mean_target_depth",
    f"_20_median_target_depth",
    f"_20_mean_prediction_depth",
    f"_20_median_prediction_depth",
    f"_20_mean_depth_error",
    f"_20_median_diff",
    f"_20_threshold_delta_1.25",
    f"_20_threshold_delta_1.25^2",
    f"_20_threshold_delta_1.25^3",
    f"_30_abs_rel_diff",
    f"_30_squ_rel_diff",
    f"_30_RMS_linear",
    f"_30_RMS_log",
    f"_30_SILog",
    f"_30_mean_target_depth",
    f"_30_median_target_depth",
    f"_30_mean_prediction_depth",
    f"_30_median_prediction_depth",
    f"_30_mean_depth_error",
    f"_30_median_diff",
    f"_30_threshold_delta_1.25",
    f"_30_threshold_delta_1.25^2",
    f"_30_threshold_delta_1.25^3",
]

dataset2params = {
    # "dense": {"clip_distance": 1000.0, "reg_factor": 6.2044},
    # "dense": {"clip_distance": 80.0, "reg_factor": 3.70378},
    "dense": {"clip_distance": 1000.0, "reg_factor": 5.7},
    "mvsec": {"clip_distance": 80.0, "reg_factor": 3.70378},
    "eventscape": {"clip_distance": 1000.0, "reg_factor": 5.7},
}


def prepare_depth(depth, reg_factor, d_max):
    # Convert metric depth to normalized log depth
    depth = np.clip(depth, 0.0, d_max)
    depth = depth / d_max
    depth = np.log(depth + 1e-6) / reg_factor + 1.0
    depth = depth.clip(0.0, 1.0)
    return depth


def inv_depth_to_depth(prediction, reg_factor=3.70378, eps=1e-6):
    # convert to normalize depth (target is coming in log inverse depth)
    prediction = np.exp(reg_factor * (prediction - 1.0))

    # Perform inverse depth (so now is normalized depth)
    prediction = 1 / prediction
    prediction = prediction / np.amax(prediction)

    # Convert back to log depth (but now it is log  depth)
    prediction = 1.0 + np.log(prediction) / reg_factor
    return prediction


def prepare_depth_data(target, prediction, clip_distance, reg_factor=3.70378):
    # normalize prediction (0 - 1)
    # print(prediction.min(), prediction.max())
    prediction = np.exp(reg_factor * (prediction - 1.0))

    # clip target and normalize
    target = np.clip(target, 0, clip_distance)
    target = target / np.amax(target[~np.isnan(target)])

    # Get back to the absolute values
    target *= clip_distance
    prediction *= clip_distance

    return target, prediction


def align_depth_least_square(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    return_scale_shift=True,
):
    ori_shape = pred_arr.shape

    gt = gt_arr.squeeze().reshape((-1, 1))
    pred = pred_arr.squeeze().reshape((-1, 1))

    # numpy solver
    _ones = np.ones_like(pred)
    A = np.concatenate([pred, _ones], axis=-1)
    X = np.linalg.lstsq(A, gt, rcond=None)[0]
    scale, shift = X

    aligned_pred = pred * scale + shift

    # restore dimensions
    aligned_pred = aligned_pred.reshape(ori_shape)

    if return_scale_shift:
        return aligned_pred, scale, shift
    else:
        return aligned_pred


def filter_list2(list1, list2):
    import os

    filenames_list1 = {os.path.splitext(os.path.basename(path))[0] for path in list1}

    filtered_list2 = [
        path
        for path in list2
        if os.path.splitext(os.path.basename(path))[0] in filenames_list1
    ]

    return filtered_list2


def display_high_contrast_colormap(
    idx,
    target,
    prediction,
    prefix="",
    colormap="terrain",
    debug=False,
    folder_name=None,
):

    if folder_name is not None or debug:
        percent = 1.0
        fig, ax = plt.subplots(ncols=1, nrows=2)
        target_plot = np.flip(np.fliplr(np.clip(target, 0, percent * np.max(target))))
        # ax[0].contour(target_plot, levels=[0.5 * np.median(target)], colors='k', linestyles='-')
        pcm = ax[0].pcolormesh(
            target_plot,
            cmap=colormap,
            vmin=np.min(target),
            vmax=percent * np.max(target),
        )
        ax[0].set_xticklabels([])  # no tick numbers in the target plot horizontal axis
        ax[0].set_title("Target")
        fig.colorbar(pcm, ax=ax[0], extend="both", orientation="vertical")
        prediction_plot = np.flip(
            np.fliplr(np.clip(prediction, 0, percent * np.max(prediction)))
        )
        # ax[1].contour(prediction_plot, levels=[0.5 * np.median(target)], colors='k', linestyles='-')
        pcm = ax[1].pcolormesh(
            prediction_plot,
            cmap=colormap,
            vmin=np.min(target),
            vmax=percent * np.max(target),
        )
        ax[1].set_title("Prediction")
        fig.colorbar(pcm, ax=ax[1], extend="both", orientation="vertical")
        fig.canvas.set_window_title(prefix + "High_Contrast_Depth_Evaluation")
    if folder_name is not None:
        plt.savefig("%s/frame_%010d.png" % (folder_name, idx))
        plt.close(fig)
    if debug:
        plt.show()


def display_high_contrast_color_logmap(
    idx, data, prefix="", name="data", colormap="tab20c", debug=False, folder_name=None
):

    if debug and folder_name is not None:
        percent = 1.0
        fig, ax = plt.subplots(ncols=1, nrows=1)
        target_plot = np.flip(np.fliplr(np.clip(data, 0, percent * np.max(data))))
        # print ("median: ", np.median(data))
        # ax.contour(target_plot, Z = np.median(data), levels=[np.median(data)], colors='k', linestyles='-')
        # pcm = ax.pcolormesh(target_plot, vmin=np.min(data), vmax=np.max(data), cmap=colormap)
        pcm = ax.pcolormesh(
            target_plot,
            norm=colors.LogNorm(vmin=np.min(data), vmax=np.max(data)),
            cmap=colormap,
        )
        ax.set_yticklabels([])  # no tick numbers in the target plot horizontal axis
        ax.set_xticklabels([])  # no tick numbers in the target plot horizontal axis
        # cbar = fig.colorbar(pcm, ax=ax, extend='both', orientation='vertical')
        # cbar.ax.set_yticklabels(['10', '20', '30', '40', '50' '60'])  # vertically oriented colorbar
        fig.canvas.set_window_title(prefix + "High_Contrast_Depth_Evaluation")
        plt.savefig("%s/%s_frame_%010d.png" % (folder_name, name, idx))
        # plt.show()


def add_to_metrics(
    idx,
    metrics,
    target_,
    prediction_,
    mask,
    event_frame=None,
    prefix="",
    debug=False,
    output_folder=None,
):
    if len(metrics) == 0:
        metrics = {k: 0 for k in metrics_keywords}

    prediction_mask = (prediction_ > 0) & (
        prediction_ <= np.amax(target_[~np.isnan(target_)])
    )
    depth_mask = (target_ > 0) & (
        target_ <= np.amax(target_[~np.isnan(target_)])
    )  # make (target> 3) for mvsec might drives
    mask = mask & depth_mask & prediction_mask
    eps = 1e-5

    target = target_[
        mask
    ]  # np.where(mask, target_, np.max(target_[~np.isnan(target_)]))# target_[mask] but without lossing shape
    prediction = prediction_[
        mask
    ]  # np.where(mask, prediction_, np.max(target_[~np.isnan(target_)]))# prediction_[mask] but without lossing shape

    # thresholds
    # ratio = np.max(np.stack([target/(prediction+eps),prediction/(target+eps)]), axis=0)
    ratio = np.maximum((target / (prediction + eps)), (prediction / (target + eps)))

    new_metrics = {}
    new_metrics[f"{prefix}threshold_delta_1.25"] = np.mean(ratio <= 1.25)
    new_metrics[f"{prefix}threshold_delta_1.25^2"] = np.mean(ratio <= 1.25**2)
    new_metrics[f"{prefix}threshold_delta_1.25^3"] = np.mean(ratio <= 1.25**3)

    # abs diff
    # log_diff = np.log(target+eps)-np.log(prediction+eps)
    log_diff = np.log(prediction + eps) - np.log(target + eps)
    # log_diff = np.abs(log_target - log_prediction)
    abs_diff = np.abs(target - prediction)

    new_metrics[f"{prefix}abs_rel_diff"] = (abs_diff / (target + eps)).mean()
    # new_metrics[f"{prefix}squ_rel_diff"] = (abs_diff**2 / (target**2 + eps)).mean()
    new_metrics[f"{prefix}squ_rel_diff"] = (abs_diff**2 / (target + eps)).mean()
    new_metrics[f"{prefix}RMS_linear"] = np.sqrt((abs_diff**2).mean())
    new_metrics[f"{prefix}RMS_log"] = np.sqrt((log_diff**2).mean())
    new_metrics[f"{prefix}SILog"] = (log_diff**2).mean() - (log_diff.mean()) ** 2
    new_metrics[f"{prefix}mean_target_depth"] = target.mean()
    new_metrics[f"{prefix}median_target_depth"] = np.median(target)
    new_metrics[f"{prefix}mean_prediction_depth"] = prediction.mean()
    new_metrics[f"{prefix}median_prediction_depth"] = np.median(prediction)
    new_metrics[f"{prefix}mean_depth_error"] = abs_diff.mean()
    new_metrics[f"{prefix}median_diff"] = np.abs(
        np.median(target) - np.median(prediction)
    )

    for k, v in new_metrics.items():
        metrics[k] += v

    if debug:
        pprint(new_metrics)
        {print("%s : %f" % (k, v)) for k, v in new_metrics.items()}
        fig, ax = plt.subplots(ncols=3, nrows=4)
        ax[0, 0].imshow(target_, vmin=0, vmax=200)
        ax[0, 0].set_title("target depth")
        ax[0, 1].imshow(prediction_, vmin=0, vmax=200)
        ax[0, 1].set_title("prediction depth")
        target_debug = target_.copy()
        target_debug[~mask] = 0
        ax[0, 2].imshow(target_debug, vmin=0, vmax=200)
        ax[0, 2].set_title("target depth masked")

        ax[1, 0].imshow(np.log(target_ + eps), vmin=0, vmax=np.log(200))
        ax[1, 0].set_title("log target")
        ax[1, 1].imshow(np.log(prediction_ + eps), vmin=0, vmax=np.log(200))
        ax[1, 1].set_title("log prediction")
        ax[1, 2].imshow(
            np.max(
                np.stack(
                    [target_ / (prediction_ + eps), prediction_ / (target_ + eps)]
                ),
                axis=0,
            )
        )
        ax[1, 2].set_title("max ratio")

        ax[2, 0].imshow(np.abs(np.log(target_ + eps) - np.log(prediction_ + eps)))
        ax[2, 0].set_title("abs log diff")
        ax[2, 1].imshow(np.abs(target_ - prediction_))
        ax[2, 1].set_title("abs diff")
        if event_frame is not None:
            a = np.zeros(event_frame.shape)
            a[:, :, 0] = np.sum(event_frame.astype("float32"), axis=-1) > 0
            a[:, :, 1] = np.clip(target_.copy(), 0, 1)
            ax[2, 2].imshow(a)
            ax[2, 2].set_title("event frame")

        log_diff_ = np.abs(np.log(target_ + eps) - np.log(prediction_ + eps))
        log_diff_[~mask] = 0
        ax[3, 0].imshow(log_diff_)
        ax[3, 0].set_title("abs log diff masked")
        abs_diff_ = np.abs(target_ - prediction_)
        abs_diff_[~mask] = 0
        ax[3, 1].imshow(abs_diff_)
        ax[3, 1].set_title("abs diff masked")
        ax[3, 2].imshow(mask)
        ax[3, 2].set_title("mask frame")

        mx = np.max(abs_diff_)
        # print(np.where(abs_diff_> 0.9*mx))
        fig.canvas.set_window_title(prefix + "_Depth_Evaluation")
        plt.show()

    return metrics


if __name__ == "__main__":
    flags = FLAGS()

    reg_factor = dataset2params[flags.dataset]["reg_factor"]
    min_depth = np.exp(-1 * reg_factor) * flags.clip_distance
    max_depth = flags.clip_distance
    print(f"min_depth: {min_depth}, max_depth: {max_depth}")

    # predicted labels
    prediction_files = sorted(glob.glob(join(flags.predictions_dataset, "*.npy")))
    prediction_files = prediction_files[flags.prediction_offset :]

    target_files = sorted(glob.glob(join(flags.target_dataset, "*.npy")))
    target_files = target_files[flags.target_offset :]

    if flags.event_masks != "":
        event_frame_files = sorted(glob.glob(join(flags.event_masks, "*png")))
        event_frame_files = event_frame_files[flags.prediction_offset :]

    # Information about the dataset length
    print("len of prediction files", len(prediction_files))
    print("len of target files", len(target_files))

    if flags.event_masks != "":
        print("len of events files", len(event_frame_files))

    assert len(prediction_files) > 0
    assert len(target_files) > 0

    # target_files = filter_list2(prediction_files, target_files)

    assert len(target_files) == len(prediction_files)

    if flags.event_masks != "":
        use_event_masks = len(event_frame_files) > 0
    else:
        use_event_masks = False

    metrics = {}

    num_it = len(target_files)
    for idx in tqdm.tqdm(range(num_it)):
        p_file, t_file = prediction_files[idx], target_files[idx]

        # Read absolute scale ground truth
        target_depth = np.load(t_file)
        # Crop depth height according to argument
        target_depth = target_depth[: flags.crop_ymax]

        # Read predicted depth data
        predicted_depth = np.load(p_file)
        # Crop depth height according to argument
        predicted_depth = predicted_depth[: flags.crop_ymax]

        # only considering pixels with valid depth measurements.
        mask = (target_depth >= min_depth) & (target_depth <= max_depth)
        target_depth = target_depth[mask]
        predicted_depth = predicted_depth[mask]

        assert target_depth.ndim <= 2
        assert predicted_depth.ndim <= 2

        # Flat numpy arrary
        if target_depth.ndim == 2:
            target_depth = target_depth.ravel()
            predicted_depth = predicted_depth.ravel()

        # Convert disparity to depth
        if flags.disparity:
            predicted_depth[predicted_depth == 0.0] = 1e-6
            predicted_depth = 1 / predicted_depth
            predicted_depth = np.clip(predicted_depth, min_depth, max_depth)

        # Convert normalized depth to the metric scale
        if not flags.metric:
            target_depth, predicted_depth = prepare_depth_data(
                target_depth,
                predicted_depth,
                flags.clip_distance,
                reg_factor=reg_factor,
            )
        # target_depth = np.clip(target_depth, min_depth, max_depth)

        # Align with GT using least square
        if flags.alignment:
            # print(predicted_depth.min(), predicted_depth.max())
            predicted_depth, scale, shift = align_depth_least_square(
                gt_arr=target_depth,
                pred_arr=predicted_depth,
                return_scale_shift=True,
            )
            # print(scale, shift)
            # print(predicted_depth.min(), predicted_depth.max())
            # exit()
            predicted_depth = np.clip(predicted_depth, min_depth, max_depth)

        assert predicted_depth.shape == target_depth.shape

        depth_mask = np.ones_like(target_depth) > 0
        debug = flags.debug and idx == flags.idx

        # print("min pred", np.min(predicted_depth[~np.isnan(predicted_depth)]))
        # print("max pred", np.max(predicted_depth[~np.isnan(predicted_depth)]))
        # print("min target", np.min(target_depth[~np.isnan(target_depth)]))
        # print("max target", np.max(target_depth[~np.isnan(target_depth)]))
        # diff = np.abs(target_depth - predicted_depth)
        # print(diff.min(), diff.max(), diff.mean())
        # exit()
        metrics = add_to_metrics(
            idx,
            metrics,
            target_depth,
            predicted_depth,
            depth_mask,
            event_frame=None,
            prefix="_",
            debug=debug,
            output_folder=flags.output_folder,
        )

        if flags.dataset == "mvsec" or flags.dataset == "dense":
            for depth_threshold in [10, 20, 30]:
                depth_threshold_mask = np.nan_to_num(target_depth) < depth_threshold
                add_to_metrics(
                    -1,
                    metrics,
                    target_depth,
                    predicted_depth,
                    depth_threshold_mask,
                    prefix=f"_{depth_threshold}_",
                    debug=debug,
                )

        if use_event_masks:
            ev_frame_file = event_frame_files[idx]
            event_frame = cv2.imread(ev_frame_file)
            event_frame = event_frame[: flags.crop_ymax]
            event_mask = np.sum(event_frame.astype("float32"), axis=-1) > 0

            assert event_mask.shape == target_depth.shape
            add_to_metrics(
                -1,
                metrics,
                target_depth,
                predicted_depth,
                event_mask,
                event_frame=event_frame,
                prefix="event_masked_",
                debug=debug,
            )

            for depth_threshold in [10, 20, 30]:
                depth_threshold_mask = np.nan_to_num(target_depth) < depth_threshold
                # $debug=True
                add_to_metrics(
                    -1,
                    metrics,
                    target_depth,
                    predicted_depth,
                    event_mask & depth_threshold_mask,
                    event_frame=event_frame,
                    prefix=f"event_masked_{depth_threshold}_",
                    debug=debug,
                )

    # pprint({k: v / num_it for k, v in metrics.items()})
    new_metrics = {k: v / num_it for k, v in metrics.items()}
    print(json.dumps(new_metrics, indent=4))
    # {print("%s : %f" % (k, v / num_it)) for k, v in metrics.items()}
