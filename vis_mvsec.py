import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from PIL import Image
import matplotlib
import sys
import os
from tqdm import tqdm


def vis_depth_map(
    dep, show_colorbar=False, cmap_name="Spectral", save_fig=False, save_path="test.png"
):
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    cmap.set_bad(color="black")
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.axis("off")

    im = ax.imshow(dep, cmap=cmap, extent=[0, dep.shape[1], 0, dep.shape[0]])
    if show_colorbar:
        plt.colorbar(im)

    dpi = fig.get_dpi()
    fig.set_size_inches(dep.shape[1] / dpi, dep.shape[0] / dpi)

    if save_fig:
        plt.savefig(save_path)
    # plt.show()
    plt.close()


def log_nor(np_array):
    offset = abs(np.min(np_array)) + 1 if np.min(np_array) <= 0 else 0
    np_array_shifted = np_array + offset + 1e-6

    # 应用log变换
    np_array_log = np.log(np_array_shifted)

    # 计算最大值和最小值
    max_val = np.max(np_array_log)
    min_val = np.min(np_array_log)

    # 执行最小-最大归一化
    np_array_normalized = (np_array_log - min_val) / (max_val - min_val)
    return np_array_normalized


def log_nor_nan(np_array):
    non_nan_indices = ~np.isnan(np_array)

    # 如果数据中存在小于或等于0的值，则需要先平移数值以确保所有值都大于0。
    # 注意这里只考虑非nan值
    safe_array = np_array[non_nan_indices]
    offset = abs(np.min(safe_array)) + 1 if np.min(safe_array) <= 0 else 0

    # 应用log变换到非nan部分
    safe_array_log = np.log(safe_array + offset)

    # 计算这部分的最大值和最小值
    max_val = np.max(safe_array_log)
    min_val = np.min(safe_array_log)

    # 对非nan部分执行最小-最大归一化
    normalized_safe_array = (safe_array_log - min_val) / (max_val - min_val)

    # 创建一个与原数组形状相同的输出数组，并填入归一化的值
    np_array_normalized = np.copy(np_array)
    np_array_normalized[non_nan_indices] = normalized_safe_array
    return np_array_normalized


# min_depth, max_depth = 1.97, 80
# out_dir = "/data_nvme/sph/vis_mvsec/day1"
# # myoutdir = os.path.join(out_dir, "lorot")
# myoutdir = os.path.join(out_dir, "da2")
# gtoutdir = os.path.join(out_dir, "gt")
# hmnetoutdir = os.path.join(out_dir, "hmnet_fuse_l3")

# os.makedirs(myoutdir, exist_ok=True)
# os.makedirs(gtoutdir, exist_ok=True)
# os.makedirs(hmnetoutdir, exist_ok=True)

# # res_dir = "/home/sph/event/da2-prompt-tuning/results/test/LoRoT_mvsec_night1_20/npy/"
# res_dir = "/home/sph/event/da2-prompt-tuning/da2_metric_depth/results/da2_sigloss_mvsec_day1/npy"
# gt_dir = "/data_nvme/sph/mvsec_processed/outdoor_day1/depths"
# # hmnet_resp = "/home/sph/event/vis/HMNet_pth/experiments/depth/workspace/hmnet_L3_fuse_rgb/result/pred_night1/outdoor_night1_data.npy"
# # hmnets = np.load(hmnet_resp)

# allres = os.listdir(res_dir)

# # for sam in allres:
# for sam in tqdm(allres, desc="Processing", unit="item"):
#     mypath = os.path.join(res_dir, sam)
#     gtpath = os.path.join(gt_dir, sam)
#     idx = int(sam.split(".")[0])

#     my = np.load(mypath)
#     gt = np.load(gtpath)
#     # hmnet = hmnets[idx][0]

#     mask = (gt >= min_depth) & (gt <= max_depth)

#     my = log_nor(my)
#     # hmnet = log_nor_nan(hmnet)
#     # gt = log_nor_nan(gt)

#     my[~mask] = np.nan
#     # hmnet[~mask] = np.nan
#     # gt[~mask] = np.nan

#     myoutpath = os.path.join(myoutdir, sam.replace("npy", "png"))
#     # gtoutpath = os.path.join(gtoutdir, sam.replace("npy", "png"))
#     # hmnetoutpath = os.path.join(hmnetoutdir, sam.replace("npy", "png"))

#     vis_depth_map(
#         my, save_fig=True, show_colorbar=False, cmap_name="magma_r", save_path=myoutpath
#     )
#     # vis_depth_map(
#     #     gt, save_fig=True, show_colorbar=False, cmap_name="magma_r", save_path=gtoutpath
#     # )
#     # vis_depth_map(
#     #     hmnet,
#     #     save_fig=True,
#     #     show_colorbar=False,
#     #     cmap_name="magma_r",
#     #     save_path=hmnetoutpath,
#     # )


"""
Vis ON DENSE
"""
out_dir = "/data_nvme/sph/vis_mvsec/dense_test_imgcor3"
myoutdir = os.path.join(out_dir, "da2")
gtoutdir = os.path.join(out_dir, "gt")

os.makedirs(myoutdir, exist_ok=True)
os.makedirs(gtoutdir, exist_ok=True)

res_dir = "/home/sph/event/da2-prompt-tuning/da2_metric_depth/results/da2zero_dense_test/npy"
gt_dir = "/data_nvme/sph/DENSE/test/seq0/depth/data"

allres = os.listdir(res_dir)
for sam in tqdm(allres, desc="Processing", unit="item"):
    mypath = os.path.join(res_dir, sam)
    gtpath = os.path.join(gt_dir, sam.replace("frame", "depth"))

    my = np.load(mypath)
    gt = np.load(gtpath)

    mask = (gt == 1000)

    my = log_nor(my)
    # gt = log_nor(gt)
    my[mask] = np.nan
    # gt[mask] = np.nan

    # magma Spectral_r
    myoutpath = os.path.join(myoutdir, sam.replace("npy", "png"))
    gtpath = os.path.join(gtoutdir, sam.replace("npy", "png"))
    vis_depth_map(
        my, save_fig=True, show_colorbar=False, cmap_name="magma", save_path=myoutpath
    )
    # vis_depth_map(
    #     gt, save_fig=True, show_colorbar=False, cmap_name="magma_r", save_path=gtpath
    # )
    
# _dir = "/home/sph/event/da2-prompt-tuning/da2_metric_depth/results/da2_sigloss_mvsec_night1/npy"
# files = os.listdir(_dir)

# for f in files:
#     old_p = os.path.join(_dir, f)
#     new_p = old_p.replace("outdoor_night1_", "")
#     new_f = f.replace("_", "")
#     os.rename(old_p, new_p)
