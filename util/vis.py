import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from PIL import Image
import matplotlib
import cv2
import torch

def token2feature(tokens, patch_grid_size):
    """add token transfer to feature"""
    B, L, D = tokens.shape
    # H = W = int(L**0.5)
    H, W = patch_grid_size[0], patch_grid_size[1]
    x = tokens.permute(0, 2, 1).view(B, D, H, W).contiguous()
    return x


def feature2token(x):
    B, C, H, W = x.shape
    L = W * H
    tokens = x.view(B, C, L).permute(0, 2, 1).contiguous()
    return tokens


def merge_channel(vox):
    vox = vox - vox.min()
    new_voxel = vox[0]
    for i in range(1, vox.shape[0]):
        new_voxel = new_voxel + vox[i]
    return new_voxel


def vis_1_channel(arr, cmap_name="Spectral"):
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255.0
    arr = arr.astype(np.uint8)

    plt.imshow(arr, cmap=cmap)
    # Turn off axis for better visualization
    plt.axis("off")
    plt.show()


def event_npz2npy(npz_data):
    # Convert original data to [N, 4] (t, x, y, p)
    x = npz_data["x"].astype(int)
    y = npz_data["y"].astype(int)
    p = npz_data["p"].astype(int)
    t = npz_data["t"]

    return np.vstack((t, x, y, p)).T


def events_to_voxel_grid(events, num_bins, width, height):
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.

    :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
    :param num_bins: number of bins in the temporal axis of the voxel grid
    :param width, height: dimensions of the voxel grid
    """

    assert events.shape[1] == 4
    assert num_bins > 0
    assert width > 0
    assert height > 0

    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()

    # normalize the event timestamps so that they lie between 0 and num_bins
    last_stamp = events[-1, 0]
    first_stamp = events[0, 0]
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0
    # print("events 1", events[:,0])
    events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT
    ts = events[:, 0]
    xs = events[:, 1].astype(int)
    ys = events[:, 2].astype(int)
    pols = events[:, 3].astype(np.int32)
    pols[pols == 0] = -1  # polarity should be +1 / -1

    tis = ts.astype(int)
    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    valid_indices = tis < num_bins
    np.add.at(
        voxel_grid,
        xs[valid_indices]
        + ys[valid_indices] * width
        + tis[valid_indices] * width * height,
        vals_left[valid_indices],
    )

    valid_indices = (tis + 1) < num_bins
    np.add.at(
        voxel_grid,
        xs[valid_indices]
        + ys[valid_indices] * width
        + (tis[valid_indices] + 1) * width * height,
        vals_right[valid_indices],
    )

    voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))

    return voxel_grid


def compare_depth_gt(
    dep,
    gt,
    min_depth=0,
    max_depth=1000,
    show_colorbar=False,
    set_title=False,
    cmap_name="Spectral",
    pad=10,
    color_bar_width=15,
    dpi=100,
):
    cmap = matplotlib.colormaps.get_cmap(cmap_name)

    # Mask invalid values in the predicted depth
    mask = ~np.isfinite(gt)
    dep[mask] = np.nan

    # Clip depth values to the specified range
    gt = np.clip(gt, min_depth, max_depth)
    dep = np.clip(dep, min_depth, max_depth)
    vmin = min(gt[~mask].min(), dep[~mask].min())
    vmax = max(gt[~mask].max(), dep[~mask].max())
    print(vmin, vmax)
    h, w = dep.shape
    # Compute figure size: Maps tightly fill the space, extra for titles and colorbar
    if show_colorbar:
        # 2 maps + small gap + color bar
        width = 2 * w + 2 * pad + color_bar_width
    else:
        # 2 maps + small gap
        width = 2 * w + pad

    if set_title:
        # Map height + space for titles
        height = h + pad
    else:
        height = h
    figure_width = width / dpi  # 2 maps + small gap
    figure_height = height / dpi  # Map height + space for titles

    fig = plt.figure(figsize=(figure_width, figure_height), dpi=dpi)

    # Predicted Depth Map
    ax1 = fig.add_axes([0, 0, w / width, h / height])
    im1 = ax1.imshow(dep, cmap=cmap, vmin=vmin, vmax=vmax)
    ax1.axis("off")

    # Ground Truth Map
    ax2 = fig.add_axes([(w + pad) / width, 0, w / width, h / height])
    im2 = ax2.imshow(gt, cmap=cmap, vmin=vmin, vmax=vmax)
    ax2.axis("off")

    # Shared Colorbar
    if show_colorbar:
        colorbar_ax = fig.add_axes(
            [(2 * w + 2 * pad) / width, 0, color_bar_width / width, h / height]
        )  # Adjust height to align with maps
        cbar = fig.colorbar(im1, cax=colorbar_ax, orientation="vertical")
        cbar.set_label("Depth", fontsize=10)

    if set_title:
        ax2.set_title("GT Depth", fontsize=10)
        ax1.set_title("Predicted Depth", fontsize=10)
    plt.show()


def visualize_scene_from_paths(
    img_path,
    vox_path,
    dep_path,
    show_colorbar=False,
    set_title=False,
    cmap_name="Spectral",
    pad=10,
    color_bar_width=15,
    dpi=100,
):
    """
    Visualizes an image, voxel grid, and depth map from file paths.
    """
    cmap = matplotlib.colormaps.get_cmap(cmap_name)

    img = np.array(Image.open(img_path))
    vox = np.load(vox_path)
    dep = np.load(dep_path)

    h, w = dep.shape
    dpi = 100

    # Compute figure size: Maps tightly fill the space, extra for titles and colorbar
    if show_colorbar:
        # 2 maps + small gap + color bar
        width = 3 * w + 3 * pad + color_bar_width
    else:
        # 3 maps + 2 small gaps
        width = 3 * w + 2 * pad

    if set_title:
        # Map height + space for titles
        height = h + pad
    else:
        height = h
    figure_width = width / dpi  # 2 maps + small gap
    figure_height = height / dpi  # Map height + space for titles

    # Plotting
    fig = plt.figure(figsize=(figure_width, figure_height), dpi=dpi)

    # 1. Display the image
    ax1 = fig.add_axes([0, 0, w / width, h / height])
    if img.ndim == 3:  # Color image
        ax1.imshow(img)
    else:  # Grayscale image
        ax1.imshow(img, cmap="gray")
    ax1.axis("off")

    # 2. Display the voxel grid
    vox = merge_channel(vox)
    vox = (vox - vox.min()) / (vox.max() - vox.min()) * 255.0
    vox = vox.astype(np.uint8)

    ax2 = fig.add_axes([(w + pad) / width, 0, w / width, h / height])
    ax2.imshow(vox, cmap="gray")
    ax2.axis("off")

    # 3. Display the depth map
    ax3 = fig.add_axes([(2 * w + 2 * pad) / width, 0, w / width, h / height])
    im3 = ax3.imshow(dep, cmap=cmap)
    ax3.axis("off")
    if show_colorbar:
        colorbar_ax = fig.add_axes(
            [(3 * w + 3 * pad) / width, 0, color_bar_width / width, h / height]
        )  # Adjust height to align with maps
        cbar = fig.colorbar(im3, cax=colorbar_ax, orientation="vertical")
        cbar.set_label("Depth", fontsize=10)

    if set_title:
        ax1.set_title("Image")
        ax2.set_title("Voxel")
        ax3.set_title("Ground Truth")

    # plt.tight_layout()
    plt.show()


def plot_spatial_distribution(
    events, height, width, save_fig=False, save_path="test.png"
):
    # The coordinate origin is shifted to the upper left corner
    events[:, 2] = 0 - events[:, 2]
    positive_events = events[events[:, 3] == 1]
    negative_events = events[events[:, 3] == -1]

    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    print(positive_events.shape, negative_events.shape)
    plt.scatter(
        positive_events[:, 1],
        positive_events[:, 2],
        c="red",
        s=1,
        label="Polarity: 1",
        alpha=0.5,
    )
    plt.scatter(
        negative_events[:, 1],
        negative_events[:, 2],
        c="blue",
        s=1,
        label="Polarity: -1",
        alpha=0.5,
    )

    plt.axis("off")

    dpi = fig.get_dpi()
    fig.set_size_inches(width / dpi, height / dpi)
    if save_fig:
        plt.savefig(save_path)
    plt.show()


def plot_3d_events(events):
    # The coordinate origin is shifted to the upper left corner
    events[:, 2] = 0 - events[:, 2]
    events[:, 0] = events[:, 0] - events[0][0]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        events[:, 0],
        events[:, 1],
        events[:, 2],
        c=events[:, 3],
        s=1,
        cmap="bwr",
        alpha=0.7,
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("x")
    ax.set_zlabel("y")
    plt.title("3D Event Visualization")
    plt.show()


def detect_sparsity(events, width=346, height=260):
    corrds = events[:, 1:3]
    unique_coords, counts = np.unique(corrds, axis=0, return_counts=True)
    number_of_unique_coords = len(unique_coords)
    total_positions = width * height
    rate = number_of_unique_coords / total_positions * 100
    print(f"(width x height): {width}x{height} = {total_positions}")
    print(f"Number of coordinates with signals: {number_of_unique_coords}")
    print(f"Proportion: {rate:.2f}%")


def normalize_voxelgrid(event_tensor):
    mask = np.nonzero(event_tensor)
    if mask[0].size > 0:
        mean, stddev = event_tensor[mask].mean(), event_tensor[mask].std()
        print(mean, stddev)
        if stddev > 0:
            event_tensor[mask] = (event_tensor[mask] - mean) / stddev
    return event_tensor


def vis_voxelgrid(voxel):
    new_voxel = merge_channel(voxel)
    vis_1_channel(new_voxel, "gray")


def vis_depth_map(
    dep, show_colorbar=False, cmap_name="Spectral", save_fig=False, save_path="test.png"
):
    cmap = matplotlib.colormaps.get_cmap(cmap_name)

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
    plt.show()


def vis_diff(dep, gt):
    diff = gt - dep
    diff = 0 - abs(diff)
    vis_depth_map(diff)


def vis_abs_diff(dep, gt, eps=1e-6, show_colorbar=False):
    diff = abs(gt - dep)
    diff = diff / (gt + eps)
    diff = 0 - abs(diff)
    vis_depth_map(diff, show_colorbar=show_colorbar)


def vis_depth_map_filter(dep, gt, show_colorbar=False, cmap_name="Spectral"):
    cmap = matplotlib.colormaps.get_cmap(cmap_name)

    mask = ~np.isfinite(gt)
    dep[mask] = np.nan

    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.axis("off")

    im = ax.imshow(dep, cmap=cmap, extent=[0, dep.shape[1], 0, dep.shape[0]])
    if show_colorbar:
        plt.colorbar(im)

    dpi = fig.get_dpi()
    fig.set_size_inches(dep.shape[1] / dpi, dep.shape[0] / dpi)

    # plt.savefig("test.png")
    plt.show()


def visualize_scene_from_paths_event(
    img_path,
    eve_path,
    dep_path,
    show_colorbar=False,
    set_title=False,
    cmap_name="Spectral",
    pad=10,
    color_bar_width=15,
    dpi=100,
):
    """
    Visualizes an image, voxel grid, and depth map from file paths.
    """
    cmap = matplotlib.colormaps.get_cmap(cmap_name)

    img = np.array(Image.open(img_path))
    eve = np.load(eve_path)
    if eve_path.endswith(".npz"):
        eve = event_npz2npy(eve)
        eve[eve[:, 3] == 0, 3] = -1
    dep = np.load(dep_path)

    h, w = dep.shape
    dpi = 100

    # Compute figure size: Maps tightly fill the space, extra for titles and colorbar
    if show_colorbar:
        # 2 maps + small gap + color bar
        width = 3 * w + 3 * pad + color_bar_width
    else:
        # 3 maps + 2 small gaps
        width = 3 * w + 2 * pad

    if set_title:
        # Map height + space for titles
        height = h + pad
    else:
        height = h
    figure_width = width / dpi  # 2 maps + small gap
    figure_height = height / dpi  # Map height + space for titles

    # Plotting
    fig = plt.figure(figsize=(figure_width, figure_height), dpi=dpi)

    # 1. Display the image
    ax1 = fig.add_axes([0, 0, w / width, h / height])
    if img.ndim == 3:  # Color image
        ax1.imshow(img)
    else:  # Grayscale image
        ax1.imshow(img, cmap="gray")
    ax1.axis("off")

    # 2. Display the event stream
    eve[:, 2] = 0 - eve[:, 2]

    positive_events = eve[eve[:, 3] == 1]
    negative_events = eve[eve[:, 3] == -1]

    ax2 = fig.add_axes([(w + pad) / width, 0, w / width, h / height])
    ax2.scatter(
        positive_events[:, 1],
        positive_events[:, 2],
        c="red",
        s=1,
        label="Polarity: 1",
        alpha=0.5,
    )
    ax2.scatter(
        negative_events[:, 1],
        negative_events[:, 2],
        c="blue",
        s=1,
        label="Polarity: -1",
        alpha=0.5,
    )
    # ax2.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize='x-small')
    ax2.axis("off")

    # 3. Display the depth map
    ax3 = fig.add_axes([(2 * w + 2 * pad) / width, 0, w / width, h / height])
    im3 = ax3.imshow(dep, cmap=cmap)
    ax3.axis("off")
    if show_colorbar:
        colorbar_ax = fig.add_axes(
            [(3 * w + 3 * pad) / width, 0, color_bar_width / width, h / height]
        )  # Adjust height to align with maps
        cbar = fig.colorbar(im3, cax=colorbar_ax, orientation="vertical")
        cbar.set_label("Depth", fontsize=10)

    if set_title:
        ax1.set_title("Image")
        ax2.set_title("Events")
        ax3.set_title("Ground Truth")

    # plt.tight_layout()
    plt.show()


def overlay_depth_on_image(
    image_path,
    depth_map,
    alpha=0.6,
    cmap_name="Spectral",
    save_fig=False,
    save_path="overlay.png",
):
    """
    Overlays a depth map on an image with a colormap.

    Parameters:
        image_path (str): Path to the image file.
        depth_map (np.ndarray): Depth map (2D numpy array).
        alpha (float): Transparency level for the overlay (0=fully transparent, 1=fully opaque).
        cmap_name (str): Name of the colormap to apply to the depth map.
        save_fig (bool): Whether to save the result.
        save_path (str): Path to save the result.
    """
    # Load the image
    image = np.array(Image.open(image_path))

    # Ensure the image is in RGB format
    if image.ndim == 2:  # Grayscale image
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA image
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    # Normalize the depth map to [0, 1] for colormap mapping
    depth_map = (depth_map - np.min(depth_map)) / (
        np.max(depth_map) - np.min(depth_map)
    )

    # Apply colormap to the depth map
    cmap = plt.get_cmap(cmap_name)
    depth_colored = (cmap(depth_map)[:, :, :3] * 255).astype(
        np.uint8
    )  # Remove alpha channel and scale to [0, 255]

    # Resize depth_colored to match the image dimensions if necessary
    if depth_colored.shape[:2] != image.shape[:2]:
        depth_colored = cv2.resize(
            depth_colored,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    # Blend the depth map with the image
    overlay = cv2.addWeighted(image, 1 - alpha, depth_colored, alpha, 0)

    # Display the result
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.imshow(overlay)

    if save_fig:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.show()


def vis_feature_var(tokens: torch.tensor, patch_size):
    if tokens.dim() == 2:
        tokens = tokens.unsqueeze(0)
    
    feas = token2feature(tokens, patch_size)
    var = torch.var(feas, dim=1, keepdim=True).squeeze().numpy()
    vis_depth_map(var, cmap_name="viridis")
