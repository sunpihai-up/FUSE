import argparse
import os
import numpy as np
from tqdm import tqdm

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
    pols = events[:, 3]
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


def event_npz2npy(npz_data):
    # Convert original data to [N, 4] (t, x, y, p)
    x = npz_data["x"].astype(int)
    y = npz_data["y"].astype(int)
    p = npz_data["p"].astype(int)
    t = npz_data["t"]

    return np.vstack((t, x, y, p)).T


def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--new-numbins", type=int)
    parser.add_argument("--split-dir", type=str)
    parser.add_argument("--dataset", type=str, choices=["mvsec", "eventscape"])
    
    return parser.parse_args()

if __name__ == "__main__":
    # split_names = ["train.txt", "val.txt", "test.txt"]
    split_names = ["test.txt"]
    args = parse_arguments()
    
    voxel_paths = []
    event_paths = []
    
    for txt_name in split_names:
        txt_path = os.path.join(args.split_dir, txt_name)
        with open(txt_path, "r") as f:
            lines = f.readlines()
        # Get all voxel paths
        lines = [line.split(' ')[2].strip() for line in lines]
        voxel_paths.extend(lines)
    
    if args.dataset == "mvsec":
        event_paths = [p.replace("voxels", "events") for p in voxel_paths]
    elif args.dataset == "eventscape":
        event_paths = [p.replace("voxels", "data") for p in voxel_paths]
        event_paths = [p.replace("npy", "npz") for p in event_paths]
    
    v = np.load(voxel_paths[0])
    b, h, w = v.shape
    # h, w = 256, 512
    
    for i in tqdm(range(len(event_paths)), total=len(event_paths)):
        event = np.load(event_paths[i])
        
        if args.dataset == "eventscape":
            event = event_npz2npy(event)
            event[event[:, 3] == 0, 3] = -1
        voxel = events_to_voxel_grid(event, args.new_numbins, w, h)
        
        dir = os.path.dirname(voxel_paths[i])
        os.makedirs(dir, exist_ok=True)
        np.save(voxel_paths[i], voxel)
        
'''
python scripts/change_bins.py \
    --new-numbins 3 \
    --split-dir ./dataset/splits/mvsec \
    --dataset mvsec
'''