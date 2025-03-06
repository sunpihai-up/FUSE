import zipfile
import os
import numpy as np
import argparse
import shutil
from tqdm import tqdm


def delete_redundant_files(data_root):
    towns = os.listdir(data_root)

    for town in towns:
        town_path = os.path.join(data_root, town)
        sequences = os.listdir(town_path)
        for seq in sequences:
            seq_path = os.path.join(town_path, seq)
            
            depth_frames = os.path.join(seq_path, 'depth', 'frames')
            event_frames = os.path.join(seq_path, 'events', 'frames')
            semantic = os.path.join(seq_path, 'semantic')
            vehicles = os.path.join(seq_path, 'vehicle_data')

            if os.path.isdir(depth_frames):
                shutil.rmtree(depth_frames)
                
            if os.path.isdir(event_frames):
                shutil.rmtree(event_frames)
            
            if os.path.isdir(semantic):
                shutil.rmtree(semantic)
            
            if os.path.isdir(vehicles):
                shutil.rmtree(vehicles)
                

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


def process_events_dir(events_dir, voxels_dir, num_bins=3, width=512, height=256):
    os.makedirs(voxels_dir, exist_ok=True)
    events = os.listdir(events_dir)

    seq_name = events_dir.split("/")[-4] + "/" + events_dir.split("/")[-3]
    for event in tqdm(events, desc=f"Processing events in {seq_name}", unit="event"):
        src_path = os.path.join(events_dir, event)
        dst_path = os.path.join(voxels_dir, event.replace("npz", "npy"))

        if event.endswith("txt"):
            # Copy timestamps.txt and boundary_timestamps.txt
            shutil.copy(src_path, dst_path)
            continue

        # Load original data
        npz_data = np.load(src_path)
        # Convert original data to [N, 4] (t, x, y, p)
        x = npz_data["x"].astype(int)
        y = npz_data["y"].astype(int)
        p = npz_data["p"].astype(int)
        t = npz_data["t"]

        event_arr = np.vstack((t, x, y, p)).T
        # Ensure polarity is 1 or -1
        event_arr[event_arr[:, 3] == 0, 3] = -1
        voxel = events_to_voxel_grid(event_arr, num_bins, width, height)
        np.save(dst_path, voxel)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process script parameters.")
    parser.add_argument(
        "data_root", type=str, help="Path to the root directory of input data."
    )
    parser.add_argument(
        "output_root", type=str, help="Path to the root directory for output data."
    )
    parser.add_argument(
        "--numbins",
        type=int,
        default=3,
        help="Number of bins for voxel grid. Default is 3.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    zip_name = {
        "train": "Town01-03_train.zip",
        "test": "Town05_test.zip",
        "val": "Town05_val.zip",
    }
    width, height = 512, 256
    split_dir = "./dataset/splits/eventscape/"

    args = parse_arguments()
    for mode, name in zip_name.items():
        print(f"##############  Processing {name} ##############")
        images_path = []
        depths_path = []
        voxels_path = []
        # events_path = []

        zip_path = os.path.join(args.data_root, name)
        output_path = os.path.join(args.output_root, name.split(".")[0])

        print(f"Unzip {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(output_path)

        towns = os.listdir(output_path)
        for town in towns:
            town_dir = os.path.join(output_path, town)
            seqs = os.listdir(town_dir)
            for seq in seqs:
                seq_path = os.path.join(town_dir, seq)
                images_dir = os.path.join(seq_path, "rgb", "data")
                depths_dir = os.path.join(seq_path, "depth", "data")
                events_dir = os.path.join(seq_path, "events", "data")
                voxels_dir = os.path.join(seq_path, "events", "voxels")

                # Convert all events in events_dir to voxels and save them to voxels_dir
                process_events_dir(events_dir, voxels_dir, args.numbins, width, height)

                # Add file paths to list
                images = os.listdir(images_dir)
                depths = os.listdir(depths_dir)
                voxels = os.listdir(voxels_dir)
                # events = os.listdir(events_dir)

                images = [os.path.join(images_dir, p) for p in images if not p.endswith(".txt")]
                depths = [os.path.join(depths_dir, p) for p in depths if not p.endswith(".txt")]
                voxels = [os.path.join(voxels_dir, p) for p in voxels if not p.endswith(".txt")]
                # events = [os.path.join(voxels_dir, p) for p in events if not p.endswith(".txt")]

                images_path.extend(images)
                depths_path.extend(depths)
                voxels_path.extend(voxels)
                # events_path.extend(events)

        images_path = sorted(images_path)
        depths_path = sorted(depths_path)
        voxels_path = sorted(voxels_path)
        # events_path = sorted(events_path)
        # voxels_path = [p.replace("data", "voxels") for p in events_path]
        # voxels_path = [p.replace("npz", "npy") for p in events_path]

        lines = []
        for i in range(len(images_path)):
            line = f"{images_path[i]} {depths_path[i]} {voxels_path[i]}" + "\n"
            lines.append(line)

        os.makedirs(split_dir, exist_ok=True)
        split_path = split_dir + mode + ".txt"
        with open(split_path, "w") as f:
            f.writelines(lines)

        # delete_redundant_files(output_path)
        print(f"##############  Finished Processing {name} ##############")

"""
python scripts/process_eventscape.py \
    /data_nvme/sph/EventScape \
    /data_nvme/sph/EventScape_processed \
    --numbins 3
"""
