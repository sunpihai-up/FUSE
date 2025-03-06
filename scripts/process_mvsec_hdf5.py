import os
import json
import random
import argparse
from tqdm import tqdm
import numpy as np
from PIL import Image
import h5py


def map_depth_to_image(image_ts, depth_ts, output_dir):
    """
    Map depth timestamps to the nearest previous image timestamps.

    Args:
        image_ts (numpy.ndarray): timestamps of each image.
        depth_ts (numpy.ndarray): timestamps of each depth.
        output_dir (str): Directory to save the output JSON files.

    Output:
        Saves two JSON files:
        - depth_to_image.json: Mapping from depth indices to nearest image indices.
    """
    print("Mapping depth to image timestamps...")
    # Create mappings
    depth_to_image = {}

    image_idx = 0
    # Compute nearest previous image for each depth
    for depth_idx, depth_time in tqdm(enumerate(depth_ts), total=len(depth_ts)):
        # nearest_image_idx = None
        # for idx, ts in enumerate(image_ts):
        #     if ts > depth_time:
        #         break
        #     nearest_image_idx = idx
        # nearest_image_idx = nearest_image_idx if nearest_image_idx is not None else len(image_ts) - 1

        # Compute nearest image for each depth
        while image_idx < len(image_ts) - 1 and abs(
            image_ts[image_idx + 1] - depth_time
        ) < abs(image_ts[image_idx] - depth_time):
            image_idx += 1
        nearest_image_idx = image_idx

        item = {
            "nearest_img_idx": nearest_image_idx,
            "depth_ts": depth_time,
            "img_ts": image_ts[nearest_image_idx],
        }
        item["diff"] = abs(item["depth_ts"] - item["img_ts"])

        depth_to_image[depth_idx] = item

    # Save mappings to JSON files
    with open(f"{output_dir}/depth_to_image.json", "w") as f:
        json.dump(depth_to_image, f, indent=4)

    print(f"Mapping saved to {output_dir}/depth_to_image.json")
    return depth_to_image


def map_depth_to_events(depth_ts, events, output_dir):
    """
    Maps each depth map timestamp to a range of events within a specific time window.

    Parameters:
        depth_ts (numpy.ndarray): timestamps of each depth map.
        events (numpy.ndarray): each row represents an event as (x, y, timestamp, p).
        output_dir (str): Directory to save the output JSON file.

    Output:
        A JSON file saved to output_dir containing the correspondence between depth maps and events.
    """
    print("Mapping depth to event timestamps...")

    # Initialize result dictionary
    depth_to_events_mapping = {}

    # Start index for events
    current_event_index = 0

    # Iterate through each depth timestamp
    for i, depth_time in tqdm(enumerate(depth_ts), total=len(depth_ts)):
        # Move the start index forward to skip past events older than 50ms before the depth timestamp
        while (
            current_event_index < len(events)
            and events[current_event_index, 2] < depth_time - 0.05
        ):
            current_event_index += 1

        # Find the closest event index before the depth timestamp
        end_event_index = current_event_index
        while (
            end_event_index < len(events) and events[end_event_index, 2] <= depth_time
        ):
            end_event_index += 1

        if current_event_index < len(events) and end_event_index > current_event_index:
            # Save the mapping
            depth_to_events_mapping[i] = {
                "depth_timestamp": float(depth_time),
                "start_event_index": int(current_event_index),
                "end_event_index": int(end_event_index - 1),
                "start_event_timestamp": float(events[current_event_index, 2]),
                "end_event_timestamp": float(events[end_event_index - 1, 2]),
            }

    # Save the mapping to a JSON file
    output_file = os.path.join(output_dir, "depth_to_events.json")
    with open(output_file, "w") as f:
        json.dump(depth_to_events_mapping, f, indent=4)

    print(f"Mapping saved to {output_file}")
    return depth_to_events_mapping


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


def save_depths2dir(depths, selected_depths, output_dir):
    print("Saving depths to directory:", output_dir)
    for dpt_id in tqdm(selected_depths, desc="Saving depths"):
        dpt = depths[dpt_id]
        np.save(output_dir + "/%05d.npy" % dpt_id, dpt)
    print(f"Saved depths to {output_dir}")


def save_images2dir(images, selected_depths, depth2image, output_dir):
    print("Saving images to directory:", output_dir)
    for dpt_id in tqdm(selected_depths, desc="Saving images"):
        img_id = depth2image[dpt_id]["nearest_img_idx"]

        img = images[img_id]
        img = Image.fromarray(img)
        img.save(output_dir + "/%05d.png" % dpt_id)
    print(f"Saved images to {output_dir}")


def save_voxels2dir(
    events, selected_depths, depth2event, output_dir, num_bins=5, width=346, height=260
):
    print("Saving voxels to directory:", output_dir)
    for dpt_id in tqdm(selected_depths, desc="Saving voxels"):
        st_id = depth2event[dpt_id]["start_event_index"]
        en_id = depth2event[dpt_id]["end_event_index"]

        event_slice = events[st_id : en_id + 1]
        event_slice = event_slice[:, [2, 0, 1, 3]]  # (x, y, t, p) --> (t, x, y, p)

        voxels = events_to_voxel_grid(event_slice, num_bins, width, height)
        np.save(output_dir + "/%05d.npy" % dpt_id, voxels)
    print(f"Saved voxels to {output_dir}")


def save_events2dir(events, selected_depths, depth2event, output_dir):
    print("Saving events to directory:", output_dir)
    for dpt_id in tqdm(selected_depths, desc="Saving events"):
        st_id = depth2event[dpt_id]["start_event_index"]
        en_id = depth2event[dpt_id]["end_event_index"]

        event_slice = events[st_id : en_id + 1]
        event_slice = event_slice[:, [2, 0, 1, 3]]  # (x, y, t, p) --> (t, x, y, p)

        np.save(output_dir + "/%05d.npy" % dpt_id, event_slice)
    print(f"Saved events to {output_dir}")


def gen_split(scene, data_dir, output_dir="./dataset/splits/mvsec/"):
    depths_dir = os.path.join(data_dir, "depths")
    images_dir = os.path.join(data_dir, "images")
    voxels_dir = os.path.join(data_dir, "voxels")

    depths = os.listdir(depths_dir)
    images = os.listdir(images_dir)
    voxels = os.listdir(voxels_dir)

    depths.sort()
    images.sort()
    voxels.sort()

    assert len(depths) == len(images) and len(depths) == len(voxels)

    depths = [os.path.join(depths_dir, file_name) for file_name in depths]
    images = [os.path.join(images_dir, file_name) for file_name in images]
    voxels = [os.path.join(voxels_dir, file_name) for file_name in voxels]

    lines = []
    for i in range(len(depths)):
        line = f"{images[i]} {depths[i]} {voxels[i]}" + "\n"
        lines.append(line)

    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}{scene}.txt", "w") as f:
        f.writelines(lines)
    print(f"Saved split file to {output_dir}")


def main(data_hdf5_path, gt_hdf5_path, scene, output_dir, numbins, width, height):
    os.makedirs(output_dir, exist_ok=True)

    assert os.path.isfile(data_hdf5_path), print(
        f"Data file {data_hdf5_path} does not exist"
    )
    assert os.path.isfile(gt_hdf5_path), print(f"GT file {gt_hdf5_path} does not exist")

    data = h5py.File(data_hdf5_path)
    gt = h5py.File(gt_hdf5_path)

    print(
        "Extracting timestamps info (The sequence of events is so long and may take a long time)..."
    )
    image_ts = np.array(data["davis"]["left"]["image_raw_ts"])
    assert np.all(image_ts[:-1] <= image_ts[1:]), print(
        "Image timestamps are not in ascending order"
    )

    # depth_image_raw_ts = np.array(gt["davis"]["left"]["depth_image_raw_ts"])
    # assert np.all(depth_image_raw_ts[:-1] <= depth_image_raw_ts[1:]), print(
    #     "Depth timestamps are not in ascending order"
    # )
    depth_image_rect_ts = np.array(gt["davis"]["left"]["depth_image_rect_ts"])
    assert np.all(depth_image_rect_ts[:-1] <= depth_image_rect_ts[1:]), print(
        "Depth timestamps are not in ascending order"
    )

    events = np.array(data["davis"]["left"]["events"])  # x, y, t, p
    event_timestamps = events[:, 2]
    assert np.all(event_timestamps[:-1] <= event_timestamps[1:]), print(
        "Event timestamps are not in ascending order"
    )
    print("Timestamps extracted.")

    # depth2image = map_depth_to_image(image_ts, depth_image_raw_ts, output_dir)
    # depth2event = map_depth_to_events(depth_image_raw_ts, events, output_dir)
    depth2image = map_depth_to_image(image_ts, depth_image_rect_ts, output_dir)
    depth2event = map_depth_to_events(depth_image_rect_ts, events, output_dir)

    selected_depths = []

    for depth_id, item in depth2image.items():
        diff = item["diff"]
        if diff * 1000 > 10:  # 10ms
            print(f"Depth {depth_id}: No recent image, discard the GT")
            continue

        if depth_id not in depth2event:
            print(f"Depth {depth_id}: There are not enough events, discard the GT")
            continue

        selected_depths.append(depth_id)
    print(f"Length of selected samples: {len(selected_depths)}")

    depths_dir = os.path.join(output_dir, "depths")
    images_dir = os.path.join(output_dir, "images")
    events_dir = os.path.join(output_dir, "events")
    voxels_dir = os.path.join(output_dir, "voxels")

    os.makedirs(depths_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(events_dir, exist_ok=True)
    os.makedirs(voxels_dir, exist_ok=True)

    images = data["davis"]["left"]["image_raw"]
    # depths = gt['davis']['left']['depth_image_raw']
    depths = gt["davis"]["left"]["depth_image_rect"]

    save_depths2dir(depths, selected_depths, depths_dir)
    save_images2dir(images, selected_depths, depth2image, images_dir)
    save_events2dir(events, selected_depths, depth2event, events_dir)
    save_voxels2dir(
        events, selected_depths, depth2event, voxels_dir, numbins, width, height
    )

    gen_split(scene, output_dir)


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
        default=5,
        help="Number of bins for voxel grid. Default is 5.",
    )
    # parser.add_argument("--width", type=int, default=346, help="Width of the voxel grid.")
    # parser.add_argument("--height", type=int, default=260, help="Height of the voxel grid.")
    return parser.parse_args()


if __name__ == "__main__":
    process_list = [
        ("outdoor_day1_data.hdf5", "outdoor_day1_gt.hdf5"),
        ("outdoor_day2_data.hdf5", "outdoor_day2_gt.hdf5"),
        ("outdoor_night1_data.hdf5", "outdoor_night1_gt.hdf5"),
        ("outdoor_night2_data.hdf5", "outdoor_night2_gt.hdf5"),
        ("outdoor_night3_data.hdf5", "outdoor_night3_gt.hdf5"),
    ]
    width, height = 346, 260

    args = parse_arguments()

    for pair in process_list:
        # outdoor_day1, outdoor_day2 ...
        scene = pair[0].split("_")[0] + "_" + pair[0].split("_")[1]

        data_hdf5_path = os.path.join(args.data_root, pair[0])
        gt_hdf5_path = os.path.join(args.data_root, pair[1])
        output_dir = os.path.join(args.output_root, scene)

        print(f"##############  Processing {scene} ##############")
        main(
            data_hdf5_path,
            gt_hdf5_path,
            scene,
            output_dir,
            args.numbins,
            width,
            height,
        )
        print(f"##############  Finished Processing {scene} ##############")

# python process_mvsec.py F:\\MVSEC\\mvsec-hdf5 F:\\MVSEC\\mvsec-hdf5\\test --numbins 3 --width 346 --height 260
"""
python scripts/process_mvsec_hdf5.py \
    /data_nvme/sph/mvsec \
    /data_nvme/sph/mvsec_processed \
    --numbins 3
"""
