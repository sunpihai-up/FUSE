from tqdm import tqdm

import json
import numpy as np
from PIL import Image

import h5py
import os

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
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create mappings
    depth_to_image = {}

    image_idx = 0
    # Compute nearest previous image for each depth
    for depth_idx, depth_time in enumerate(depth_ts):
        # nearest_image_idx = None
        # for idx, ts in enumerate(image_ts):
        #     if ts > depth_time:
        #         break
        #     nearest_image_idx = idx
        # nearest_image_idx = nearest_image_idx if nearest_image_idx is not None else len(image_ts) - 1
        
        # Compute nearest image for each depth
        while image_idx < len(image_ts) - 1 and abs(image_ts[image_idx + 1] - depth_time) < abs(image_ts[image_idx] - depth_time):
            image_idx += 1
        nearest_image_idx = image_idx
        
        item = {}
        item['nearest_img_idx'] = nearest_image_idx
        item['depth_ts'] = depth_time
        item['img_ts'] = image_ts[nearest_image_idx]
        item['diff'] = abs(item['depth_ts'] - item['img_ts'])
        
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
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize result dictionary
    depth_to_events_mapping = {}

    # Start index for events
    current_event_index = 0

    # Iterate through each depth timestamp
    for i, depth_time in enumerate(depth_ts):
        # Move the start index forward to skip past events older than 50ms before the depth timestamp
        while current_event_index < len(events) and events[current_event_index, 2] < depth_time - 0.05:
            current_event_index += 1

        # Find the closest event index before the depth timestamp
        end_event_index = current_event_index
        while end_event_index < len(events) and events[end_event_index, 2] <= depth_time:
            end_event_index += 1

        if current_event_index < len(events) and end_event_index > current_event_index:
            # Save the mapping
            depth_to_events_mapping[i] = {
                "depth_timestamp": float(depth_time),
                "start_event_index": int(current_event_index),
                "end_event_index": int(end_event_index - 1),
                "start_event_timestamp": float(events[current_event_index, 2]),
                "end_event_timestamp": float(events[end_event_index - 1, 2])
            }

    # Save the mapping to a JSON file
    output_file = os.path.join(output_dir, "depth_to_events_mapping.json")
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

    assert(events.shape[1] == 4)
    assert(num_bins > 0)
    assert(width > 0)
    assert(height > 0)

    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()

    # normalize the event timestamps so that they lie between 0 and num_bins
    last_stamp = events[-1, 0]
    first_stamp = events[0, 0]
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0
    #print("events 1", events[:,0])
    events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT
    ts = events[:, 0]
    xs = events[:, 1].astype(np.int)
    ys = events[:, 2].astype(np.int)
    pols = events[:, 3]
    pols[pols == 0] = -1  # polarity should be +1 / -1

    tis = ts.astype(np.int)
    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    valid_indices = tis < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width +
              tis[valid_indices] * width * height, vals_left[valid_indices])

    valid_indices = (tis + 1) < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width +
              (tis[valid_indices] + 1) * width * height, vals_right[valid_indices])

    voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))

    return voxel_grid

def save_depths2dir(depths, selected_depths, output_dir):
    for dpt_id in selected_depths:
        dpt = depths[dpt_id]
        np.save(output_dir + '/%05d.npy' % dpt_id, dpt)

def save_images2dir(images, selected_depths, depth2image, output_dir):    
    for dpt_id in selected_depths:
        img_id = depth2image[dpt_id]["nearest_img_idx"]

        img = images[img_id]
        img = Image.fromarray(img)
        img.save(output_dir + '/%05d.png' % dpt_id)


def save_voxels2dir(
    events, selected_depths, depth2event, output_dir, num_bins=5, width=346, height=260
):
    for dpt_id in selected_depths:
        st_id = depth2event[dpt_id]["start_event_index"]
        en_id = depth2event[dpt_id]["end_event_index"]

        event_slice = events[st_id : en_id + 1]
        event_slice = event_slice[:, [2, 0, 1, 3]] # (x, y, t, p) --> (t, x, y, p)

        voxels = events_to_voxel_grid(event_slice, num_bins, width, height)
        np.save(output_dir + '/%05d.npy' % dpt_id, voxels)
        
if 