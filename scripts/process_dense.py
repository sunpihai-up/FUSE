import os
import numpy as np

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


def covert_all_events2voxel(rdir, numbins, width, height):
    types = ["test", "train", "val"]
    for t in types:
        type_dir = os.path.join(rdir, t)
        seqs = os.listdir(type_dir)
        
        for seq in seqs:
            seq_dir = os.path.join(type_dir, seq)
            event_dir = os.path.join(seq_dir, "events/data")
            
            # Create vox dir
            vox_dir = os.path.join(seq_dir, "voxels")
            os.makedirs(vox_dir, exist_ok=True)
            
            eve_paths = os.listdir(event_dir)
            for p in eve_paths:
                if p.endswith('txt'):
                    continue
                
                eve_p = os.path.join(event_dir, p)
                eve  = np.load(eve_p)
                vox = events_to_voxel_grid(eve, num_bins=numbins, width=width, height=height)
                
                vox_path = os.path.join(vox_dir, p.replace("events", "voxel"))
                np.save(vox_path, vox)
            
            print(f"{t}-{seq} is OK! ")


def generate_split_file(rdir):
    types = ["test", "train", "val"]
    for t in types:
        imgs, deps, voxs = [], [], []
        type_dir = os.path.join(rdir, t)

        seqs = os.listdir(type_dir)

        for seq in seqs:
            seq_dir = os.path.join(type_dir, seq)
            img_dir = os.path.join(seq_dir, "rgb/frames")
            vox_dir = os.path.join(seq_dir, "voxels")
            dep_dir = os.path.join(seq_dir, "depth/data")

            img_ps = os.listdir(img_dir)
            vox_ps = os.listdir(vox_dir)
            dep_ps = os.listdir(dep_dir)

            img_ps = [os.path.join(img_dir, p) for p in img_ps if not p.endswith(".txt")]
            vox_ps = [os.path.join(vox_dir, p) for p in vox_ps if not p.endswith(".txt")]
            dep_ps = [os.path.join(dep_dir, p) for p in dep_ps if not p.endswith(".txt")]

            imgs.extend(img_ps)
            deps.extend(dep_ps)
            voxs.extend(vox_ps)

        imgs = sorted(imgs)
        deps = sorted(deps)
        voxs = sorted(voxs)

        assert len(imgs) == len(deps)
        assert len(imgs) == len(voxs)

        lines = []
        for i in range(len(imgs)):
            line = f"{imgs[i]} {deps[i]} {voxs[i]}" + "\n"
            lines.append(line)

        with open(f"{t}.txt", "w") as f:
            f.writelines(lines)
            
if __name__ == "__main__":
    numbins = 3
    rdir = "/data_nvme/sph/DENSE"
    width, height = 346, 260

    # covert_all_events2voxel(rdir, numbins, width, height)
    generate_split_file(rdir)
    