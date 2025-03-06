import argparse
import cv2
import matplotlib
import numpy as np
import os
import torch
import torch.nn.functional as F

# from model.FUSE_dis import FUSE
from model.FUSE import FUSE
from model.fuse.utils import clean_pretrained_weight

from dataset.mvsec import MVSEC
from dataset.eventscape import EventScape
from dataset.dense import Dense
from torch.utils.data import DataLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EPDE")

    parser.add_argument("--input-size", type=int)
    parser.add_argument("--encoder", type=str, choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument("--max-depth", type=float, default=20)
    parser.add_argument("--event-voxel-chans", type=int)
    parser.add_argument("--outdir", type=str)
    parser.add_argument("--dataset", choices=["mvsec", "eventscape", "dense"])
    parser.add_argument("--save-numpy", action="store_true")
    parser.add_argument("--pred-only", action="store_true")
    parser.add_argument("--grayscale", action="store_true")
    parser.add_argument("--normalized-depth", action="store_true")
    parser.add_argument("--return-feature", action="store_true")
    parser.add_argument("--load-from", type=str)
    parser.add_argument(
        "--scene",
        choices=["day1", "night1", "train", "test"],
    )

    args = parser.parse_args()

    size = (args.input_size, args.input_size)
    if args.dataset == "mvsec" and args.scene == "day1":
        valset = MVSEC(
            "dataset/splits/mvsec/outdoor_day1.txt",
            "val",
            normalized_d=args.normalized_depth,
            size=size,
        )
    elif args.dataset == "mvsec" and args.scene == "night1":
        valset = MVSEC(
            "dataset/splits/mvsec/outdoor_night1.txt",
            "val",
            normalized_d=args.normalized_depth,
            size=size,
        )
    elif args.dataset == "mvsec" and args.scene == "train":
        valset = MVSEC(
            "dataset/splits/mvsec/train.txt",
            "val",
            normalized_d=args.normalized_depth,
            size=size,
        )
    elif args.dataset == "eventscape" and args.scene == "test":
        valset = EventScape(
            "dataset/splits/eventscape/test.txt",
            "val",
            normalized_d=args.normalized_depth,
            size=size,
        )
    elif args.dataset == "dense" and args.scene == "test":
        valset = Dense(
            "dataset/splits/dense/test.txt",
            "val",
            normalized_d=args.normalized_depth,
            size=size,
        )
    else:
        raise NotImplementedError

    valloader = DataLoader(
        valset,
        batch_size=1,
        pin_memory=True,
        num_workers=4,
        drop_last=True,
    )

    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # Instantiate Model and Load Pretrained Weight
    model = FUSE(
        model_name=args.encoder,
        max_depth=args.max_depth,
        event_voxel_chans=args.event_voxel_chans,
        return_feature=args.return_feature,
    )

    model.eval()
    checkpoint = torch.load(args.load_from, map_location="cpu")
    checkpoint = clean_pretrained_weight(checkpoint)
    model.load_state_dict(checkpoint)
    print(f"Model weights load from {args.load_from} successfully!")
    model = model.to(DEVICE).eval()

    os.makedirs(args.outdir, exist_ok=True)
    npy_dir = os.path.join(args.outdir, "npy")
    os.makedirs(npy_dir, exist_ok=True)
    vis_dir = os.path.join(args.outdir, "vis")
    os.makedirs(vis_dir, exist_ok=True)

    cmap = matplotlib.colormaps.get_cmap("Spectral")

    for k, sample in enumerate(valloader):
        rgb_path = sample["image_path"][0]
        print(f"Progress {k+1}/{len(valloader)}: {rgb_path}")

        raw_image = cv2.imread(rgb_path)
        h, w = raw_image.shape[0], raw_image.shape[1]

        with torch.no_grad():
            inputs = sample["input"].to(DEVICE)
            depth = model(inputs)
            depth = F.interpolate(
                depth[:, None], (h, w), mode="bilinear", align_corners=True
            )[0, 0]

        depth = depth.cpu().numpy()

        # scene = rgb_path.split("/")[-3]
        # name = rgb_path.split("/")[-1]
        if args.save_numpy:
            output_path = os.path.join(
                npy_dir,
                os.path.splitext(os.path.basename(rgb_path))[0] + ".npy",
                # os.path.splitext(f"{scene}_{name}")[0] + ".npy",
            )
            np.save(output_path, depth)

        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)

        if args.grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

        output_path = os.path.join(
            # vis_dir, f"{scene}_{name}"
            vis_dir,
            os.path.splitext(os.path.basename(rgb_path))[0] + ".png",
            #     vis_dir, os.path.basename(rgb_path) + ".png"
        )
        if args.pred_only:
            cv2.imwrite(output_path, depth)
        else:
            split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
            combined_result = cv2.hconcat([raw_image, split_region, depth])

            cv2.imwrite(output_path, combined_result)
