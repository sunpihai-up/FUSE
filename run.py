import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch

# from depth_anything_v2.dpt import DepthAnythingV2
from model.epde.epde import EPDE
from util.metric import convert_nl2abs_depth, dataset2params


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2 Metric Depth Estimation')
    parser.add_argument('--split_path', default=None, type=str)
    parser.add_argument('--img_dir', default=None, type=str)
    parser.add_argument('--event_dir', default=None, type=str)

    parser.add_argument("--dataset", default="dense", choices=["dense", "mvsec", "eventscape"])
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./vis_depth')

    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--load-from', type=str, default='checkpoints/depth_anything_v2_metric_hypersim_vitl.pth')

    parser.add_argument('--save-numpy', dest='save_numpy', action='store_true', help='save the model raw output')
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')

    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    # Instantiate Model and Load Pretrained Weight
    model = EPDE(model_name=args.encoder, dataset=args.dataset)

    checkpoint = torch.load(args.load_from, map_location='cpu')
    if 'model' in checkpoint.keys():
        checkpoint = checkpoint['model']
        checkpoint = {
            (key[7:] if key.startswith("module.") else key): value
            for key, value in checkpoint.items()
        }
    model.load_state_dict(checkpoint)
    print(f"Model weights load from {args.load_from} successfully!")

    model = model.to(DEVICE).eval()

    if args.split_path is not None:
        with open(args.split_path, 'r') as f:
            lines = f.readlines()
            rgb_filenames = [line.split()[0] for line in lines]
            event_filenames = [line.split()[2] for line in lines]
    elif args.img_dir != None and args.event_dir != None:
        rgb_filenames = glob.glob(os.path.join(args.img_dir, '**/*'), recursive=True)
        event_filenames = glob.glob(os.path.join(args.event_dir, '**/*'), recursive=True)

        rgb_filenames = [f for f in rgb_filenames if f.endswith('png') or f.endswith('npy')]
        event_filenames = [f for f in event_filenames if f.endswith('png') or f.endswith('npy')]

        rgb_filenames.sort()
        event_filenames.sort()
    else:
        raise NotImplementedError

    os.makedirs(args.outdir, exist_ok=True)

    cmap = matplotlib.colormaps.get_cmap('Spectral')

    for k, rgb_path in enumerate(rgb_filenames):
        event_path = event_filenames[k]
        print(f'Progress {k+1}/{len(rgb_filenames)}: {rgb_path}')

        raw_image = cv2.imread(rgb_path)

        if event_path.endswith('png'):
            raw_event = cv2.imread(event_path)
        elif event_path.endswith('npy'):
            raw_event = np.load(event_path)
        else:
            raise NotImplementedError

        depth = model.infer(image=raw_image, event=raw_event, input_size=args.input_size)

        clip_distance = dataset2params[args.dataset]['clip_distance']
        reg_factor = dataset2params[args.dataset]['reg_factor']
        # print(clip_distance, reg_factor, depth.min(), depth.max())
        depth = convert_nl2abs_depth(depth, clip_distance, reg_factor)

        if args.save_numpy:
            out_dir = os.path.join(args.outdir, "npy")
            os.makedirs(out_dir, exist_ok=True)
            output_path = os.path.join(
                out_dir,
                os.path.splitext(os.path.basename(rgb_path))[0] + ".npy",
            )
            np.save(output_path, depth)

        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)

        if args.grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

        out_dir = os.path.join(args.outdir, "vis")
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(
            out_dir, os.path.splitext(os.path.basename(rgb_path))[0] + ".png"
        )
        if args.pred_only:
            cv2.imwrite(output_path, depth)
        else:
            split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
            combined_result = cv2.hconcat([raw_image, split_region, depth])

            cv2.imwrite(output_path, combined_result)
