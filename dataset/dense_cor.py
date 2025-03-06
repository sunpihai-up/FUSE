import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from dataset.transform import Resize, NormalizeImage, PrepareForNet, Crop

def mask_left(image):
    height, width = image.shape[:2]

    left_half = width // 2

    image[:, :left_half] = 0
    return image

class Dense(Dataset):
    def __init__(self, filelist_path, mode, normalized_d, size=(518, 518)):
        self.mode = mode
        self.size = size
        self.normalized_d = normalized_d
        
        with open(filelist_path, "r") as f:
            self.filelist = f.read().splitlines()

        net_w, net_h = size
        self.transform = Compose(
            [
                Resize(
                    width=net_w,
                    height=net_h,
                    resize_target=True if mode == "train" else False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
            + ([Crop(size[0])] if self.mode == "train" else [])
        )

    def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.float32)

    def gray2rgb(gray):
        return np.stack((gray,) * 3, axis=-1)

    def prepare_depth(self, depth, reg_factor, d_max):
        # Normalize depth
        # depth = np.clip(depth, 0.0, d_max)
        depth = depth / d_max
        depth = np.log(depth + 1e-6) / reg_factor + 1.0
        # depth = depth.clip(0.0, 1.0)
        return depth

    def __getitem__(self, item):
        # reg_factor, d_max = 6.2044, 1000
        reg_factor, d_min, d_max = 3.70378, 1.97041, 80

        img_path = self.filelist[item].split(" ")[0]
        depth_path = self.filelist[item].split(" ")[1]
        event_voxel_path = self.filelist[item].split(" ")[2]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        # image = self.gray2rgb(self.rgb2gray(image))
        
        depth = np.load(depth_path)
        event_voxel = np.load(event_voxel_path)
        if self.normalized_d:
            # Convert absolute scale depth to normalized log depth
            depth = self.prepare_depth(depth, reg_factor, d_max)
        sample = self.transform({"image": image, "depth": depth, "event_voxel": event_voxel})


        image = torch.from_numpy(sample["image"])
        event_voxel = torch.from_numpy(sample["event_voxel"])
        sample["depth"] = torch.from_numpy(sample["depth"])
        sample["input"] = torch.cat([image, event_voxel], dim=0)

        del sample['image']
        del sample['event_voxel']
        
        # sample['valid_mask'] = torch.isfinite(sample["depth"])
        # sample['valid_mask'] = torch.logical_and(sample["valid_mask"], sample['depth'] > 0)
        sample['valid_mask'] = torch.logical_and(sample['depth'] > 0, sample['depth'] < 1000)
        # sample["valid_mask"] = np.isfinite(sample["depth"]) & (sample["depth"] > 0)
        sample["image_path"] = img_path

        return sample

    def __len__(self):
        return len(self.filelist)
