import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from dataset.transform import (
    Resize,
    NormalizeImage,
    PrepareForNet,
    Crop,
    Image_Corruption,
)

DEPTH_MAX = 1000
ALPHA = 5.7

# RGB_MEAN  = [ 93.693515, 94.399027, 96.141395 ]
# RGB_STD   = [ 53.938046, 52.463300, 54.016670]
# GRAY_MEAN = [ 94.386783 ]
# GRAY_STD  = [ 52.621100 ]
GRAY_MEAN = 0.370166
GRAY_STD = 0.206369


class EventScape_Fuse_Cor(Dataset):
    def __init__(self, filelist_path, mode, size):
        self.mode = mode
        self.size = size

        with open(filelist_path, "r") as f:
            self.filelist = f.read().splitlines()

        net_w, net_h = size
        self.image_cor_process = Image_Corruption(
            mask_pro=0.5,
            cor_types=["blur", "mask", "overexpose"],
            brightness_range=[0.5, 1.5],
        )
        self.voxel_cor_process = Image_Corruption(
            mask_pro=0.5, cor_types=["blur", "mask"]
        )

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
                # NormalizeImage(mean=mean, std=std),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
            + ([Crop(size[0])] if self.mode == "train" else [])
        )

    def __getitem__(self, item):
        img_path = self.filelist[item].split(" ")[0]
        event_voxel_path = self.filelist[item].split(" ")[2]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_cor = image
        if self.mode == "train":
            image_cor = self.image_cor_process(image_cor)

        image = image / 255.0
        image_cor = image_cor / 255.0

        event_voxel = np.load(event_voxel_path)
        event_voxel = event_voxel.transpose(1, 2, 0)
        if self.mode == "train":
            event_voxel = self.voxel_cor_process(event_voxel)
        event_voxel = event_voxel.transpose(2, 0, 1)

        sample = self.transform(
            {"image": image, "event_voxel": event_voxel, "image_cor": image_cor}
        )

        sample["image"] = torch.from_numpy(sample["image"])
        event_voxel = torch.from_numpy(sample["event_voxel"])
        image_cor = torch.from_numpy(sample["image_cor"])
        sample["input"] = torch.cat([image_cor, event_voxel], dim=0)

        del sample["image_cor"]
        del sample["event_voxel"]

        sample["image_path"] = img_path

        return sample

    def __len__(self):
        return len(self.filelist)


class EventScape_Fuse(Dataset):
    def __init__(self, filelist_path, mode, size):
        self.mode = mode
        self.size = size

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
                # NormalizeImage(mean=mean, std=std),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
            + ([Crop(size[0])] if self.mode == "train" else [])
        )

    def __getitem__(self, item):
        img_path = self.filelist[item].split(" ")[0]
        event_voxel_path = self.filelist[item].split(" ")[2]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_cor = image

        image = image / 255.0
        image_cor = image_cor / 255.0

        event_voxel = np.load(event_voxel_path)

        sample = self.transform(
            {"image": image, "event_voxel": event_voxel, "image_cor": image_cor}
        )

        sample["image"] = torch.from_numpy(sample["image"])
        event_voxel = torch.from_numpy(sample["event_voxel"])
        image_cor = torch.from_numpy(sample["image_cor"])
        sample["input"] = torch.cat([image_cor, event_voxel], dim=0)

        del sample["image_cor"]
        del sample["event_voxel"]

        sample["image_path"] = img_path

        return sample

    def __len__(self):
        return len(self.filelist)
