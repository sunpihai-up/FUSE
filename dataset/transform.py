import cv2
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
import pdb
import math


def apply_min_size(sample, size, image_interpolation_method=cv2.INTER_AREA):
    """Rezise the sample to ensure the given size. Keeps aspect ratio.

    Args:
        sample (dict): sample
        size (tuple): image size

    Returns:
        tuple: new size
    """
    shape = list(sample["disparity"].shape)

    if shape[0] >= size[0] and shape[1] >= size[1]:
        return sample

    scale = [0, 0]
    scale[0] = size[0] / shape[0]
    scale[1] = size[1] / shape[1]

    scale = max(scale)

    shape[0] = math.ceil(scale * shape[0])
    shape[1] = math.ceil(scale * shape[1])

    # resize
    sample["image"] = cv2.resize(
        sample["image"], tuple(shape[::-1]), interpolation=image_interpolation_method
    )

    if "image_cor" in sample:
        sample["image_cor"] = cv2.resize(
            sample["image_cor"],
            tuple(shape[::-1]),
            interpolation=image_interpolation_method,
        )

    sample["disparity"] = cv2.resize(
        sample["disparity"], tuple(shape[::-1]), interpolation=cv2.INTER_NEAREST
    )
    sample["mask"] = cv2.resize(
        sample["mask"].astype(np.float32),
        tuple(shape[::-1]),
        interpolation=cv2.INTER_NEAREST,
    )
    sample["mask"] = sample["mask"].astype(bool)

    return tuple(shape)


class Resize(object):
    """Resize sample to given size (width, height)."""

    def __init__(
        self,
        width,
        height,
        resize_target=True,
        keep_aspect_ratio=False,
        ensure_multiple_of=1,
        resize_method="lower_bound",
        image_interpolation_method=cv2.INTER_AREA,
    ):
        """Init.

        Args:
            width (int): desired output width
            height (int): desired output height
            resize_target (bool, optional):
                True: Resize the full sample (image, mask, target).
                False: Resize image only.
                Defaults to True.
            keep_aspect_ratio (bool, optional):
                True: Keep the aspect ratio of the input sample.
                Output sample might not have the given width and height, and
                resize behaviour depends on the parameter 'resize_method'.
                Defaults to False.
            ensure_multiple_of (int, optional):
                Output width and height is constrained to be multiple of this parameter.
                Defaults to 1.
            resize_method (str, optional):
                "lower_bound": Output will be at least as large as the given size.
                "upper_bound": Output will be at max as large as the given size. (Output size might be smaller than given size.)
                "minimal": Scale as least as possible.  (Output size might be smaller than given size.)
                Defaults to "lower_bound".
        """
        self.__width = width
        self.__height = height

        self.__resize_target = resize_target
        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method
        self.__image_interpolation_method = image_interpolation_method

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if max_val is not None and y > max_val:
            y = (np.floor(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if y < min_val:
            y = (np.ceil(x / self.__multiple_of) * self.__multiple_of).astype(int)

        return y

    def get_size(self, width, height):
        # determine new height and width
        scale_height = self.__height / height
        scale_width = self.__width / width

        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                # scale such that output size is lower bound
                if scale_width > scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                # scale such that output size is upper bound
                if scale_width < scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "minimal":
                # scale as least as possbile
                if abs(1 - scale_width) < abs(1 - scale_height):
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            else:
                raise ValueError(
                    f"resize_method {self.__resize_method} not implemented"
                )

        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, min_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, min_val=self.__width
            )
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, max_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, max_val=self.__width
            )
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(f"resize_method {self.__resize_method} not implemented")

        return (new_width, new_height)

    def __call__(self, sample):
        width, height = self.get_size(
            sample["image"].shape[1], sample["image"].shape[0]
        )

        # resize sample
        sample["image"] = cv2.resize(
            sample["image"],
            (width, height),
            interpolation=self.__image_interpolation_method,
        )

        if "image_cor" in sample:
            sample["image_cor"] = cv2.resize(
                sample["image_cor"],
                (width, height),
                interpolation=self.__image_interpolation_method,
            )

        if "event_voxel" in sample:
            sample["event_voxel"] = sample["event_voxel"].transpose(1, 2, 0)
            sample["event_voxel"] = cv2.resize(
                sample["event_voxel"],
                (width, height),
                interpolation=self.__image_interpolation_method,
            )

        if self.__resize_target:
            if "disparity" in sample:
                sample["disparity"] = cv2.resize(
                    sample["disparity"],
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )

            if "depth" in sample:
                sample["depth"] = cv2.resize(
                    sample["depth"], (width, height), interpolation=cv2.INTER_NEAREST
                )

            if "semseg_mask" in sample:
                # sample["semseg_mask"] = cv2.resize(
                #     sample["semseg_mask"], (width, height), interpolation=cv2.INTER_NEAREST
                # )
                sample["semseg_mask"] = F.interpolate(
                    torch.from_numpy(sample["semseg_mask"]).float()[None, None, ...],
                    (height, width),
                    mode="nearest",
                ).numpy()[0, 0]

            if "mask" in sample:
                sample["mask"] = cv2.resize(
                    sample["mask"].astype(np.float32),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
                # sample["mask"] = sample["mask"].astype(bool)

        return sample


class NormalizeImage(object):
    """Normlize image by given mean and std."""

    def __init__(self, mean, std):
        self.__mean = mean
        self.__std = std

    def normalize_voxelgrid(self, event_tensor):
        mask = np.nonzero(event_tensor)
        if mask[0].size > 0:
            mean, stddev = event_tensor[mask].mean(), event_tensor[mask].std()
            if stddev > 0:
                event_tensor[mask] = (event_tensor[mask] - mean) / stddev
        return event_tensor

    def __call__(self, sample):
        sample["image"] = (sample["image"] - self.__mean) / self.__std

        if "image_cor" in sample:
            sample["image_cor"] = (sample["image_cor"] - self.__mean) / self.__std

        if "event_voxel" in sample:
            sample["event_voxel"] = self.normalize_voxelgrid(sample["event_voxel"])
        return sample


class PrepareForNet(object):
    """Prepare sample for usage as network input."""

    def __init__(self):
        pass

    def __call__(self, sample):
        image = np.transpose(sample["image"], (2, 0, 1))
        sample["image"] = np.ascontiguousarray(image).astype(np.float32)

        if "image_cor" in sample:
            image_cor = np.transpose(sample["image_cor"], (2, 0, 1))
            sample["image_cor"] = np.ascontiguousarray(image_cor).astype(np.float32)

        if "mask" in sample:
            sample["mask"] = sample["mask"].astype(np.float32)
            sample["mask"] = np.ascontiguousarray(sample["mask"])

        if "depth" in sample:
            depth = sample["depth"].astype(np.float32)
            sample["depth"] = np.ascontiguousarray(depth)

        if "event_voxel" in sample:
            event_voxel = np.transpose(sample["event_voxel"], (2, 0, 1))
            event_voxel = event_voxel.astype(np.float32)
            sample["event_voxel"] = np.ascontiguousarray(event_voxel)

        if "semseg_mask" in sample:
            sample["semseg_mask"] = sample["semseg_mask"].astype(np.float32)
            sample["semseg_mask"] = np.ascontiguousarray(sample["semseg_mask"])

        return sample


class Crop(object):
    """Crop sample for batch-wise training. Image is of shape CxHxW"""

    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, sample):
        h, w = sample["image"].shape[-2:]
        assert h >= self.size[0] and w >= self.size[1], "Wrong size"

        h_start = np.random.randint(0, h - self.size[0] + 1)
        w_start = np.random.randint(0, w - self.size[1] + 1)
        h_end = h_start + self.size[0]
        w_end = w_start + self.size[1]

        sample["image"] = sample["image"][:, h_start:h_end, w_start:w_end]

        if "image_cor" in sample:
            sample["image_cor"] = sample["image_cor"][:, h_start:h_end, w_start:w_end]

        if "event_voxel" in sample:
            sample["event_voxel"] = sample["event_voxel"][
                :, h_start:h_end, w_start:w_end
            ]

        if "depth" in sample:
            sample["depth"] = sample["depth"][h_start:h_end, w_start:w_end]

        if "mask" in sample:
            sample["mask"] = sample["mask"][h_start:h_end, w_start:w_end]

        if "semseg_mask" in sample:
            sample["semseg_mask"] = sample["semseg_mask"][h_start:h_end, w_start:w_end]

        return sample


class Noise(object):
    def __init__(self):
        pass

    # def add_gaussian_noise(self, image, mean=0, sigma=25):
    #     """
    #     Add Gaussian noise to a grayscale image.
    #     """
    #     # Ensure the image is of float type for correct mathematical operations
    #     noisy_image = np.copy(image).astype(float)
    #     # Generate Gaussian noise
    #     gauss = np.random.normal(mean, sigma, image.shape)
    #     # Add noise to the image
    #     noisy_image = noisy_image + gauss

    #     # Clip image pixel values to the valid range [0, 255]
    #     noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    #     return noisy_image

    def add_gaussian_noise(self, image, mean=0, sigma=0.05):
        """
        Add Gaussian noise to multi-channel data with automatic range adaptation.
        The noise level (sigma) should be specified in normalized [0,1] range.
        """
        # Store original dtype and convert to float32 for calculations
        original_dtype = image.dtype
        image = image.astype(np.float32)

        data_min = np.min(image)
        data_max = np.max(image)

        # Normalize to [0, 1] range with epsilon to avoid division by zero
        normalized = (image - data_min) / (data_max - data_min + 1e-8)

        # Generate Gaussian noise in normalized space
        noise = np.random.normal(mean, sigma, image.shape)
        print(f"noise: {noise.min()}, {noise.max()}")
        # Add noise and clip to valid normalized range
        noisy_normalized = np.clip(normalized + noise, 0, 1)

        # Restore original data range
        noisy_image = noisy_normalized * (data_max - data_min) + data_min

        # Convert back to original data type
        return noisy_image.astype(original_dtype)

    def add_salt_and_pepper_noise(self, image, amount=0.05, salt_vs_pepper=0.5):
        """
        Add salt and pepper noise to a grayscale image.
        """
        noisy_image = np.copy(image)
        num_salt = np.ceil(amount * image.size * salt_vs_pepper)
        num_pepper = np.ceil(amount * image.size * (1.0 - salt_vs_pepper))

        # Add salt noise (white)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        noisy_image[tuple(coords)] = 255

        # Add pepper noise (black)
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        noisy_image[tuple(coords)] = 0

        return noisy_image

    def __call__(self, image):
        image = self.add_gaussian_noise(image)
        # image = self.add_salt_and_pepper_noise(image)
        return image


import random


class Image_Corruption(object):
    def __init__(
        self,
        # noise_pro=0.3,
        mask_pro=0.3,
        num_masks=3,
        mask_radio=0.2,
        exposure_alpha=2.5,
        exposure_beta=200,
        blur_size=25,
        blur_sigmaX=2,
        brightness_range=(1, 1),
        cor_types=["blur", "overexpose", "mask"],
    ):
        # self.add_noise = Noise()
        # self.noise_pro = noise_pro
        self.mask_pro = mask_pro

        self.num_masks = num_masks
        self.mask_radio = mask_radio

        self.exposure_alpha = exposure_alpha
        self.exposure_beta = exposure_beta

        self.blur_size = blur_size
        self.blur_sigmaX = blur_sigmaX

        self.brightness_range = brightness_range 
        self.cor_types = cor_types

    def generate_masks(self, image_shape, num_masks, width_ratio=0.2, height_ratio=0.2):
        assert width_ratio <= 1
        assert height_ratio <= 1

        height, width, _ = image_shape
        masks = []

        # Convert ratios to actual dimensions
        w = int(width * width_ratio)
        h = int(height * height_ratio)

        for _ in range(num_masks):
            mask = np.zeros((height, width), dtype=np.uint8)
            # Generate a rectangle mask with size based on the percentage of the image dimensions
            x = random.randint(0, width - w) if width > w else 0
            y = random.randint(0, height - h) if height > h else 0
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

            masks.append(mask)

        return masks

    def apply_gaussian_blur_region(self, image, mask, k_size=(15, 15), sigmaX=0):
        if isinstance(k_size, int):
            k_size = (k_size, k_size)
        blurred = cv2.GaussianBlur(image, k_size, sigmaX)
        return np.where(mask[..., None] == 255, blurred, image)

    def apply_overexposure_region(self, image, mask, alpha=1.5, beta=50):
        overexposed = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return np.where(mask[..., None] == 255, overexposed, image)

    def mask_region(self, image, mask, mask_color=(0, 0, 0)):
        colored_mask = np.zeros_like(image)
        colored_mask[:] = mask_color
        return np.where(mask[..., None] == 255, colored_mask, image)

    def change_brightness(self, image, factor=0.5):
        image_float = image.astype(np.float32) * factor
        return image_float.round().clip(0, 255).astype(np.uint8)
    
    def __call__(self, image):
        # if random.random() <= self.noise_pro:
        #     image = self.add_noise(image)

        # Change Image Brightness
        brightness_factor = random.uniform(self.brightness_range[0], self.brightness_range[1])
        image = self.change_brightness(image=image, factor=brightness_factor)
        
        if random.random() <= self.mask_pro:
            masks = self.generate_masks(
                image_shape=image.shape,
                num_masks=self.num_masks,
                width_ratio=self.mask_radio,
                height_ratio=self.mask_radio,
            )
            for mask in masks:
                effect_type = random.choice(self.cor_types)
                if effect_type == "blur":
                    image = self.apply_gaussian_blur_region(
                        image,
                        mask,
                        k_size=self.blur_size,
                        sigmaX=self.blur_sigmaX,
                    )
                elif effect_type == "overexpose":
                    image = self.apply_overexposure_region(
                        image, mask, alpha=self.exposure_alpha, beta=self.exposure_beta
                    )
                elif effect_type == "mask":
                    image = self.mask_region(image, mask, mask_color=(0, 0, 0))
                    # image = self.mask_region(image, mask, mask_color=(255, 255, 255))

        return image
