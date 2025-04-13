import mmcv
import numpy as np
import torch

__all__ = [
    'ImageTransform', 'Numpy2Tensor'
]


class ImageTransform(object):
    """Preprocess an image.

    1. rescale the image to expected size
    2. normalize the image
    3. flip the image (if needed)
    4. pad the image (if needed)
    5. transpose to (c, h, w)
    """

    def __init__(self,
                 mean=(0, 0, 0),
                 std=(1, 1, 1),
                 to_rgb=True,
                 size_divisor=None):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb
        self.size_divisor = size_divisor

    def __call__(self, img, scale, flip=False, keep_ratio=True):
        if keep_ratio:
            img, scale_factor = mmcv.imrescale(img, scale, return_scale=True)
        else:
            img, w_scale, h_scale = mmcv.imresize(
                img, scale, return_scale=True)
            scale_factor = np.array(
                [w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
        img_shape = img.shape
        img = mmcv.imnormalize(img, self.mean, self.std, self.to_rgb)
        if flip:
            img = mmcv.imflip(img)
        if self.size_divisor is not None:
            img = mmcv.impad_to_multiple(img, self.size_divisor)
            pad_shape = img.shape
        else:
            pad_shape = img_shape
        img = img.transpose(2, 0, 1)
        # print(img.max())
        return img, img_shape, pad_shape, scale_factor
    # def __init__(self, 
    #              mean=(0.485, 0.456, 0.406), 
    #              std=(0.229, 0.224, 0.225), 
    #              to_rgb=True, 
    #              size_divisor=None):
    #     self.mean = np.array(mean, dtype=np.float32)
    #     self.std = np.array(std, dtype=np.float32)
    #     self.to_rgb = to_rgb
    #     self.size_divisor = size_divisor

    # def __call__(self, img, scale, flip=False, keep_ratio=True):
    #     if keep_ratio:
    #         img, scale_factor = mmcv.imrescale(img, scale, return_scale=True)
    #     else:
    #         img, w_scale, h_scale = mmcv.imresize(img, scale, return_scale=True)
    #         scale_factor = np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
        
    #     img_shape = img.shape

    #     # Convert to RGB if needed
    #     if self.to_rgb:
    #         img = mmcv.bgr2rgb(img)

    #     # Scale pixel values to [0,1]
    #     img = img.astype(np.float32) / 255.0

    #     # Normalize using ImageNet mean and std
    #     img = (img - self.mean) / self.std

    #     if flip:
    #         img = mmcv.imflip(img)

    #     if self.size_divisor is not None:
    #         img = mmcv.impad_to_multiple(img, self.size_divisor)
    #         pad_shape = img.shape
    #     else:
    #         pad_shape = img_shape

    #     img = img.transpose(2, 0, 1)  # Convert (H, W, C) → (C, H, W)

    #     # print(img.max())  # Should be around 2.2 (since it's normalized)
    #     return img, img_shape, pad_shape, scale_factor


class Numpy2Tensor(object):

    def __init__(self):
        pass

    def __call__(self, *args):
        if len(args) == 1:
            return torch.from_numpy(args[0])
        else:
            return tuple([torch.from_numpy(np.array(array)) for array in args])
