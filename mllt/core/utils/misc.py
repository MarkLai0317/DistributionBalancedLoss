from functools import partial

import mmcv
import numpy as np

def tensor2imgs(tensor, mean=(0, 0, 0), std=(1, 1, 1), to_rgb=True):
    num_imgs = tensor.size(0)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    imgs = []
    for img_id in range(num_imgs):
        img = tensor[img_id, ...].cpu().numpy().transpose(1, 2, 0)
        img = mmcv.imdenormalize(
            img, mean, std, to_bgr=to_rgb).astype(np.uint8)
        imgs.append(np.ascontiguousarray(img))
#     return imgs
# import numpy as np
# import mmcv
# def tensor2imgs(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), to_rgb=True, apply_normalization=True):
#     print("tensor shape",tensor.shape)
#     num_imgs = tensor.size(0)
#     mean = np.array(mean, dtype=np.float32)
#     std = np.array(std, dtype=np.float32)
#     imgs = []
#     for img_id in range(num_imgs):
#         img = tensor[img_id, ...].cpu().numpy().transpose(1, 2, 0)  # Convert (C, H, W) → (H, W, C)
#         img = img / 255.0  # Scale to [0, 1]
#         img = np.clip(img, 0, 1)  # Ensure values stay in range
        
#         if apply_normalization:
#             # Apply mean-std normalization AFTER scaling to [0,1]
#             img = (img - mean) / std
        
#         imgs.append(np.ascontiguousarray(img, dtype=np.float32))
    
#     return imgs
