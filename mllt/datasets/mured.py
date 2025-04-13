
import numpy as np

from .custom import CustomDataset
from .registry import DATASETS

import os.path as osp

import mmcv
import numpy as np
from mmcv.parallel import DataContainer as DC
from torch.utils.data import Dataset

from .registry import DATASETS
from .transforms import ImageTransform, Numpy2Tensor
from .utils import to_tensor, random_scale
from .extra_aug import ExtraAugmentation

@DATASETS.register_module
class MuredDataset(CustomDataset):
    """Custom dataset for detection.

    Annotation format:
    [
        {
            'filename': 'a.jpg',
            'width': 1280,
            'height': 720,
            'ann': {
                'labels': <np.ndarray> (n, )
            }
        },
        ...
    ]

    The `ann` field is optional for testing.
    """

    CLASSES = ('DR', 'NORMAL', 'MH', 'ODC', 'TSLN', 'ARMD', 'DN', 'MYA', 'BRVO', 'ODP',
       'CRVO', 'CNV', 'RS', 'ODE', 'LS', 'CSR', 'HTR', 'ASR', 'CRS', 'OTHER')

    def __init__(self,
                 ann_file,
                 img_prefix,
                 img_scale,
                 img_norm_cfg,
                 LT_ann_file=None,
                 multiscale_mode='value',
                 size_divisor=None,
                 flip_ratio=0,
                 extra_aug=None,
                 resize_keep_ratio=True,
                 test_mode=False,
                 class_split=None,
                 see_only=set(),
                 save_info=False):
        super(MuredDataset, self).__init__(
            ann_file=ann_file,
            img_prefix=img_prefix,
            img_scale=img_scale,
            img_norm_cfg=img_norm_cfg,
            LT_ann_file=LT_ann_file,
            multiscale_mode=multiscale_mode,
            size_divisor=size_divisor,
            flip_ratio=flip_ratio,
            extra_aug=extra_aug,
            resize_keep_ratio=resize_keep_ratio,
            test_mode=test_mode,
            class_split=class_split,
            save_info=save_info)
        

@DATASETS.register_module
class MuredDatasetGroup1(CustomDataset):
    """Custom dataset for detection.

    Annotation format:
    [
        {
            'filename': 'a.jpg',
            'width': 1280,
            'height': 720,
            'ann': {
                'labels': <np.ndarray> (n, )
            }
        },
        ...
    ]

    The `ann` field is optional for testing.
    """

    CLASSES = ('DR','NORMAL','ODC','OTHER')
    

    def __init__(self,
                 ann_file,
                 img_prefix,
                 img_scale,
                 img_norm_cfg,
                 LT_ann_file=None,
                 multiscale_mode='value',
                 size_divisor=None,
                 flip_ratio=0,
                 extra_aug=None,
                 resize_keep_ratio=True,
                 test_mode=False,
                 class_split=None,
                 see_only=set(),
                 save_info=False):
        super(MuredDatasetGroup1, self).__init__(
            ann_file=ann_file,
            img_prefix=img_prefix,
            img_scale=img_scale,
            img_norm_cfg=img_norm_cfg,
            LT_ann_file=LT_ann_file,
            multiscale_mode=multiscale_mode,
            size_divisor=size_divisor,
            flip_ratio=flip_ratio,
            extra_aug=extra_aug,
            resize_keep_ratio=resize_keep_ratio,
            test_mode=test_mode,
            class_split=class_split,
            save_info=save_info)
    def get_index_dic(self, list=False, get_labels=False):
        """ build a dict with class as key and img_ids as values
        :return: dict()
        """
        if self.single_label:
            return

        num_classes = len(self.get_ann_info(0)['labels'])
        gt_labels = []
        idx2img_id = []
        img_id2idx = dict()
        co_labels = [[] for _ in range(num_classes)]
        condition_prob = np.zeros([num_classes, num_classes])

        if list:
            index_dic = [[] for i in range(num_classes + 1)]  # +1 for all-zero class
        else:
            index_dic = dict()
            for i in range(num_classes + 1):  # +1 for all-zero class
                index_dic[i] = []

        all_zero_class_idx = num_classes  # Index for all-zero class

        for i, img_info in enumerate(self.img_infos):
            img_id = img_info['id']
            label = self.get_ann_info(i)['labels']
            gt_labels.append(label)
            idx2img_id.append(img_id)
            img_id2idx[img_id] = i

            if np.sum(label) == 0:
                # If all labels are zero, assign to all-zero class
                index_dic[all_zero_class_idx].append(i)
            else:
                for idx in np.where(np.asarray(label) == 1)[0]:
                    index_dic[idx].append(i)
                    co_labels[idx].append(label)

        for cla in range(num_classes):
            cls_labels = co_labels[cla]
            num = len(cls_labels)
            if num > 0:
                condition_prob[cla] = np.sum(np.asarray(cls_labels), axis=0) / num

        ''' save original dataset statistics, run once!'''
        if self.save_info:
            self._save_info(gt_labels, img_id2idx, idx2img_id, condition_prob)

        if get_labels:
            # Include the all-zero class in gt_labels for sampling purposes
            all_zero_labels = [[0] * num_classes] * len(index_dic[all_zero_class_idx])
            co_labels.append(all_zero_labels)
            return index_dic, co_labels
        else:
            return index_dic


@DATASETS.register_module
class MuredDatasetGroup2(CustomDataset):
    """Custom dataset for detection.

    Annotation format:
    [
        {
            'filename': 'a.jpg',
            'width': 1280,
            'height': 720,
            'ann': {
                'labels': <np.ndarray> (n, )
            }
        },
        ...
    ]

    The `ann` field is optional for testing.
    """

    CLASSES = ('MH','DN','ARMD','TSLN','MYA','BRVO', 'all_zero')
    

    def __init__(self,
                 ann_file,
                 img_prefix,
                 img_scale,
                 img_norm_cfg,
                 LT_ann_file=None,
                 multiscale_mode='value',
                 size_divisor=None,
                 flip_ratio=0,
                 extra_aug=None,
                 resize_keep_ratio=True,
                 test_mode=False,
                 class_split=None,
                 see_only=set(),
                 save_info=False):
        super(MuredDatasetGroup2, self).__init__(
            ann_file=ann_file,
            img_prefix=img_prefix,
            img_scale=img_scale,
            img_norm_cfg=img_norm_cfg,
            LT_ann_file=LT_ann_file,
            multiscale_mode=multiscale_mode,
            size_divisor=size_divisor,
            flip_ratio=flip_ratio,
            extra_aug=extra_aug,
            resize_keep_ratio=resize_keep_ratio,
            test_mode=test_mode,
            class_split=class_split,
            save_info=save_info)
    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        # load image
        img = mmcv.imread(osp.join(self.img_prefix, img_info['filename']))

        ann = self.get_ann_info(idx)
        gt_labels = ann['labels']

        gt_labels = np.array(gt_labels)[[2, 6, 5, 4, 7, 8]]

        # extra augmentation
        if self.extra_aug is not None:
            img, gt_labels = self.extra_aug(img, gt_labels)

        # apply transforms
        flip = True if np.random.rand() < self.flip_ratio else False
        # randomly sample a scale
        img_scale = random_scale(self.img_scales, self.multiscale_mode)
        img, img_shape, pad_shape, scale_factor = self.img_transform(
            img, img_scale, flip, keep_ratio=self.resize_keep_ratio)
        img = img.copy()

        ori_shape = (img_info['height'], img_info['width'], 3)
        img_meta = dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=flip)
        data = dict(
            img=DC(to_tensor(img), stack=True),
            img_meta=DC(img_meta, cpu_only=True),
            gt_labels=to_tensor(gt_labels))
        return data
    # def get_index_dic(self, list=False, get_labels=False):
    #     """ build a dict with class as key and img_ids as values
    #     :return: dict()
    #     """
    #     if self.single_label:
    #         return

    #     num_classes = len(self.get_ann_info(0)['labels'])
    #     gt_labels = []
    #     idx2img_id = []
    #     img_id2idx = dict()
    #     co_labels = [[] for _ in range(num_classes)]
    #     condition_prob = np.zeros([num_classes, num_classes])

    #     if list:
    #         index_dic = [[] for i in range(num_classes + 1)]  # +1 for all-zero class
    #     else:
    #         index_dic = dict()
    #         for i in range(num_classes + 1):  # +1 for all-zero class
    #             index_dic[i] = []

    #     all_zero_class_idx = num_classes  # Index for all-zero class

    #     for i, img_info in enumerate(self.img_infos):
    #         img_id = img_info['id']
    #         label = self.get_ann_info(i)['labels']
    #         gt_labels.append(label)
    #         idx2img_id.append(img_id)
    #         img_id2idx[img_id] = i

    #         if np.sum(label) == 0:
    #             # If all labels are zero, assign to all-zero class
    #             index_dic[all_zero_class_idx].append(i)
    #         else:
    #             for idx in np.where(np.asarray(label) == 1)[0]:
    #                 index_dic[idx].append(i)
    #                 co_labels[idx].append(label)

    #     for cla in range(num_classes):
    #         cls_labels = co_labels[cla]
    #         num = len(cls_labels)
    #         if num > 0:
    #             condition_prob[cla] = np.sum(np.asarray(cls_labels), axis=0) / num

    #     ''' save original dataset statistics, run once!'''
    #     if self.save_info:
    #         self._save_info(gt_labels, img_id2idx, idx2img_id, condition_prob)

    #     if get_labels:
    #         # Include the all-zero class in gt_labels for sampling purposes
    #         all_zero_labels = [[0] * num_classes] * len(index_dic[all_zero_class_idx])
    #         co_labels.append(all_zero_labels)
    #         return index_dic, co_labels
    #     else:
    #         return index_dic


@DATASETS.register_module
class MuredDatasetGroup3(CustomDataset):
    """Custom dataset for detection.

    Annotation format:
    [
        {
            'filename': 'a.jpg',
            'width': 1280,
            'height': 720,
            'ann': {
                'labels': <np.ndarray> (n, )
            }
        },
        ...
    ]

    The `ann` field is optional for testing.
    """

    CLASSES = ('ODP','CNV','RS','ODE','CRVO','LS','CSR','HTR','ASR','CRS')
    

    def __init__(self,
                 ann_file,
                 img_prefix,
                 img_scale,
                 img_norm_cfg,
                 LT_ann_file=None,
                 multiscale_mode='value',
                 size_divisor=None,
                 flip_ratio=0,
                 extra_aug=None,
                 resize_keep_ratio=True,
                 test_mode=False,
                 class_split=None,
                 see_only=set(),
                 save_info=False):
        super(MuredDatasetGroup3, self).__init__(
            ann_file=ann_file,
            img_prefix=img_prefix,
            img_scale=img_scale,
            img_norm_cfg=img_norm_cfg,
            LT_ann_file=LT_ann_file,
            multiscale_mode=multiscale_mode,
            size_divisor=size_divisor,
            flip_ratio=flip_ratio,
            extra_aug=extra_aug,
            resize_keep_ratio=resize_keep_ratio,
            test_mode=test_mode,
            class_split=class_split,
            save_info=save_info)
        
    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        # load image
        img = mmcv.imread(osp.join(self.img_prefix, img_info['filename']))

        ann = self.get_ann_info(idx)
        gt_labels = ann['labels']

        gt_labels = np.array(gt_labels)[[9, 11, 12, 13, 10, 14, 15, 16, 17, 18]]

        # extra augmentation
        if self.extra_aug is not None:
            img, gt_labels = self.extra_aug(img, gt_labels)

        # apply transforms
        flip = True if np.random.rand() < self.flip_ratio else False
        # randomly sample a scale
        img_scale = random_scale(self.img_scales, self.multiscale_mode)
        img, img_shape, pad_shape, scale_factor = self.img_transform(
            img, img_scale, flip, keep_ratio=self.resize_keep_ratio)
        img = img.copy()

        ori_shape = (img_info['height'], img_info['width'], 3)
        img_meta = dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=flip)
        data = dict(
            img=DC(to_tensor(img), stack=True),
            img_meta=DC(img_meta, cpu_only=True),
            gt_labels=to_tensor(gt_labels))
        return data
    # def get_index_dic(self, list=False, get_labels=False):
    #     """ build a dict with class as key and img_ids as values
    #     :return: dict()
    #     """
    #     if self.single_label:
    #         return

    #     num_classes = len(self.get_ann_info(0)['labels'])
    #     gt_labels = []
    #     idx2img_id = []
    #     img_id2idx = dict()
    #     co_labels = [[] for _ in range(num_classes)]
    #     condition_prob = np.zeros([num_classes, num_classes])

    #     if list:
    #         index_dic = [[] for i in range(num_classes + 1)]  # +1 for all-zero class
    #     else:
    #         index_dic = dict()
    #         for i in range(num_classes + 1):  # +1 for all-zero class
    #             index_dic[i] = []

    #     all_zero_class_idx = num_classes  # Index for all-zero class

    #     for i, img_info in enumerate(self.img_infos):
    #         img_id = img_info['id']
    #         label = self.get_ann_info(i)['labels']
    #         gt_labels.append(label)
    #         idx2img_id.append(img_id)
    #         img_id2idx[img_id] = i

    #         if np.sum(label) == 0:
    #             # If all labels are zero, assign to all-zero class
    #             index_dic[all_zero_class_idx].append(i)
    #         else:
    #             for idx in np.where(np.asarray(label) == 1)[0]:
    #                 index_dic[idx].append(i)
    #                 co_labels[idx].append(label)

    #     for cla in range(num_classes):
    #         cls_labels = co_labels[cla]
    #         num = len(cls_labels)
    #         if num > 0:
    #             condition_prob[cla] = np.sum(np.asarray(cls_labels), axis=0) / num

    #     ''' save original dataset statistics, run once!'''
    #     if self.save_info:
    #         self._save_info(gt_labels, img_id2idx, idx2img_id, condition_prob)

    #     if get_labels:
    #         # Include the all-zero class in gt_labels for sampling purposes
    #         all_zero_labels = [[0] * num_classes] * len(index_dic[all_zero_class_idx])
    #         co_labels.append(all_zero_labels)
    #         return index_dic, co_labels
    #     else:
    #         return index_dic
