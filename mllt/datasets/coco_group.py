import numpy as np
from pycocotools.coco import COCO
from .custom import CustomDataset
from .registry import DATASETS
import mmcv


@DATASETS.register_module
class CocoDatasetOriginalGroup1(CustomDataset):

    CLASSES = ('person', 'car', 'backpack', 'handbag', 'bottle', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'chair', 'couch', 'potted_plant', 'dining_table', 'tv', 'laptop', 'cell_phone', 'oven', 'sink', 'refrigerator', 'book', 'vase')

    def __init__(self, **kwargs):
        super(CocoDatasetOriginalGroup1, self).__init__(**kwargs)
        self.index_dic = self.get_index_dic()

    def load_annotations(self, ann_file, LT_ann_file=None):

        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }

        self.categories = self.cat_ids  # cat_ids for coco and cat_names for voc
        if LT_ann_file is not None:
            self.img_ids = []
            for LT_ann_file in LT_ann_file:
                self.img_ids += mmcv.list_from_file(LT_ann_file)
        else:
            self.img_ids = self.coco.getImgIds()
        img_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([int(i)])[0]
            info['filename'] = info['file_name']
            img_infos.append(info)
        return img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        ann = self._parse_ann_info(ann_info)
        return ann

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.img_infos):
            if self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _parse_ann_info(self, ann_info):
        """Parse label annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.

        Returns:
            dict: A dict containing the following key: labels
        """

        gt_labels = np.zeros((len(self.CLASSES), ), dtype=np.int64)
        cat_ids = set()
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            cat_ids.add(ann['category_id'])
        for cat_id in cat_ids:
            gt_labels[self.cat2label[cat_id]-1] = 1

        ann = dict(labels=gt_labels)

        return ann
    
@DATASETS.register_module
class CocoDatasetOriginalGroup2(CustomDataset):

    CLASSES = ('bicycle', 'motorcycle', 'bus', 'truck', 'boat', 'traffic_light', 'bench', 'cat', 'dog', 'umbrella', 'tie', 'suitcase', 'sports_ball', 'baseball_bat', 'baseball_glove', 'tennis_racket', 'wine_glass', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'pizza', 'cake', 'bed', 'toilet', 'mouse', 'remote', 'keyboard', 'microwave', 'clock', 'teddy_bear')

    def __init__(self, **kwargs):
        super(CocoDatasetOriginalGroup2, self).__init__(**kwargs)
        self.index_dic = self.get_index_dic()

    def load_annotations(self, ann_file, LT_ann_file=None):

        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }

        self.categories = self.cat_ids  # cat_ids for coco and cat_names for voc
        if LT_ann_file is not None:
            self.img_ids = []
            for LT_ann_file in LT_ann_file:
                self.img_ids += mmcv.list_from_file(LT_ann_file)
        else:
            self.img_ids = self.coco.getImgIds()
        img_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([int(i)])[0]
            info['filename'] = info['file_name']
            img_infos.append(info)
        return img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        ann = self._parse_ann_info(ann_info)
        return ann

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.img_infos):
            if self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _parse_ann_info(self, ann_info):
        """Parse label annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.

        Returns:
            dict: A dict containing the following key: labels
        """

        gt_labels = np.zeros((len(self.CLASSES), ), dtype=np.int64)
        cat_ids = set()
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            cat_ids.add(ann['category_id'])
        for cat_id in cat_ids:
            gt_labels[self.cat2label[cat_id]-1] = 1

        ann = dict(labels=gt_labels)

        return ann




@DATASETS.register_module
class CocoDatasetOriginalGroup3(CustomDataset):

    CLASSES = ('airplane', 'train', 'fire_hydrant', 'stop_sign', 'parking_meter', 'bird', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'frisbee', 'skis', 'snowboard', 'kite', 'skateboard', 'surfboard', 'hot_dog', 'donut', 'toaster', 'scissors', 'hair_drier', 'toothbrush')

    def __init__(self, **kwargs):
        super(CocoDatasetOriginalGroup3, self).__init__(**kwargs)
        self.index_dic = self.get_index_dic()

    def load_annotations(self, ann_file, LT_ann_file=None):

        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }

        self.categories = self.cat_ids  # cat_ids for coco and cat_names for voc
        if LT_ann_file is not None:
            self.img_ids = []
            for LT_ann_file in LT_ann_file:
                self.img_ids += mmcv.list_from_file(LT_ann_file)
        else:
            self.img_ids = self.coco.getImgIds()
        img_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([int(i)])[0]
            info['filename'] = info['file_name']
            img_infos.append(info)
        return img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        ann = self._parse_ann_info(ann_info)
        return ann

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.img_infos):
            if self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _parse_ann_info(self, ann_info):
        """Parse label annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.

        Returns:
            dict: A dict containing the following key: labels
        """

        gt_labels = np.zeros((len(self.CLASSES), ), dtype=np.int64)
        cat_ids = set()
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            cat_ids.add(ann['category_id'])
        for cat_id in cat_ids:
            gt_labels[self.cat2label[cat_id]-1] = 1

        ann = dict(labels=gt_labels)

        return ann

