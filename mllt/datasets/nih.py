import numpy as np
from .custom import CustomDataset
from .registry import DATASETS
import mmcv


@DATASETS.register_module
# class CocoDataset(CustomDataset):

#     CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
#                'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
#                'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
#                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
#                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
#                'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
#                'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
#                'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
#                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
#                'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
#                'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
#                'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
#                'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
#                'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush')

#     def __init__(self, **kwargs):
#         super(CocoDataset, self).__init__(**kwargs)
#         self.index_dic = self.get_index_dic()

#     def load_annotations(self, ann_file, LT_ann_file=None):

#         self.coco = COCO(ann_file)
#         self.cat_ids = self.coco.getCatIds()
#         self.cat2label = {
#             cat_id: i + 1
#             for i, cat_id in enumerate(self.cat_ids)
#         }

#         self.categories = self.cat_ids  # cat_ids for coco and cat_names for voc
#         if LT_ann_file is not None:
#             self.img_ids = []
#             for LT_ann_file in LT_ann_file:
#                 self.img_ids += mmcv.list_from_file(LT_ann_file)
#         else:
#             self.img_ids = self.coco.getImgIds()
#         img_infos = []
#         for i in self.img_ids:
#             info = self.coco.loadImgs([int(i)])[0]
#             info['filename'] = info['file_name']
#             img_infos.append(info)
#         return img_infos

#     def get_ann_info(self, idx):
#         img_id = self.img_infos[idx]['id']
#         ann_ids = self.coco.getAnnIds(imgIds=[img_id])
#         ann_info = self.coco.loadAnns(ann_ids)
#         ann = self._parse_ann_info(ann_info)
#         return ann

#     def _filter_imgs(self, min_size=32):
#         """Filter images too small or without ground truths."""
#         valid_inds = []
#         ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
#         for i, img_info in enumerate(self.img_infos):
#             if self.img_ids[i] not in ids_with_ann:
#                 continue
#             if min(img_info['width'], img_info['height']) >= min_size:
#                 valid_inds.append(i)
#         return valid_inds

#     def _parse_ann_info(self, ann_info):
#         """Parse label annotation.

#         Args:
#             ann_info (list[dict]): Annotation info of an image.

#         Returns:
#             dict: A dict containing the following key: labels
#         """

#         gt_labels = np.zeros((len(self.CLASSES), ), dtype=np.int64)
#         cat_ids = set()
#         for i, ann in enumerate(ann_info):
#             if ann.get('ignore', False):
#                 continue
#             x1, y1, w, h = ann['bbox']
#             if ann['area'] <= 0 or w < 1 or h < 1:
#                 continue
#             cat_ids.add(ann['category_id'])
#         for cat_id in cat_ids:
#             gt_labels[self.cat2label[cat_id]-1] = 1

#         ann = dict(labels=gt_labels)

#         return ann

class NIHDataset(CustomDataset):

    CLASSES = ('Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
    'No Finding', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax')


    def __init__(self, **kwargs):
        super(NIHDataset, self).__init__(**kwargs)
        self.index_dic = self.get_index_dic()

@DATASETS.register_module
class NIHDatasetGroup1(CustomDataset):

    CLASSES = ('Infiltration', 'Effusion', 'Atelectasis')


    def __init__(self, **kwargs):
        super(NIHDatasetGroup1, self).__init__(**kwargs)
        self.index_dic = self.get_index_dic()

@DATASETS.register_module
class NIHDatasetGroup2(CustomDataset):

    CLASSES = ('Mass', 'Nodule', 'Consolidation', 'Pleural_Thickening', 'Pneumothorax')


    def __init__(self, **kwargs):
        super(NIHDatasetGroup2, self).__init__(**kwargs)
        self.index_dic = self.get_index_dic()

@DATASETS.register_module
class NIHDatasetGroup3(CustomDataset):

    CLASSES = ('Edema', 'Emphysema', 'Cardiomegaly', 'Fibrosis', 'Pneumonia', 'Hernia', 'No Finding')


    def __init__(self, **kwargs):
        super(NIHDatasetGroup3, self).__init__(**kwargs)
        self.index_dic = self.get_index_dic()