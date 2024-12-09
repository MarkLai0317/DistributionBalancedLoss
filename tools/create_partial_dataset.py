import numpy as np
import mmcv
from mmcv import mkdir_or_exist
from typing import List


class PartialVOCDataset():
    def __init__(self, year: int, save_dir: str, labels: List[int]):
        self.year = year
        self.save_dir = save_dir
        self.labels = labels

    def create_partial_dataset(self):
        data = mmcv.load('appendix/VOCdevkit/terse_gt_{}.pkl'.format(self.year))
        gt_labels = np.asarray(data['gt_labels'])
        img_id2idx = data['img_id2idx']
        idx2img_id = data['idx2img_id']

        save_dir = self.save_dir + 'partial' + str(self.year) + '/'
        mkdir_or_exist(save_dir)

        # Select given labels
        gt_labels = [arr[self.labels] for arr in gt_labels]
        num_classes = len(self.labels)

        # Write class_freq.pkl
        freq = dict()
        freq['gt_labels'] = gt_labels
        freq['class_freq'] = np.sum(gt_labels, axis=0)
        freq['neg_class_freq'] = np.shape(gt_labels)[0] - freq['class_freq']

        condition_prob = np.zeros([num_classes, num_classes])
        gt_labels = np.array(gt_labels)

        for i in range(num_classes):
            mask = gt_labels[:, i] == 1
            count_i = np.sum(mask)
            if count_i > 0:
                for j in range(num_classes):
                    count_j_given_i = np.sum(gt_labels[mask, j] == 1)
                    condition_prob[i, j] = count_j_given_i / count_i

        freq['condition_prob'] = condition_prob

        path = save_dir + 'class_freq.pkl'
        mmcv.dump(freq, path)

        # Write class_split.pkl
        split = dict()
        split['head'], split['middle'], split['tail'] = [set(np.where(freq['class_freq'] >= 100)[0]),
                                                         set(np.where((freq['class_freq'] < 100) * (freq['class_freq'] >= 20))[0]),
                                                         set(np.where(freq['class_freq'] < 20)[0])]

        path = save_dir + 'class_split.pkl'
        mmcv.dump(split, path)

        # Write img_id.txt
        path = save_dir + 'img_id.txt'
        with open(path, "w") as f:
            for img_id in idx2img_id:
                f.writelines("%s\n" % img_id)


if __name__ == '__main__':
    dataset = PartialVOCDataset(2012, './appendix/VOCdevkit/', [0, 2, 8, 15, 19])
    dataset.create_partial_dataset()
