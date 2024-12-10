import numpy as np
import os
import pickle


# {
#     'gt_labels': list of array, shape = [number_of_samples, number_of_labels]
#     'class_freq': 1d array, number of positive samples for each labels, length = number of labels
#     'neg_class_freq': 1d array, number of negative samples for each labels, length = number of labels
#     'condition_prob': 2d array, shape = [number_of_labels, number_of_labels], array[i,j] = prob of j = 1 if i = 1
# }

def calculate_ratio(array: np.array, base_index: int, target_index: int):
    # Ensure the array is a 2D numpy array
    array = np.array(array)

    # Count of indices where both array[0][i] == 1 and array[1][i] == 1
    shared_count = np.sum((array[:,base_index] == 1) & (array[:,target_index] == 1))

    # Count of indices where array[0][i] == 1
    total_count = np.sum(array[:,base_index] == 1)

    # Avoid division by zero
    if total_count == 0:
        return 0
    else:
        return shared_count / total_count
def calculate_ratio_base_index(array: np.array, base_index: int):
    array = np.array(array)
    return np.array([calculate_ratio(array, base_index, i) for i in range(array.shape[1])])

# used for create "condition_prob" of class_freq.pkl"
# input is the gt_labels
def calculate_ratio_all(array: np.array):
    array = np.array(array)
    return np.array([calculate_ratio_base_index(array, i) for i in range(array.shape[1])])



def filter_label(pkl_path: str, desired_labels: str):
    coco_labels = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
        'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
        'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
        'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
        'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
        'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
        'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
        'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
        'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush'
    ]
    data = get_pkl_file(pkl_path)
    desired_indices = [coco_labels.index(label.replace(" ", "_")) for label in desired_labels]
    desired_indices.sort()

    print([coco_labels[i] for i in desired_indices])


    filtered_gt = [np.array([row[i] for i in desired_indices]) for row in data['gt_labels']]

    data["gt_labels"] = filtered_gt

    data["class_freq"] = np.sum(np.array( filtered_gt)== 1, axis=0)

    data["neg_class_freq"] = np.sum(np.array( filtered_gt)== 0, axis=0)

    data["condition_prob"] = calculate_ratio_all(np.array(filtered_gt))
    

    return data
    



def get_pkl_file(path):
    if os.path.exists(path):
        with open(path, "rb") as file:
            data = pickle.load(file)
    else:
        print("File not found at path:", path)
    return data


def split_list(lst, n):
    # Calculate the approximate size of each group
    k, m = divmod(len(lst), n)
    lst = [i for i in range(len(lst))]
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

import argparse
import data_preprocess.group_label as group_label
import pandas as pd
import data_preprocess.annotation as annotation

COCO = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
        'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
        'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
        'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
        'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
        'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
        'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
        'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
        'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush'
    ]
annotation_dir = "/home/mark/Desktop/工研院/multi-label_classification/data/coco/annotations"

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Filter labels from class_freq.pkl")
    

    parser.add_argument('--pkl_path', "-cdp",type=str, required=True, help="The name of the file to process")
    parser.add_argument('--output_dir',type=str, required=True, help="output dir")
    
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


    data = get_pkl_file(args.pkl_path)

    df = pd.DataFrame(data['gt_labels'], columns = COCO)
    df.index.name = "ID"
    df = df.reset_index()
    print(df)

    label_grouper = group_label.DPLabelGrouper(None, "ID", df)
    label_groups = label_grouper.get_label_group_list()

    for i in range(len(label_groups)):
        # filtered_data = filter_label(args.pkl_path, label_groups[i])

        # output_path = os.path.join(args.output_dir, f"class_freq_group{i+1}.pkl")
        
        # print(f"Filtered data saved to: {output_path}")
        # with open(output_path, "wb") as file:  # Open in write-binary mode
        #     pickle.dump(filtered_data, file)

        # class split

        class_split_list = split_list(label_groups[i], 3)
        class_split = {
                "head": set(class_split_list[0]),
                "middle": set(class_split_list[1]),
                "tail": set(class_split_list[2])
            }
        
        class_split_path = os.path.join(args.output_dir, f"class_split_group{i+1}.pkl")
        
        with open(class_split_path, "wb") as file:  # Open in write-binary mode
            pickle.dump(class_split, file)

        # # annotation
        # annotation.fileter_annotation(label_groups[i], "/home/mark/Desktop/工研院/multi-label_classification/data/coco/annotations/instances_train2017.json", os.path.join(annotation_dir, f"instances_train2017_group{i+1}.json"))
        # annotation.fileter_annotation(label_groups[i], "/home/mark/Desktop/工研院/multi-label_classification/data/coco/annotations/instances_val2017.json", os.path.join(annotation_dir, f"instances_val2017_group{i+1}.json"))

        
    

    


