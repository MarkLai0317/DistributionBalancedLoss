import numpy as np



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


