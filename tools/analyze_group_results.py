import pickle
import numpy as np
from mllt.core.evaluation.eval_tools import lists_to_arrays, eval_acc, eval_F1
from mllt.core.evaluation.mean_ap import eval_map

# Load all group files
group_files = [
    "work_dirs/LT_coco_resnet50_pfc_DB_pretrain_test_group1/gt_and_results_e8.pkl",
    "work_dirs/LT_coco_resnet50_pfc_DB_pretrain_test_group2/gt_and_results_e8.pkl", 
    "work_dirs/LT_coco_resnet50_pfc_DB_pretrain_test_group3/gt_and_results_e8.pkl"
]

group_names = ['Group 1', 'Group 2', 'Group 3']

# Load and process each group
group_results = {}
all_outputs = []
all_gt_labels = []

for i, file_path in enumerate(group_files):
    with open(file_path, "rb") as f:
        d = pickle.load(f)
    
    gt_labels_list = d[0]['gt_labels']
    outputs_list = d[0]['outputs']
    
    # Convert to arrays
    gt_labels, outputs = lists_to_arrays([gt_labels_list, outputs_list])
    
    group_results[group_names[i]] = {
        'gt_labels': gt_labels,
        'outputs': outputs
    }
    
    print(f"{group_names[i]} - Shape of gt_labels: {gt_labels.shape}")
    print(f"{group_names[i]} - Shape of outputs: {outputs.shape}")
    
    # Store for overall evaluation
    all_outputs.append(outputs)
    all_gt_labels.append(gt_labels)

# Evaluate each group individually
print("\n" + "="*80)
print("INDIVIDUAL GROUP EVALUATION")
print("="*80)

for group_name, data in group_results.items():
    gt_labels = data['gt_labels']
    outputs = data['outputs']
    
    print(f"\nEvaluating {group_name}...")
    
    # mAP
    mAP, APs = eval_map(outputs, gt_labels, None, print_summary=False)
    print(f"mAP: {mAP:.4f}")
    
    # F1 scores
    micro_f1, macro_f1 = eval_F1(outputs, gt_labels)
    print(f"Micro F1: {micro_f1:.4f}, Macro F1: {macro_f1:.4f}")
    
    # Accuracy
    acc, per_cls_acc = eval_acc(outputs, gt_labels)
    print(f"Accuracy: {acc:.4f}")

# Overall evaluation (concatenate all groups)
print("\n" + "="*80)
print("OVERALL EVALUATION (ALL GROUPS COMBINED)")
print("="*80)

# Stack all outputs and gt_labels
all_outputs_array = np.hstack(all_outputs)
all_gt_labels_array = np.hstack(all_gt_labels)

print(f"Overall shape - gt_labels: {all_gt_labels_array.shape}")
print(f"Overall shape - outputs: {all_outputs_array.shape}")

print("Evaluating overall...")
# mAP
overall_mAP, overall_APs = eval_map(all_outputs_array, all_gt_labels_array, None, print_summary=False)
print(f"Overall mAP: {overall_mAP:.4f}")

# F1 scores
overall_micro_f1, overall_macro_f1 = eval_F1(all_outputs_array, all_gt_labels_array)
print(f"Overall Micro F1: {overall_micro_f1:.4f}, Overall Macro F1: {overall_macro_f1:.4f}")

# Accuracy
overall_acc, overall_per_cls_acc = eval_acc(all_outputs_array, all_gt_labels_array)
print(f"Overall Accuracy: {overall_acc:.4f}")

# Summary table
print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)

print(f"{'Group':<10} {'mAP':<8} {'Micro F1':<10} {'Macro F1':<10} {'Accuracy':<10}")
print("-" * 70)

for group_name, data in group_results.items():
    gt_labels = data['gt_labels']
    outputs = data['outputs']
    
    mAP, _ = eval_map(outputs, gt_labels, None, print_summary=False)
    micro_f1, macro_f1 = eval_F1(outputs, gt_labels)
    acc, _ = eval_acc(outputs, gt_labels)
    
    print(f"{group_name:<10} {mAP:<8.4f} {micro_f1:<10.4f} {macro_f1:<10.4f} {acc:<10.4f}")

# Overall row
print("-" * 70)
print(f"{'Overall':<10} {overall_mAP:<8.4f} {overall_micro_f1:<10.4f} {overall_macro_f1:<10.4f} {overall_acc:<10.4f}")