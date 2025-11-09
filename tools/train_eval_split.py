#!/usr/bin/env python3
"""
Split img_id.txt into train and eval sets and visualize class distribution
Input: img_id.txt (training image filenames)
Output: train_img_id.txt (85%) and eval_img_id.txt (15%) + distribution plots
"""

import argparse
import random
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import mmcv

def load_class_data(img_ids, data_file, idx2img_id, gt_labels, class_names):
    """
    Load class distribution data for given image IDs
    
    Args:
        img_ids (list): List of image IDs
        data_file (str): Path to the terse_gt.pkl file
        idx2img_id (dict): Mapping from index to image ID
        gt_labels (np.array): Ground truth labels
        class_names (list): List of class names
    
    Returns:
        np.array: Class counts for the subset
    """
    # Create reverse mapping from img_id to index
    img_id2idx = {img_id: idx for idx, img_id in idx2img_id.items()}
    
    # Get indices for our image IDs
    indices = []
    for img_id in img_ids:
        if img_id in img_id2idx:
            indices.append(img_id2idx[img_id])
    
    if not indices:
        print("Warning: No matching images found in dataset")
        return np.zeros(len(class_names))
    
    # Extract labels for selected images
    selected_labels = gt_labels[indices]
    
    # Count samples per class
    class_counts = np.sum(selected_labels, axis=0)
    
    return class_counts

def plot_class_distribution(train_counts, eval_counts, class_names, output_dir, dataset_name="Dataset"):
    """
    Plot class distribution comparison between train and eval sets
    
    Args:
        train_counts (np.array): Training set class counts
        eval_counts (np.array): Evaluation set class counts
        class_names (list): List of class names
        output_dir (str): Output directory for plots
        dataset_name (str): Name of the dataset for plot titles
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate total samples
    total_train = np.sum(train_counts)
    total_eval = np.sum(eval_counts)
    
    # Calculate percentages
    train_pct = train_counts / total_train * 100 if total_train > 0 else np.zeros_like(train_counts)
    eval_pct = eval_counts / total_eval * 100 if total_eval > 0 else np.zeros_like(eval_counts)
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    x_pos = np.arange(len(class_names))
    width = 0.35
    
    # Plot 1: Absolute counts comparison
    bars1 = ax1.bar(x_pos - width/2, train_counts, width, label='Train', alpha=0.8, color='skyblue')
    bars2 = ax1.bar(x_pos + width/2, eval_counts, width, label='Eval', alpha=0.8, color='lightcoral')
    
    ax1.set_xlabel('Classes')
    ax1.set_ylabel('Sample Count')
    ax1.set_title(f'{dataset_name} - Absolute Sample Counts')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(class_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(train_counts) * 0.01,
                    f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(eval_counts) * 0.01,
                    f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Percentage comparison
    bars3 = ax2.bar(x_pos - width/2, train_pct, width, label='Train', alpha=0.8, color='skyblue')
    bars4 = ax2.bar(x_pos + width/2, eval_pct, width, label='Eval', alpha=0.8, color='lightcoral')
    
    ax2.set_xlabel('Classes')
    ax2.set_ylabel('Percentage (%)')
    ax2.set_title(f'{dataset_name} - Percentage Distribution')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(class_names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add percentage labels on bars
    for bar in bars3:
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    for bar in bars4:
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, f'{dataset_name}_class_distribution_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Class distribution plot saved: {plot_path}")
    
    # Print statistical summary
    print("\n📈 Statistical Summary:")
    print(f"Total train samples: {total_train}")
    print(f"Total eval samples: {total_eval}")
    print(f"Overall train/eval ratio: {total_train/total_eval:.3f}" if total_eval > 0 else "Overall train/eval ratio: inf")
    
    # Calculate correlation coefficient
    if len(train_counts) > 1 and np.std(train_counts) > 0 and np.std(eval_counts) > 0:
        correlation = np.corrcoef(train_counts, eval_counts)[0, 1]
        print(f"Train-Eval correlation: {correlation:.3f}")
    
    return plot_path

def split_train_eval(input_file, output_dir, train_ratio=0.85, seed=42, dataset_pkl=None):
    """
    Split img_id.txt into train and eval sets and plot class distributions
    
    Args:
        input_file (str): Path to img_id.txt
        output_dir (str): Output directory for train and eval files
        train_ratio (float): Ratio for training set (default: 0.85)
        seed (int): Random seed for reproducibility
        dataset_pkl (str): Path to terse_gt.pkl file for class distribution analysis
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Read all image IDs
    with open(input_file, 'r') as f:
        img_ids = [line.strip() for line in f if line.strip()]
    
    print(f"Total images: {len(img_ids)}")
    
    # Shuffle the list
    random.shuffle(img_ids)
    
    # Calculate split point
    train_count = int(len(img_ids) * train_ratio)
    eval_count = len(img_ids) - train_count
    
    # Split the data
    train_ids = img_ids[:train_count]
    eval_ids = img_ids[train_count:]
    
    print(f"Train images: {len(train_ids)} ({len(train_ids)/len(img_ids)*100:.1f}%)")
    print(f"Eval images: {len(eval_ids)} ({len(eval_ids)/len(img_ids)*100:.1f}%)")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Write train set
    train_file = os.path.join(output_dir, 'train_img_id.txt')
    with open(train_file, 'w') as f:
        for img_id in train_ids:
            f.write(f"{img_id}\n")
    
    # Write eval set
    eval_file = os.path.join(output_dir, 'eval_img_id.txt')
    with open(eval_file, 'w') as f:
        for img_id in eval_ids:
            f.write(f"{img_id}\n")
    
    print(f"✅ Train set saved to: {train_file}")
    print(f"✅ Eval set saved to: {eval_file}")
    
    # Plot class distribution if dataset_pkl is provided
    if dataset_pkl and os.path.exists(dataset_pkl):
        print(f"\n📊 Analyzing class distribution using {dataset_pkl}...")
        try:
            # Load dataset
            data = mmcv.load(dataset_pkl)
            gt_labels = np.array(data['gt_labels'])
            idx2img_id = data['idx2img_id']
            
            # Get class names
            class_names = data.get('class_names', [f'Class_{i}' for i in range(gt_labels.shape[1])])
            
            # Get class counts for train and eval sets
            train_counts = load_class_data(train_ids, dataset_pkl, idx2img_id, gt_labels, class_names)
            eval_counts = load_class_data(eval_ids, dataset_pkl, idx2img_id, gt_labels, class_names)
            
            # Create dataset name from input directory
            dataset_name = os.path.basename(os.path.dirname(input_file))
            if not dataset_name:
                dataset_name = "Dataset"
            
            # Plot distribution comparison
            plot_class_distribution(train_counts, eval_counts, class_names, output_dir, dataset_name)
            
        except Exception as e:
            print(f"⚠️  Warning: Could not analyze class distribution: {e}")
    elif dataset_pkl:
        print(f"⚠️  Warning: Dataset file {dataset_pkl} not found. Skipping distribution analysis.")
    
    return train_file, eval_file

def main():
    parser = argparse.ArgumentParser(description='Split img_id.txt into train and eval sets')
    parser.add_argument('--input', required=True, 
                        help='Path to input img_id.txt file')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for train and eval files')
    parser.add_argument('--train-ratio', type=float, default=0.85,
                        help='Ratio for training set (default: 0.85)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--dataset-pkl', type=str, default=None,
                        help='Path to terse_gt.pkl file for class distribution analysis')
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.input):
        print(f"❌ Error: Input file {args.input} does not exist")
        return 1
    
    # Validate train ratio
    if not 0 < args.train_ratio < 1:
        print(f"❌ Error: Train ratio must be between 0 and 1, got {args.train_ratio}")
        return 1
    
    try:
        split_train_eval(args.input, args.output_dir, args.train_ratio, args.seed, args.dataset_pkl)
        print("✅ Train/eval split completed successfully!")
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == '__main__':
    exit(main())