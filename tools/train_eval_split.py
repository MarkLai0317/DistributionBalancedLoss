#!/usr/bin/env python3
"""
Split img_id.txt into train and eval sets
Input: img_id.txt (training image filenames)
Output: train_img_id.txt (85%) and eval_img_id.txt (15%)
"""

import argparse
import random
import os

def split_train_eval(input_file, output_dir, train_ratio=0.85, seed=42):
    """
    Split img_id.txt into train and eval sets
    
    Args:
        input_file (str): Path to img_id.txt
        output_dir (str): Output directory for train and eval files
        train_ratio (float): Ratio for training set (default: 0.85)
        seed (int): Random seed for reproducibility
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
        split_train_eval(args.input, args.output_dir, args.train_ratio, args.seed)
        print("✅ Train/eval split completed successfully!")
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == '__main__':
    exit(main())