import json
import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import argparse
import glob


def load_evaluation_results(work_dir, epochs, splits=['train', 'val', 'test']):
    """
    Load evaluation results from JSON files
    
    Args:
        work_dir (str): Working directory containing evaluation files
        epochs (list): List of epochs to analyze
        splits (list): Data splits to analyze ['train', 'val', 'test']
    
    Returns:
        dict: Nested dictionary with structure {split: {epoch: evaluation_data}}
    """
    results = {}
    
    for split in splits:
        results[split] = {}
        for epoch in epochs:
            json_file = osp.join(work_dir, f'epoch_{epoch}_predictions_{split}_evaluation.json')
            if osp.exists(json_file):
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    results[split][epoch] = data
            else:
                print(f"Warning: {json_file} not found")
    
    return results


def extract_metrics(results):
    """
    Extract per-class accuracy and average precision metrics
    
    Args:
        results (dict): Results from load_evaluation_results
    
    Returns:
        dict: Extracted metrics with structure {split: {metric: {epoch: values}}}
    """
    metrics = {}
    
    for split in results:
        metrics[split] = {
            'per_class_accuracy': {},
            'average_precision': {},
            'mAP': {},
            'overall_accuracy': {}
        }
        
        for epoch in results[split]:
            data = results[split][epoch]
            if 'datasets' in data and len(data['datasets']) > 0:
                overall = data['datasets'][0]['overall']
                
                metrics[split]['per_class_accuracy'][epoch] = overall['per_class_accuracy']
                metrics[split]['average_precision'][epoch] = overall['average_precision']
                metrics[split]['mAP'][epoch] = overall['mAP']
                metrics[split]['overall_accuracy'][epoch] = overall['accuracy']
    
    return metrics


def plot_per_class_comparison_curves(metrics, class_names, save_dir, work_dir_name):
    """
    Plot individual comparison curves for each class across train/val/test splits
    Creates n separate plots for n classes
    
    Args:
        metrics (dict): Metrics from extract_metrics
        class_names (list): List of class names
        save_dir (str): Directory to save plots
        work_dir_name (str): Name of the experiment for plot titles
    """
    os.makedirs(save_dir, exist_ok=True)
    per_class_dir = osp.join(save_dir, 'per_class_curves')
    os.makedirs(per_class_dir, exist_ok=True)
    
    # Find common epochs across all splits
    all_splits = list(metrics.keys())
    common_epochs = None
    for split in all_splits:
        if metrics[split]['per_class_accuracy']:
            epochs = set(metrics[split]['per_class_accuracy'].keys())
            if common_epochs is None:
                common_epochs = epochs
            else:
                common_epochs = common_epochs.intersection(epochs)
    
    if not common_epochs:
        print("No common epochs found across splits")
        return
    
    common_epochs = sorted(list(common_epochs))
    
    # Color scheme for different splits
    split_colors = {'train': 'blue', 'val': 'orange', 'test': 'green'}
    
    # Create a plot for each class
    for class_idx, class_name in enumerate(class_names):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Per-class Accuracy across splits
        for split in all_splits:
            if split in metrics and metrics[split]['per_class_accuracy']:
                accuracies = []
                epochs_with_data = []
                for epoch in common_epochs:
                    if epoch in metrics[split]['per_class_accuracy']:
                        accuracies.append(metrics[split]['per_class_accuracy'][epoch][class_idx])
                        epochs_with_data.append(epoch)
                
                if accuracies:
                    color = split_colors.get(split, 'gray')
                    ax1.plot(epochs_with_data, accuracies, 
                            color=color, label=f'{split.upper()}', 
                            linewidth=2, marker='o', markersize=4)
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.set_title(f'{work_dir_name} - Class "{class_name}" Accuracy Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Plot 2: Average Precision across splits
        for split in all_splits:
            if split in metrics and metrics[split]['average_precision']:
                aps = []
                epochs_with_data = []
                for epoch in common_epochs:
                    if epoch in metrics[split]['average_precision']:
                        aps.append(metrics[split]['average_precision'][epoch][class_idx])
                        epochs_with_data.append(epoch)
                
                if aps:
                    color = split_colors.get(split, 'gray')
                    ax2.plot(epochs_with_data, aps, 
                            color=color, label=f'{split.upper()}', 
                            linewidth=2, marker='s', markersize=4)
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Average Precision')
        ax2.set_title(f'{work_dir_name} - Class "{class_name}" Average Precision Curves')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        
        # Save the plot
        safe_class_name = class_name.replace('/', '_').replace(' ', '_')
        class_plot_path = osp.join(per_class_dir, f'class_{class_idx:02d}_{safe_class_name}.png')
        plt.savefig(class_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Class {class_idx} ({class_name}) plot saved: {class_plot_path}")
    
    print(f"All per-class comparison plots saved in: {per_class_dir}")


def plot_per_class_summary_grid(metrics, class_names, save_dir, work_dir_name):
    """
    Create a summary grid showing final epoch performance for all classes
    
    Args:
        metrics (dict): Metrics from extract_metrics
        class_names (list): List of class names
        save_dir (str): Directory to save plots
        work_dir_name (str): Name of the experiment for plot titles
    """
    all_splits = list(metrics.keys())
    if not all_splits:
        return
    
    # Find the latest epoch with complete data
    latest_epoch = None
    for split in all_splits:
        if metrics[split]['per_class_accuracy']:
            epochs = list(metrics[split]['per_class_accuracy'].keys())
            if epochs:
                split_latest = max(epochs)
                if latest_epoch is None or split_latest > latest_epoch:
                    latest_epoch = split_latest
    
    if latest_epoch is None:
        return
    
    num_classes = len(class_names)
    num_splits = len(all_splits)
    
    # Create summary data
    acc_data = np.zeros((num_splits, num_classes))
    ap_data = np.zeros((num_splits, num_classes))
    
    for split_idx, split in enumerate(all_splits):
        if (split in metrics and 
            latest_epoch in metrics[split]['per_class_accuracy'] and
            latest_epoch in metrics[split]['average_precision']):
            
            acc_data[split_idx, :] = metrics[split]['per_class_accuracy'][latest_epoch]
            ap_data[split_idx, :] = metrics[split]['average_precision'][latest_epoch]
    
    # Create the grid plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(12, num_classes * 0.8), 10))
    
    # Accuracy heatmap
    im1 = ax1.imshow(acc_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax1.set_xticks(range(num_classes))
    ax1.set_xticklabels([name[:15] + '...' if len(name) > 15 else name for name in class_names], 
                        rotation=45, ha='right')
    ax1.set_yticks(range(num_splits))
    ax1.set_yticklabels([split.upper() for split in all_splits])
    ax1.set_title(f'{work_dir_name} - Per-class Accuracy (Epoch {latest_epoch})')
    
    # Add text annotations
    for i in range(num_splits):
        for j in range(num_classes):
            text = ax1.text(j, i, f'{acc_data[i, j]:.3f}', 
                           ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im1, ax=ax1)
    
    # Average Precision heatmap
    im2 = ax2.imshow(ap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax2.set_xticks(range(num_classes))
    ax2.set_xticklabels([name[:15] + '...' if len(name) > 15 else name for name in class_names], 
                        rotation=45, ha='right')
    ax2.set_yticks(range(num_splits))
    ax2.set_yticklabels([split.upper() for split in all_splits])
    ax2.set_title(f'{work_dir_name} - Average Precision (Epoch {latest_epoch})')
    
    # Add text annotations
    for i in range(num_splits):
        for j in range(num_classes):
            text = ax2.text(j, i, f'{ap_data[i, j]:.3f}', 
                           ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im2, ax=ax2)
    plt.tight_layout()
    
    summary_path = osp.join(save_dir, f'per_class_summary_grid.png')
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Per-class summary grid saved: {summary_path}")


def plot_learning_curves(metrics, class_names, save_dir, work_dir_name):
    """
    Plot learning curves for different metrics
    
    Args:
        metrics (dict): Metrics from extract_metrics
        class_names (list): List of class names
        save_dir (str): Directory to save plots
        work_dir_name (str): Name of the experiment for plot titles
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Set up color palette
    colors = sns.color_palette("husl", len(class_names))
    
    for split in metrics:
        if not metrics[split]['per_class_accuracy']:
            continue
            
        epochs = sorted(metrics[split]['per_class_accuracy'].keys())
        num_classes = len(class_names)
        
        # Plot 1: Per-class Accuracy Learning Curves
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for class_idx in range(num_classes):
            accuracies = [metrics[split]['per_class_accuracy'][epoch][class_idx] 
                         for epoch in epochs]
            ax.plot(epochs, accuracies, label=class_names[class_idx], 
                   color=colors[class_idx], linewidth=2, marker='o', markersize=4)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Per-class Accuracy')
        ax.set_title(f'{work_dir_name} - Per-class Accuracy Learning Curves ({split.upper()})')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        acc_path = osp.join(save_dir, f'{split}_per_class_accuracy.png')
        plt.savefig(acc_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Per-class accuracy plot saved: {acc_path}")
        
        # Plot 2: Average Precision Learning Curves
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for class_idx in range(num_classes):
            aps = [metrics[split]['average_precision'][epoch][class_idx] 
                  for epoch in epochs]
            ax.plot(epochs, aps, label=class_names[class_idx], 
                   color=colors[class_idx], linewidth=2, marker='s', markersize=4)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Average Precision')
        ax.set_title(f'{work_dir_name} - Average Precision Learning Curves ({split.upper()})')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        ap_path = osp.join(save_dir, f'{split}_average_precision.png')
        plt.savefig(ap_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Average precision plot saved: {ap_path}")
        
        # Plot 3: Overall Metrics (mAP and Overall Accuracy)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # mAP
        maps = [metrics[split]['mAP'][epoch] for epoch in epochs]
        ax1.plot(epochs, maps, 'b-o', linewidth=2, markersize=6)
        ax1.set_ylabel('mAP')
        ax1.set_title(f'{work_dir_name} - Overall Metrics ({split.upper()})')
        ax1.grid(True, alpha=0.3)
        
        # Overall Accuracy
        accuracies = [metrics[split]['overall_accuracy'][epoch] for epoch in epochs]
        ax2.plot(epochs, accuracies, 'r-s', linewidth=2, markersize=6)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Overall Accuracy')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        overall_path = osp.join(save_dir, f'{split}_overall_metrics.png')
        plt.savefig(overall_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Overall metrics plot saved: {overall_path}")
        
        # Plot 4: Heatmap of per-class metrics at final epoch
        if epochs:
            final_epoch = max(epochs)
            final_acc = metrics[split]['per_class_accuracy'][final_epoch]
            final_ap = metrics[split]['average_precision'][final_epoch]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Per-class accuracy heatmap
            acc_data = np.array(final_acc).reshape(1, -1)
            im1 = ax1.imshow(acc_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
            ax1.set_xticks(range(len(class_names)))
            ax1.set_xticklabels(class_names, rotation=45, ha='right')
            ax1.set_yticks([0])
            ax1.set_yticklabels([f'Epoch {final_epoch}'])
            ax1.set_title(f'Per-class Accuracy ({split.upper()})')
            plt.colorbar(im1, ax=ax1)
            
            # Average precision heatmap
            ap_data = np.array(final_ap).reshape(1, -1)
            im2 = ax2.imshow(ap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
            ax2.set_xticks(range(len(class_names)))
            ax2.set_xticklabels(class_names, rotation=45, ha='right')
            ax2.set_yticks([0])
            ax2.set_yticklabels([f'Epoch {final_epoch}'])
            ax2.set_title(f'Average Precision ({split.upper()})')
            plt.colorbar(im2, ax=ax2)
            
            plt.tight_layout()
            
            heatmap_path = osp.join(save_dir, f'{split}_final_epoch_heatmap.png')
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Final epoch heatmap saved: {heatmap_path}")


def plot_comparison_across_splits(metrics, class_names, save_dir, work_dir_name):
    """
    Plot comparison of metrics across different splits (train/val/test)
    
    Args:
        metrics (dict): Metrics from extract_metrics
        class_names (list): List of class names
        save_dir (str): Directory to save plots
        work_dir_name (str): Name of the experiment for plot titles
    """
    splits = list(metrics.keys())
    if len(splits) < 2:
        return
    
    # Find common epochs across all splits
    common_epochs = None
    for split in splits:
        if metrics[split]['mAP']:
            epochs = set(metrics[split]['mAP'].keys())
            if common_epochs is None:
                common_epochs = epochs
            else:
                common_epochs = common_epochs.intersection(epochs)
    
    if not common_epochs:
        return
    
    common_epochs = sorted(list(common_epochs))
    
    # Plot mAP comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    for split in splits:
        maps = [metrics[split]['mAP'][epoch] for epoch in common_epochs]
        ax.plot(common_epochs, maps, '-o', label=split.upper(), linewidth=2, markersize=6)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mAP')
    ax.set_title(f'{work_dir_name} - mAP Comparison Across Splits')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    comparison_path = osp.join(save_dir, f'mAP_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plot saved to: {comparison_path}")


def analyze_single_experiment(work_dir, epochs=None, splits=['train', 'val', 'test'], save_dir=None):
    """
    Analyze a single experiment directory
    
    Args:
        work_dir (str): Path to experiment directory
        epochs (list): List of epochs to analyze. If None, auto-detect from files
        splits (list): Data splits to analyze
        save_dir (str): Directory to save plots. If None, save in work_dir/analysis
    """
    if not osp.exists(work_dir):
        print(f"Error: Work directory {work_dir} does not exist")
        return
    
    # Auto-detect epochs if not provided
    if epochs is None:
        epochs = []
        for file in os.listdir(work_dir):
            if file.startswith('epoch_') and file.endswith('_evaluation.json'):
                try:
                    epoch = int(file.split('_')[1])
                    epochs.append(epoch)
                except:
                    continue
        epochs = sorted(list(set(epochs)))
        print(f"Auto-detected epochs: {epochs}")
    
    if not epochs:
        print("No evaluation files found")
        return
    
    # Set save directory
    if save_dir is None:
        save_dir = osp.join(work_dir, 'analysis')
    
    work_dir_name = osp.basename(work_dir)
    
    # Load results
    print("Loading evaluation results...")
    results = load_evaluation_results(work_dir, epochs, splits)
    
    # Extract metrics
    print("Extracting metrics...")
    metrics = extract_metrics(results)
    
    # Get class names from first available result
    class_names = None
    for split in results:
        for epoch in results[split]:
            if 'class_names' in results[split][epoch]:
                class_names = results[split][epoch]['class_names']
                break
        if class_names:
            break
    
    if class_names is None:
        print("Error: Could not find class names in evaluation files")
        return
    
    print(f"Found {len(class_names)} classes: {class_names}")
    
    # Create plots
    print("Creating learning curve plots...")
    plot_learning_curves(metrics, class_names, save_dir, work_dir_name)
    
    print("Creating comparison plots...")
    plot_comparison_across_splits(metrics, class_names, save_dir, work_dir_name)
    
    print("Creating per-class comparison curves...")
    plot_per_class_comparison_curves(metrics, class_names, save_dir, work_dir_name)
    
    print("Creating per-class summary grid...")
    plot_per_class_summary_grid(metrics, class_names, save_dir, work_dir_name)
    
    print(f"Analysis complete! Results saved in: {save_dir}")


def main():
    parser = argparse.ArgumentParser(description='Analyze learning curves from evaluation JSON files')
    parser.add_argument('work_dir', help='Path to experiment work directory')
    parser.add_argument('--epochs', nargs='+', type=int, default=None,
                       help='Epochs to analyze (default: auto-detect)')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'],
                       help='Data splits to analyze')
    parser.add_argument('--save_dir', default=None,
                       help='Directory to save plots (default: work_dir/analysis)')
    
    args = parser.parse_args()
    
    analyze_single_experiment(args.work_dir, args.epochs, args.splits, args.save_dir)


if __name__ == '__main__':
    # Example usage for multiple experiments
    # work_dirs = [
    #     '/tmp2/bamboochen/DistributionBalancedLoss/work_dirs/LT_coco_resnet50_pfc_DB_pretrain_test',
    #     '/tmp2/bamboochen/DistributionBalancedLoss/work_dirs/LT_coco_resnet50_pfc_DB_pretrain_test_group1',
    #     # Add more experiment directories as needed
    # ]
    # 
    # for work_dir in work_dirs:
    #     if osp.exists(work_dir):
    #         print(f"\nAnalyzing {work_dir}...")
    #         analyze_single_experiment(work_dir)
    
    main()