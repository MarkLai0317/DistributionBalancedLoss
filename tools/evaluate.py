import argparse
import os
import os.path as osp
import numpy as np
import mmcv
import sys
sys.path.append(os.getcwd())

from mllt.datasets import build_dataset
from mllt.core.evaluation.eval_tools import lists_to_arrays, eval_acc, eval_F1
from mllt.core.evaluation.mean_ap import eval_map


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Predictions')
    parser.add_argument('config', help='config file path')
    parser.add_argument('predictions', help='prediction results file (.pkl)')
    parser.add_argument('--eval', type=str, nargs='+', choices=['mAP', 'multiple'],
                        default=['multiple'], help='eval metrics')
    parser.add_argument('--testset_only', type=bool, default=True, help='only eval test set')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    
    # Load predictions
    if not osp.exists(args.predictions):
        raise FileNotFoundError(f'Predictions file not found: {args.predictions}')
    
    savedata = mmcv.load(args.predictions)
    print(f'Loaded predictions from: {args.predictions}')
    
    # Build dataset for class information
    dataset = build_dataset(cfg.data.test)
    
    display_dict = {}
    eval_metrics = args.eval
    display_dict['class'] = dataset.CLASSES
    
    for i, data in enumerate(savedata):
        if args.testset_only and i > 0:  # test-set only
            break
            
        gt_labels = data['gt_labels']
        outputs = data['outputs']
        
        print(f'\nEvaluating dataset {i}:')
        print(f'GT labels shape: {np.array(gt_labels).shape}')
        print(f'Outputs shape: {outputs.shape}')

        gt_labels, outputs = lists_to_arrays([gt_labels, outputs])
        
        # Debug: Check output range
        print(f'Outputs range: [{np.min(outputs):.4f}, {np.max(outputs):.4f}]')
        
        print('Starting evaluate {}'.format(' and '.join(eval_metrics)))
        
        for eval_metric in eval_metrics:
            if eval_metric == 'mAP':
                mAP, APs = eval_map(outputs, gt_labels, None, print_summary=True)
                display_dict[f'APs_{i}'] = APs
                
            elif eval_metric == 'multiple':
                metrics = []
                
                # Evaluate for each split (head, mid, tail)
                for split, selected in dataset.class_split.items():
                    selected = list(selected)
                    
                    # Skip empty splits
                    if len(selected) == 0:
                        continue
                    
                    selected_outputs = outputs[:, selected]
                    selected_gt_labels = gt_labels[:, selected]
                    classes = np.asarray(dataset.CLASSES)[selected]
                    
                    # Calculate metrics for this split
                    mAP, APs = eval_map(selected_outputs, selected_gt_labels, classes, print_summary=False)
                    micro_f1, macro_f1 = eval_F1(selected_outputs, selected_gt_labels)
                    acc, per_cls_acc = eval_acc(selected_outputs, selected_gt_labels)
                    metrics.append([split, mAP, micro_f1, macro_f1, acc])
                
                # Calculate overall metrics
                mAP, APs = eval_map(outputs, gt_labels, dataset, print_summary=False)
                micro_f1, macro_f1 = eval_F1(outputs, gt_labels)
                acc, per_cls_acc = eval_acc(outputs, gt_labels)
                metrics.append(['Total', mAP, micro_f1, macro_f1, acc])
                
                # Print results
                print('\n' + '='*70)
                print('EVALUATION RESULTS')
                print('='*70)
                for split, mAP, micro_f1, macro_f1, acc in metrics:
                    print('Split:{:>6s} mAP:{:.4f}  acc:{:.4f}  micro:{:.4f}  macro:{:.4f}'.format(
                        split, mAP, acc, micro_f1, macro_f1))
                print('='*70)


if __name__ == '__main__':
    main()