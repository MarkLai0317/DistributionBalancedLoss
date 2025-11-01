import argparse
import os
import tempfile
import os.path as osp
import shutil
import numpy as np
import resource
import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import load_checkpoint, get_dist_info
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
import sys
sys.path.append(os.getcwd())

from mllt.datasets.dataset_wrappers import ConcatDataset, RepeatDataset
from mllt.datasets import build_dataset
from mllt.apis import init_dist
from mllt.datasets import build_dataloader
from mllt.models import build_classifier

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))


def single_gpu_test(model, data_loader, show=False):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset), bar_width=20)
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=not show, **data)
        results.append(result.cpu().numpy())

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test(model, data_loader, tmpdir=None):
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset), bar_width=20)
    for i, data in enumerate(data_loader):
        with torch.no_grad():
             result = model(return_loss=False, rescale=True, **data)
        results.append(result.cpu().numpy())

        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    results = collect_results(results, len(dataset), tmpdir)
    return results


def collect_results(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN,), 32, dtype=torch.uint8, device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def make_dataset_list(cfg, mode='test'):
    """Build dataset list based on mode (train/val/test) - keeping original logic"""
    if mode == 'test':
        # Original test logic
        cfg.data.test.test_mode = True
        test_dataset = build_dataset(cfg.data.test)
        dataset_list = [test_dataset]
        return dataset_list
    elif mode == 'val':
        # Validation mode
        if hasattr(cfg.data, 'val') and cfg.data.val is not None:
            cfg.data.val.test_mode = True
            val_dataset = build_dataset(cfg.data.val)
            dataset_list = [val_dataset]
            return dataset_list
        else:
            raise ValueError("Validation dataset not found in config. Please check cfg.data.val")
    elif mode == 'train':
        # Train mode - only use train dataset
        dataset_list = []
        
        if cfg.data.train.get('dataset', None) is not None:
            train_cfg = cfg.data.train.dataset
        else:
            train_cfg = cfg.data.train
        train_cfg.test_mode = True
        train_cfg.extra_aug = None
        train_cfg.flip_ratio = 0
        train_dataset = build_dataset(train_cfg)

        if isinstance(train_dataset, ConcatDataset):
            train_datasets = train_dataset.datasets
        else:
            train_datasets = [train_dataset]

        for train_dataset in train_datasets:
            dataset_list.append(train_dataset)

        return dataset_list
    else:
        raise ValueError(f"Unsupported mode: {mode}. Supported modes are: train, val, test")


def parse_args():
    parser = argparse.ArgumentParser(description='Model Prediction')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--mode', choices=['train', 'val', 'test'], default='test',
                        help='dataset mode to predict on (default: test)')
    parser.add_argument('--out', help='output result file')
    parser.add_argument('--show', default=True, help='show results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')
    if args.out is None:
        epoch = args.checkpoint.split('.')[-2].split('_')[-1]
        cfg.work_dir = osp.dirname(args.checkpoint)
        args.out = osp.join(cfg.work_dir, 'predictions_{}_e{}.pkl'.format(args.mode, epoch))

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
    rank, _ = get_dist_info()

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None

    # build the dataloader
    dataset_list = make_dataset_list(cfg, args.mode)
    # build the model and load checkpoint
    model = build_classifier(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    savedata = [dict() for _ in range(len(dataset_list))]
    
    for d, dataset in enumerate(dataset_list):
        data_loader = build_dataloader(
            dataset,
            imgs_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)

        # Get ground truth labels
        gt_labels = []
        for i in range(len(dataset)):
            gt_ann = dataset.get_ann_info(i)
            gt_labels.append(gt_ann['labels'])

        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = dataset.CLASSES

        # Run prediction
        if not distributed:
            model = MMDataParallel(model, device_ids=[0])
            outputs = single_gpu_test(model, data_loader, args.show)
        else:
            model = MMDistributedDataParallel(model.cuda())
            outputs = multi_gpu_test(model, data_loader, args.tmpdir)

        if rank == 0:
            savedata[d].update(gt_labels=gt_labels, outputs=np.vstack(outputs))
    
    if rank == 0:
        print('\nSaving predictions to {}'.format(args.out))
        mmcv.dump(savedata, args.out)
        print('Prediction completed!')


if __name__ == '__main__':
    main()