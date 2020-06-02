import argparse
import os

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from tools.fuse_conv_bn import fuse_module

import json
import numpy as np
from tools.metric_polyp import Metric
import time

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.core import wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmdet.datasets.polyp_dataset_test import PolypDatasetTest


# customized evaluation for polyp detection
def bbox2box(bbox):
    box = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
    return box


def get_gt_lists(json_file, category=1):
    json_data = json.load(open(json_file))
    gt_lists = []
    image_ids = []
    dir_list = []

    for image in json_data["images"]:
        image_ids.append(image["id"])
        gt_list = []
        for annotation in json_data["annotations"]:
            if image["id"] == annotation["image_id"] and annotation["category_id"] == category:
                gt_list.append(bbox2box(annotation["bbox"]))
        dir_list.append(image["file_name"])
        gt_lists.append(gt_list)

    return gt_lists, image_ids, dir_list


def polyp_evaluate(results):
    with torch.no_grad():
        results = results
        for thresh in np.linspace(0.2, 0.95, 7):
            # polytest = PolypDatasetTest()
            data_infos = PolypDatasetTest.load_annotations('/data2/dechunwang/dataset/new_polyp_data_combination')
            gt_lists = list()
            for data_info in data_infos:
                gt_lists.append(data_info['ann']['bboxes'])
            # gt_lists, image_ids, _ = get_gt_lists('/data1/zinan_xiong/datasets/dataset/annotation/test_anno.json')
            new_results = list()
            new_scores = list()
            # print('\n', 'results: ', results)
            for result in results:
                # print('\n', 'result: ', result)
                new_result = list()
                new_score = list()
                for bbox in result[1]:
                    # print('\n', 'bbox: ', bbox)
                    if bbox[4] > thresh:
                        new_result.append(bbox2box(bbox[:4]))
                        new_score.append(bbox[4])
                new_results.append(new_result)
                new_scores.append(new_score)

            eval = Metric(visualize=False, mode='center')

            for i in range(len(gt_lists)):
                eval.eval_add_result(gt_lists[i], new_results[i])
            precision, recall, pred_bbox_count = eval.get_result()
            F1 = 2 * (precision * recall) / max((precision + recall), 1e-5)
            F2 = 5 * (precision * recall) / max((4 * precision + recall), 1e-5)
            print("detect time: ", time.time() - st)
            print("Threshold: {:5f}, Prec: {:5f}, Rec: {:5f}, F1: {:5f}, F2: {:5f}, pred_bbox_count: {}".format(thresh,
                                                                                                                precision,
                                                                                                                recall,
                                                                                                                F1, F2,
                                                                                                                pred_bbox_count))

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='arguments in dict')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # test data_loader
    # for i, item in enumerate(data_loader):
    #     print(item)

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_module(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  args.show_score_thr)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.options is None else args.options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            # dataset.evaluate(outputs, args.eval, **kwargs)
            polyp_evaluate(outputs)


if __name__ == '__main__':
    st = time.time()
    main()
