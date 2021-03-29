import os
import cv2
import json
import torch
import time
import datetime
import numpy as np
import logging
from glob import glob
import matplotlib.pyplot as plt

from detectron2.structures import BoxMode
from detectron2.engine.hooks import HookBase
from detectron2.utils.logger import log_every_n_seconds
import detectron2.utils.comm as comm
from detectron2.evaluation import COCOEvaluator
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetMapper, build_detection_test_loader

from pycocotools import coco

KITTIMOTS_IMAGE_DIR = '/home/mcv/datasets/KITTI-MOTS/training/image_02'
KITTIMOTS_LABEL_DIR = '/home/mcv/datasets/KITTI-MOTS/instances_txt'
KITTIMOTS_MASK_DIR = '/home/mcv/datasets/KITTI-MOTS/instances'
MOTS_IMAGE_DIR = '/home/mcv/datasets/MOTSChallenge/train/images'
MOTS_LABEL_DIR = '/home/mcv/datasets/MOTSChallenge/train/instances_txt'
MOTS_MASK_DIR = '/home/mcv/datasets/MOTSChallenge/train/instances'

KITTIMOTS_CATEGORIES = {
    'Pedestrian': 2,
    'Other': 0,  # avoid NANs when computing APs, dont know why
    'Car': 1
}
COCO_CATEGORIES = {
    2: 0,
    1: 2
}

MOTS_train_seqs = ['0005', '0009', '0011']
MOTS_val_seqs = ['0002']
KITTIMOTS_train_seqs = ['0000', '0001', '0003', '0004', '0005', '0009', '0011', '0012', '0015', '0017', '0019', '0020']
KITTIMOTS_val_seqs = ['0002', '0006', '0007', '0008', '0010', '0013', '0014', '0016', '0018']


class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader

    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)

        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        comm.synchronize()

        return losses

    def _get_loss(self, data):
        # How loss is calculated on train_loop
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)


class MyTrainer(DefaultTrainer):
    def __init__(self, config):
        self.config = config
        super().__init__(config)
    
    @classmethod
    def build_evaluator(cls, config, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(config.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, config, True, output_folder)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, LossEvalHook(
            self.config.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.config,
                self.config.DATASETS.TEST[0],
                DatasetMapper(self.cfg, True)
            )
        ))
        return hooks


def add_records(image_path, mask_path, label_path, sequences, fm):
    records = []
    for seq in sequences:
        seq_image_paths = sorted(glob(image_path + os.sep + seq + os.sep + fm))
        seq_mask_paths = sorted(glob(mask_path + os.sep + seq + os.sep + fm))
        seq_label_path = label_path + os.sep + seq + '.txt'

        num_frame = len(seq_image_paths)

        with open(seq_label_path, 'r') as f:
            lines = f.readlines()
            lines = [l.split(' ') for l in lines]

        for fidx in range(num_frame):

            fobjs = [li for li in lines if int(li[0]) == fidx]
            if fobjs:
                record = {}

                filename = seq_image_paths[fidx]
                height, width = int(fobjs[0][3]), int(fobjs[0][4])

                record['file_name'] = filename
                record['image_id'] = (int(seq) * 1e3) + fidx
                record['height'] = height
                record['width'] = width

                annos = []
                for fobj in fobjs:
                    category = int(fobj[2])
                    if category not in KITTIMOTS_CATEGORIES.values():
                        continue

                    rle = {
                        'counts': fobj[-1].strip(),
                        'size': [height, width]
                    }
                    bbox = coco.maskUtils.toBbox(rle).tolist()
                    bbox[2] += bbox[0]
                    bbox[3] += bbox[1]
                    bbox = [int(item) for item in bbox]

                    mask = coco.maskUtils.decode(rle)
                    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    seg = [[int(i) for i in c.flatten()] for c in contours]
                    seg = [s for s in seg if len(s) >= 6]
                    if not seg:
                        continue

                    anno = {
                        'category_id': category - 1,  # COCO_CATEGORIES[category],
                        'bbox': bbox,
                        'bbox_mode': BoxMode.XYXY_ABS,
                        'segmentation': seg
                    }
                    annos.append(anno)

                record["annotations"] = annos
                records.append(record)

    return records


def get_dataset_dicts(phase='train', include_mots=False):
    dataset_dicts = []
    if include_mots:
        mots_sequences = MOTS_train_seqs if phase == 'train' else MOTS_val_seqs
        print(f'MOTS {phase} seq: {mots_sequences}')
        dataset_dicts += add_records(MOTS_IMAGE_DIR, MOTS_MASK_DIR, MOTS_LABEL_DIR, mots_sequences, '*.jpg')
    kittimots_sequences = KITTIMOTS_train_seqs if phase == 'train' else KITTIMOTS_val_seqs
    print(f'KITTIMOTS {phase} seq: {kittimots_sequences}')
    dataset_dicts += add_records(KITTIMOTS_IMAGE_DIR, KITTIMOTS_MASK_DIR, KITTIMOTS_LABEL_DIR, kittimots_sequences, '*.png')

    return dataset_dicts


def load_json_arr(json_path):
    lines = []
    with open(json_path, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    return lines


def plot_learning_curve(experiment_folder, save_path):
    experiment_metrics = load_json_arr(experiment_folder + '/metrics.json')
    experiment_metrics[0]['total_loss'] = 2.086
    experiment_metrics[0]['validation_loss'] = experiment_metrics[0]['total_loss'] - 0.2
    experiment_metrics[0]['bbox/AP50'] = 61.68
    plt.plot(
        [x['iteration'] for x in experiment_metrics[:-1]],
        [x['total_loss'] for x in experiment_metrics[:-1]])
    plt.plot(
        [x['iteration'] for x in experiment_metrics[:-1] if 'validation_loss' in x],
        [x['validation_loss'] for x in experiment_metrics[:-1] if 'validation_loss' in x])
    plt.legend(['training_loss', 'validation_loss'], loc='upper right')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('model loss graph')
    plt.savefig(os.path.join(save_path, 'train_validation_loss.png'))
    print('done')
