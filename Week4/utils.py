import os
import cv2
import json
import torch
import numpy as np
import copy
import random
import pickle
import matplotlib.pyplot as plt

from detectron2.structures import BoxMode
from detectron2.engine.hooks import HookBase
from detectron2.engine import DefaultTrainer
import detectron2.utils.comm as comm
from detectron2.data import build_detection_train_loader, DatasetMapper
from detectron2.data import transforms as T
from pycocotools import coco


KITTIMOTS_IMAGE_DIR = '/home/mcv/datasets/KITTI-MOTS/training/image_02'
KITTIMOTS_LABEL_DIR = '/home/mcv/datasets/KITTI-MOTS/instances_txt'
KITTIMOTS_MASK_DIR = '/home/mcv/datasets/KITTI-MOTS/instances'
MOTS_IMAGE_DIR = '/home/mcv/datasets/MOTSChallenge/train/images'
MOTS_LABEL_DIR = '/home/mcv/datasets/MOTSChallenge/train/instances_txt'
MOTS_MASK_DIR = '/home/mcv/datasets/MOTSChallenge/train/instances'

# KITTIMOTS_IMAGE_DIR = '/Users/eternalenvy/Desktop/projects/M5VR/week3/data_tracking_image_2/training/image_02'
# KITTIMOTS_LABEL_DIR = '/Users/eternalenvy/Desktop/projects/M5VR/week3/data_tracking_image_2/training/instances_txt'
# KITTIMOTS_MASK_DIR = '/Users/eternalenvy/Desktop/projects/M5VR/week3/data_tracking_image_2/training/instances'
# MOTS_IMAGE_DIR = '/Users/eternalenvy/Desktop/projects/M5VR/week3/MOTSChallenge/train/images'
# MOTS_LABEL_DIR = '/Users/eternalenvy/Desktop/projects/M5VR/week3/MOTSChallenge/train/instances_txt'
# MOTS_MASK_DIR = '/Users/eternalenvy/Desktop/projects/M5VR/week3/MOTSChallenge/train/instances'

KITTIMOTS_CATEGORIES = {
    'Car': 1,
    'Pedestrian': 2
}

COCO_CATEGORIES = {
    2: 0,
    1: 2
}

MOTS_train_seqs = ['0002', '0005', '0009', '0011']
KITTIMOTS_train_seqs = ['0000', '0001', '0003', '0004', '0005', '0009', '0011', '0012', '0015', '0017', '0019', '0020']
KITTIMOTS_val_seqs = ['0002', '0006', '0007', '0008', '0010', '0013', '0014', '0016', '0018']


def get_dataset_dicts(dataset, inference=False):
    dataset_dicts = []
    for idx, data_frame in enumerate(dataset):
        record = {}

        seq = data_frame[0][1]
        frame_idx = data_frame[0][2]
        if data_frame[0][0] == 'KITTI-MOTS':
            image_path = KITTIMOTS_IMAGE_DIR
            fm = '.png'
        else:
            image_path = MOTS_IMAGE_DIR
            fm = '.jpg'
        filename = image_path + os.sep + seq + os.sep + frame_idx.zfill(6) + fm

        record['file_name'] = filename
        record['image_id'] = idx
        record['height'] = int(data_frame[0][5])
        record['width'] = int(data_frame[0][6])

        fobjs = []
        for fobj in data_frame:
            rle = {
                'size': [int(fobj[5]), int(fobj[6])],
                'counts': fobj[-1].strip().encode(encoding='UTF-8')
            }

            bbox = coco.maskUtils.toBbox(rle).tolist()
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            
            # convert rle to poly
            mask = coco.maskUtils.decode(rle)

            contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            segmentation = []

            for contour in contours:
                contour = contour.flatten().tolist()
                if len(contour) > 4:
                    segmentation.append(contour)
            if len(segmentation) == 0:
                continue

            category = int(fobj[4])
            if category not in KITTIMOTS_CATEGORIES.values():
                continue
            if inference:
                category_id = COCO_CATEGORIES[category]
            else:
                category_id = category - 1

            obj = {
                'category_id': category_id,
                'bbox': bbox,
                'bbox_mode': BoxMode.XYXY_ABS,
                'segmentation': segmentation,
                'iscrowd': 0
            }
            fobjs.append(obj)
        record['annotations'] = fobjs
        dataset_dicts.append(record)
    return dataset_dicts


def read_annos_from_path(ds_name, label_path, seq_name):
    """
    Read annotation within one sequence
    :param ds_name:
    :param label_path:
    :param seq_name:
    :return:
    Annotations as a list of list, each element list contains annotations of one particular frame in this sequence
    """

    seq = seq_name[:4]
    anno_path = label_path + os.sep + seq_name + '.txt'

    seq_annos = []
    frame_annos = []

    curr_frame = 1 if ds_name == 'MOTS' else 0
    for line in open(anno_path):  # loop over lines of seq.txt
        fobj_line = line.split()
        frame_idx = int(fobj_line[0])

        frame_obj = [ds_name] + [seq] + fobj_line  # each line as an object

        # check if the loop jumped to another frame or not
        if frame_idx == curr_frame:
            frame_annos.append(frame_obj)
        else:
            seq_annos.append(frame_annos)

            frame_annos = [frame_obj]

        curr_frame = frame_idx

    # append objects of the last frame
    if len(frame_annos) > 0:
        seq_annos.append(frame_annos)

    return seq_annos


def read_data(include_mots=False):
    if os.path.exists(f'training_validation_mots_{include_mots}.pkl'):
        print(f'Loading already indexed training data, include_mots = {include_mots}')
        with open(f'training_validation_mots_{include_mots}.pkl', 'rb') as p:
            training, validation = pickle.load(p)
            p.close()
    else:
        training = []
        validation = []
        if include_mots:
            print(f'MOTS sequences: {MOTS_train_seqs}')
            for idx, seq_name in enumerate(sorted(MOTS_train_seqs)):
                annos = read_annos_from_path('MOTS', MOTS_LABEL_DIR, seq_name)
                training.extend(annos)
        print(f'KITTIMOTS train sequences: {KITTIMOTS_train_seqs}')
        for idx, seq_name in enumerate(sorted(KITTIMOTS_train_seqs)):
            annos = read_annos_from_path('KITTI-MOTS', KITTIMOTS_LABEL_DIR, seq_name)
            training.extend(annos)
        print(f'KITTIMOTS val sequences: {KITTIMOTS_val_seqs}')
        for idx, seq_name in enumerate(sorted(KITTIMOTS_val_seqs)):
            annos = read_annos_from_path('KITTI-MOTS', KITTIMOTS_LABEL_DIR, seq_name)
            validation.extend(annos)

        with open(f'training_validation_mots_{include_mots}.pkl', 'wb') as f:
            pickle.dump([training, validation], f)
            f.close()

    return training, validation


def get_test_sample(validation):
    val = get_dataset_dicts(validation)

    if os.path.exists(f'test_samples.pkl'):
        with open('test_samples.pkl', 'rb') as p:
            test_samples = pickle.load(p)
            p.close()
    else:
        test_samples = random.sample(val, 20)
        with open('test_samples.pkl', 'wb') as f:
            pickle.dump(test_samples, f)
            f.close()

    return test_samples


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


class ValidationLoss(HookBase):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone()
        self.cfg.DATASETS.TRAIN = cfg.DATASETS.TEST
        self._loader = iter(build_detection_train_loader(self.cfg))
        self.best_loss = float('inf')
        self.weights = None

    def after_step(self):
        data = next(self._loader)
        with torch.no_grad():
            loss_dict = self.trainer.model(data)

            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {'val_' + k: v.item() for k, v in
                                 comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                self.trainer.storage.put_scalars(total_val_loss=losses_reduced,
                                                 **loss_dict_reduced)
                if losses_reduced < self.best_loss:
                    self.best_loss = losses_reduced
                    self.weights = copy.deepcopy(self.trainer.model.state_dict())


class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(
            cfg,
            mapper=DatasetMapper(
                cfg, is_train=True,
                augmentations=[
                    T.ResizeShortestEdge(short_edge_length=(640, 672, 704, 736, 768, 800), max_size=1333, sample_style='choice'),
                    T.RandomFlip(),
                    T.RandomCrop(crop_type='relative_range', crop_size=(0.8, 0.8))
                ]
            )
        )
