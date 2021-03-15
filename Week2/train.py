import os
import cv2
import random
import numpy as np
import string
import matplotlib.pyplot as plt
from glob import glob


from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader, DatasetMapper
from detectron2.evaluation import COCOEvaluator
from detectron2.structures import BoxMode
from detectron2.utils.logger import setup_logger

from MCV_M5_VisualRecognition.Week2.utils.LossEvalHook import LossEvalHook

setup_logger()

train_val_split = 0.8

NUM_IMAGES = 7481
SAVE_PATH = '~/week2/output'

KITTI_DATA_DIR = '/home/mcv/datasets/KITTI/'
KITTI_TRAIN_IMAGES = os.path.join(KITTI_DATA_DIR, 'data_object_image_2/training/image_2')
KITTI_TRAIN_LABELS = os.path.join(KITTI_DATA_DIR, 'training/label_2')

CATEGORIES = {
        'Car': 0,
        'Van': 1,
        'Truck': 2,
        'Pedestrian': 3,
        'Person_sitting': 4,
        'Cyclist': 5,
        'Tram': 6,
        'Misc': 7,
        'DontCare': 8
    }


def get_kitti_dicts(phase='train'):
    img_paths = np.array(sorted(glob(KITTI_TRAIN_IMAGES + os.sep + '*.png')))
    label_paths = np.array(sorted(glob(KITTI_TRAIN_LABELS + os.sep + '*.txt')))

    indices = np.arange(len(img_paths))
    split_point = int(indices.shape[0] * train_val_split)
    train_indices = indices[:split_point]
    test_indices = indices[split_point:]

    if phase == 'train':
        image_paths = img_paths[train_indices]
        label_paths = label_paths[train_indices]
    else:
        image_paths = img_paths[test_indices]
        label_paths = label_paths[test_indices]

    dataset_dicts = []
    for idx, (image_path, label_path) in enumerate(zip(image_paths, label_paths)):
        record = {}

        filename = os.path.join(image_path)
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        objs = []
        with open(label_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            items = line.split(' ')
            category = CATEGORIES[items[0]]
            bbox = [float(items[4]), float(items[5]), float(items[6]), float(items[7])]
            obj = {
                "bbox": bbox,
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": category
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, LossEvalHook(
            cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg, True)
            )
        ))
        return hooks


if __name__ == '__main__':
    model_name = 'FasterRCNN_R_101'
    model_config = 'COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'

    print("Registering datasets...")
    for d in ['train', 'test']:
        DatasetCatalog.register("KITTI_" + d, lambda d=d: get_kitti_dicts(d))
        MetadataCatalog.get("KITTI_" + d).set(thing_classes=[cat for cat, _ in CATEGORIES.items()])
    kitti_metadata = MetadataCatalog.get("KITTI_train")
    print(kitti_metadata)

    # Load model using config file and specify hyperparameters
    print('Loading Faster R-CNN Model...')
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_config))
    cfg.DATASETS.TRAIN = ('KITTI_train',)
    cfg.DATASETS.TEST = ('KITTI_test',)
    cfg.TEST.EVAL = 100
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.OUTPUT_DIR = SAVE_PATH
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_config)
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = NUM_IMAGES // cfg.SOLVER.IMS_PER_BATCH + 1
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 9

    # Start training model
    print('Training...')
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    print('Training finished.')

    # Evaluation process
    print('Evaluating...')

    token = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    os.makedirs(SAVE_PATH + os.sep + token, exist_ok=True)

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    predictor = DefaultPredictor(cfg)
    kitti_test_dicts = get_kitti_dicts('test')
    for i, d in enumerate(random.sample(kitti_test_dicts, 5)):
        im = cv2.imread(d['file_name'])
        outputs = predictor(im)
        v = Visualizer(
            im[:, :, ::-1],
            metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            scale=0.8,
            instance_mode=ColorMode.IMAGE
        )
        v = v.draw_instance_predictions(outputs['instances'].to('cpu'))
        cv2.imwrite(os.path.join(SAVE_PATH + os.sep + token, f'{str(i)}_.png'),
                    v.get_image()[:, :, ::-1])

    print('COCO EVALUATOR....')
    evaluator = COCOEvaluator('KITTI_test', cfg, False, output_dir=SAVE_PATH + os.sep + token)
    trainer.test(cfg, trainer.model, evaluators=[evaluator])

