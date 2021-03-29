import os
import cv2
import string
import random

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.data import DatasetCatalog
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.evaluation import COCOEvaluator
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

from .utils import get_dataset_dicts
from .utils import KITTIMOTS_CATEGORIES

MOTS_train_seqs = ['0005', '0009', '0011']
MOTS_val_seqs = ['0002']
KITTIMOTS_train_seqs = ['0000', '0001', '0003', '0004', '0005', '0009', '0011', '0012', '0015', '0017', '0019', '0020']
KITTIMOTS_val_seqs = ['0002', '0006', '0007', '0008', '0010', '0013', '0014', '0016', '0018']

WORK_DIR = '/home/group06/week4/output/inference'


def inference(model_config, sample_images, save_dir, threshold):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_config))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_config)
    predictor = DefaultPredictor(cfg)

    for i, s_img in enumerate(sample_images):
        img = cv2.imread(s_img)
        outputs = predictor(img)
        v = Visualizer(
            img[:, :, ::-1],
            metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            scale=0.8,
            instance_mode=ColorMode.IMAGE
        )
        out = v.draw_instance_predictions(outputs['instances'].to('cpu'))
        cv2.imwrite(os.path.join(save_dir, f'{str(i)}.png'), out.get_image()[:, :, ::-1])


def inference_evaluate(model_name, model_config, sample_images, score_thresh=0.5):
    token = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    save_dir = WORK_DIR + os.sep + model_name + os.sep + token
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        print('Created path ', save_dir)

    # Load model and configuration
    print('Loading Model...')
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_config))
    cfg.DATASETS.TRAIN = ('KITTIMOTS_train',)
    cfg.DATASETS.TEST = ('KITTIMOTS_test',)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.OUTPUT_DIR = save_dir
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_config)

    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

    # Evaluation
    print('Evaluating...')
    evaluator = COCOEvaluator('KITTIMOTS_test', cfg, False, output_dir=save_dir)
    trainer = DefaultTrainer(cfg)
    trainer.test(cfg, model, evaluators=[evaluator])

    # Visualization
    print('Visualizing...')
    inference(model_config, sample_images, save_dir, score_thresh)


if __name__ == '__main__':
    """
    Model name and config file:
        -   "MaskRCNN_R_50_C4", "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml"
        -   "MaskRCNN_R_50_DC5", "COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml"
        -   "MaskRCNN_R_50_FPN", "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        -   "MaskRCNN_R_101_C4", "COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml"
        -   "MaskRCNN_R_101_DC5", "COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml"
        -   "MaskRCNN_R_101_FPN", "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
        -   "MaskRCNN_R_50_FPN_Cityscapes", "Cityscapes/mask_rcnn_R_50_FPN.yaml"
    """

    model_names = [
        'MaskRCNN_R_50_C4',
        'MaskRCNN_R_50_DC5',
        'MaskRCNN_R_50_FPN',
        'MaskRCNN_R_101_C4',
        'MaskRCNN_R_101_DC5',
        'MaskRCNN_R_101_FPN',
        'MaskRCNN_R_50_FPN_Cityscapes'
    ]
    model_configs = [
        'COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml',
        'COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml',
        'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml',
        'COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml',
        'COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml',
        'COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml',
        'Cityscapes/mask_rcnn_R_50_FPN.yaml'
    ]

    samples = [
        '/home/mcv/datasets/KITTI-MOTS/training/image_02/0002/000082.png',
        '/home/mcv/datasets/KITTI-MOTS/training/image_02/0002/000019.png',
        '/home/mcv/datasets/KITTI-MOTS/training/image_02/0006/000217.png',
        '/home/mcv/datasets/KITTI-MOTS/training/image_02/0006/000128.png',
        '/home/mcv/datasets/KITTI-MOTS/training/image_02/0007/000416.png',
        '/home/mcv/datasets/KITTI-MOTS/training/image_02/0007/000207.png',
        '/home/mcv/datasets/KITTI-MOTS/training/image_02/0008/000275.png',
        '/home/mcv/datasets/KITTI-MOTS/training/image_02/0008/000332.png',
        '/home/mcv/datasets/KITTI-MOTS/training/image_02/0010/000164.png',
        '/home/mcv/datasets/KITTI-MOTS/training/image_02/0010/000226.png',
        '/home/mcv/datasets/KITTI-MOTS/training/image_02/0013/000153.png',
        '/home/mcv/datasets/KITTI-MOTS/training/image_02/0013/000028.png',
        '/home/mcv/datasets/KITTI-MOTS/training/image_02/0014/000028.png',
        '/home/mcv/datasets/KITTI-MOTS/training/image_02/0014/000025.png',
        '/home/mcv/datasets/KITTI-MOTS/training/image_02/0016/000193.png',
        '/home/mcv/datasets/KITTI-MOTS/training/image_02/0016/000081.png',
        '/home/mcv/datasets/KITTI-MOTS/training/image_02/0018/000190.png',
        '/home/mcv/datasets/KITTI-MOTS/training/image_02/0018/000106.png'
    ]

    # Register Data
    print('Registering Datasets...')
    for d in ['train', 'test']:
        DatasetCatalog.register("KITTIMOTS_" + d, lambda d=d: get_dataset_dicts(d, False))
        MetadataCatalog.get("KITTIMOTS_" + d).set(thing_classes=list(KITTIMOTS_CATEGORIES.keys()))
    kitti_metadata = MetadataCatalog.get("KITTIMOTS_train")
    print(kitti_metadata)

    thresh = 0.5

    for (m_name, m_config) in zip(model_names, model_configs):
        print(f'Running evaluation with model: {m_name}, config: {m_config}')
        inference_evaluate(m_name, m_config, samples, thresh)
