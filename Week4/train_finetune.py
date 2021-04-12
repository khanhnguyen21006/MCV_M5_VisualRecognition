import os
import cv2
import random
import string

from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator

from utils import read_data, get_dataset_dicts, get_test_sample
from utils import ValidationLoss, CustomTrainer
from utils import KITTIMOTS_CATEGORIES


model_name = 'MaskRCNN_R_50_FPN_Cityscapes'
model_config = 'Cityscapes/mask_rcnn_R_50_FPN.yaml'

exp_names = ['nms_top_train_1000']  # 'coco_kitti_mots'
batch_size = 4  # batch_sizes = [2, 4, 6] for exp_name, batch_size in zip(exp_names, batch_sizes):
lr = 0.001  # [0.0001, 0.00025, 0.001, 0.003] lr = 0.00025
num_iter = 5000  # [1600, 2000, 3000]
num_roi_per_image = 256  # [128, 256, 512]
thresh_test = 0.5  # [0.5, 0.6, 0.7, 0.8, 0.9]
rpm_iou_thresh = [0.4, 0.6]
nms_top_train = 1000
OUTPUT_PATH = '/home/group06/week4/output/training'


if __name__ == '__main__':
    print(f'Loading Train/Validation datasets...')
    training, validation = read_data(include_mots=True)
    print(f'Number of initial training image: {len(training)}')
    print(f'Number of initial validation image: {len(validation)}')

    # Register Data
    print('Registering Datasets...')
    def kittimots_train(): return get_dataset_dicts(training)
    def kittimots_test(): return get_dataset_dicts(validation)
    DatasetCatalog.register('KITTIMOTS_train', kittimots_train)
    MetadataCatalog.get('KITTIMOTS_train').set(thing_classes=list(KITTIMOTS_CATEGORIES.keys()))
    DatasetCatalog.register('KITTIMOTS_test', kittimots_test)
    MetadataCatalog.get('KITTIMOTS_test').set(thing_classes=list(KITTIMOTS_CATEGORIES.keys()))
    kittimots_metadata = MetadataCatalog.get("KITTIMOTS_train")
    print(kittimots_metadata)

    for exp_name in exp_names:
        token = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        SAVE_PATH = OUTPUT_PATH + os.sep + model_name + os.sep + exp_name + os.sep + token
        print(f'Token {token}, experiment name {exp_name}, batch_size {batch_size}, lr {lr}, num_iter {num_iter}, nms_top_train {nms_top_train}, thresh_test {thresh_test}')
        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH, exist_ok=True)
            print('Created path ', SAVE_PATH)

        # Load model using config file and specify hyper-parameters
        print('Loading Mask R-CNN Model...')
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(model_config))
        cfg.DATASETS.TRAIN = ('KITTIMOTS_train',)
        cfg.DATASETS.TEST = ('KITTIMOTS_test',)
        cfg.DATALOADER.NUM_WORKERS = 0
        cfg.OUTPUT_DIR = SAVE_PATH
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_config)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh_test
        cfg.SOLVER.IMS_PER_BATCH = batch_size
        cfg.SOLVER.BASE_LR = lr
        cfg.SOLVER.MAX_ITER = num_iter
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = num_roi_per_image
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
        cfg.TEST.SCORE_THRESH = thresh_test
        # cfg.MODEL.RPN.IOU_THRESHOLDS = rpm_iou_thresh
        cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = nms_top_train

        # Start training model
        print('Training...')
        trainer = DefaultTrainer(cfg)
        # trainer = CustomTrainer(cfg)
        val_loss = ValidationLoss(cfg)
        trainer.register_hooks([val_loss])
        trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]
        trainer.resume_or_load(resume=False)
        trainer.train()
        print('Training Done.')

        # Evaluation process
        print('Evaluating...')
        print('COCO EVALUATOR....')
        evaluator = COCOEvaluator('KITTIMOTS_test', cfg, False, output_dir=SAVE_PATH)
        trainer.model.load_state_dict(val_loss.weights)
        trainer.test(cfg, trainer.model, evaluators=[evaluator])
        
        # Qualitative results
        print('Apply trained model on inference...')
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh_test
        predictor = DefaultPredictor(cfg)
        samples = get_test_sample(validation)
        for i, d in enumerate(samples):
            im = cv2.imread(d['file_name'])
            outputs = predictor(im)
            v = Visualizer(
                im[:, :, ::-1],
                metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                scale=0.8,
                instance_mode=ColorMode.IMAGE
            )
            v = v.draw_instance_predictions(outputs['instances'].to('cpu'))
            cv2.imwrite(os.path.join(SAVE_PATH, f'{str(i)}.png'), v.get_image()[:, :, ::-1])
