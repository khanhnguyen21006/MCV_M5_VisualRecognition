import os
import cv2
import random
import string

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator

from utils import get_dataset_dicts
from utils import MyTrainer

MOTS_train_seqs = ['0005', '0009', '0011']
MOTS_val_seqs = ['0002']
KITTIMOTS_train_seqs = ['0000', '0001', '0003', '0004', '0005', '0009', '0011', '0012', '0015', '0017', '0019', '0020']
KITTIMOTS_val_seqs = ['0002', '0006', '0007', '0008', '0010', '0013', '0014', '0016', '0018']

model_name = 'MaskRCNN_R_50_FPN'
model_config = 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'

exp_names = ['lr_00001', 'lr_00002']
batch_size = 4  # [4, 6, >8]
lrs = [0.0001, 0.0002]  # [0.0001, 0.0002, 0.0003, 0.001, 0.003]
num_iter = 3000  # [1600, 2000, 3000]
num_roi_per_image = 256  # [128, 256, 512]
thresh_test = 0.5  # [0.5, 0.6, 0.7, 0.8, 0.9]
OUTPUT_PATH = '/home/group06/week4/output/training'

KITTIMOTS_TRAIN_CATEGORIES = {
    'Car': 1,
    'Pedestrian': 2
}

if __name__ == '__main__':
    # Register Data
    print('Registering Datasets...')
    for d in ['train', 'test']:
        DatasetCatalog.register("KITTIMOTS_" + d, lambda d=d: get_dataset_dicts(d, True))
        MetadataCatalog.get("KITTIMOTS_" + d).set(thing_classes=list(KITTIMOTS_TRAIN_CATEGORIES.keys()))
    kitti_metadata = MetadataCatalog.get("KITTIMOTS_train")
    print(kitti_metadata)

    for (exp_name, lr) in zip(exp_names, lrs):
        token = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        SAVE_PATH = OUTPUT_PATH + os.sep + model_name + os.sep + exp_name + os.sep + token
        print(f'Token {token}, experiment name {exp_name}, batch_size {batch_size}, lr {lr}, num_iter {num_iter}, num_roi_per_image {num_roi_per_image}, thresh_test {thresh_test}')
        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH, exist_ok=True)
            print('Created path ', SAVE_PATH)

        # Load model using config file and specify hyper-parameters
        print('Loading Mask R-CNN Model...')
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(model_config))
        cfg.DATASETS.TRAIN = ('KITTIMOTS_train',)
        cfg.DATASETS.TEST = ('KITTIMOTS_test',)
        cfg.TEST.EVAL_PERIOD = 200
        cfg.DATALOADER.NUM_WORKERS = 0
        cfg.OUTPUT_DIR = SAVE_PATH
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_config)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh_test
        cfg.SOLVER.IMS_PER_BATCH = batch_size
        cfg.SOLVER.BASE_LR = lr
        cfg.SOLVER.MAX_ITER = num_iter
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = num_roi_per_image
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

        # Start training model
        print('Training...')
        trainer = MyTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()
        print('Training Done.')

        # Evaluation process
        print('Evaluating...')
        print('COCO EVALUATOR....')
        evaluator = COCOEvaluator('KITTIMOTS_test', cfg, False, output_dir=SAVE_PATH)
        trainer.test(cfg, trainer.model, evaluators=[evaluator])

        # Qualitative results
        print('Inference on trained model...')
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh_test
        predictor = DefaultPredictor(cfg)
        kitti_test_dicts = get_dataset_dicts('test', True)
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
            cv2.imwrite(os.path.join(SAVE_PATH, f'{str(i)}.png'), v.get_image()[:, :, ::-1])
