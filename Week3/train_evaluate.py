import os
import cv2
import random
import string
from glob import glob

# import some common detectron2 utilities
from detectron2.utils.logger import setup_logger

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator
from detectron2.structures import BoxMode

from pycocotools import coco
from detectron2.data import DatasetMapper, build_detection_test_loader

from MCV_M5_VisualRecognition.Week2.utils.LossEvalHook import LossEvalHook

setup_logger()

KITTIMOTS_IMAGE_DIR = '/home/mcv/datasets/KITTI-MOTS/training/image_02'
KITTIMOTS_LABEL_DIR = '/home/mcv/datasets/KITTI-MOTS/instances_txt'
KITTIMOTS_TEST_DIR = '/home/mcv/datasets/KITTI-MOTS/testing/image_02'
MOTS_IMAGE_DIR = '/home/mcv/datasets/MOTSChallenge/train/images'
MOTS_LABEL_DIR = '/home/mcv/datasets/MOTSChallenge/train/instances_txt'

MOTS_train_seqs = ['0005', '0009', '0011']
MOTS_val_seqs = ['0002']
KITTIMOTS_train_seqs = ['0000', '0001', '0003', '0004', '0005', '0009', '0011', '0012', '0015', '0017', '0019', '0020']
KITTIMOTS_val_seqs = ['0002', '0006', '0007', '0008', '0010', '0013', '0014', '0016', '0018']

model_name = 'FasterRCNN_X_101'
model_config = 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'

exp_name = 'lr_0001'
NUM_IMAGES = 7266
batch_size = 6  # [4, 6, >8]
lr = 0.001  # [0.0001, 0.0003, 0.001, 0.003]
num_iter = 1600  # [1600, 2000, 3000]
OUTPUT_PATH = '/home/group06/week3/output'

KITTIMOTS_CATEGORIES = {
    'Car': 1,
    'Pedestrian': 2,
}
COCO_CATEGORIES = {
    1: 2,
    2: 0
}


def add_records(i_path, l_path, sequences, fmt):
    records = []
    for seq in sequences:
        seq_image_paths = sorted(glob(i_path + os.sep + seq + os.sep + fmt))
        num_frame = len(seq_image_paths)
        seq_label_path = l_path + os.sep + seq + '.txt'
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
                record['image_id'] = fidx
                record['height'] = height
                record['width'] = width

                annos = []
                for fobj in fobjs:
                    rle = {
                        'counts': fobj[-1].strip(),
                        'size': [height, width]
                    }
                    # category = COCO_CATEGORIES[int(fobj[2][0])]
                    category = int(fobj[2][0]) - 1
                    bbox = coco.maskUtils.toBbox(rle).tolist()
                    bbox[2] += bbox[0]
                    bbox[3] += bbox[1]
                    bbox = [int(item) for item in bbox]
                    anno = {
                        "bbox": bbox,
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "category_id": category
                    }
                    annos.append(anno)
                record["annotations"] = annos
                records.append(record)
    return records


def get_merge_dicts(phase='train'):
    mots_sequences = MOTS_train_seqs if phase == 'train' else MOTS_val_seqs
    kittimots_sequences = KITTIMOTS_train_seqs if phase == 'train' else KITTIMOTS_val_seqs
    dataset_dicts = []
    dataset_dicts += add_records(MOTS_IMAGE_DIR, MOTS_LABEL_DIR, mots_sequences, '*.jpg')
    dataset_dicts += add_records(KITTIMOTS_IMAGE_DIR, KITTIMOTS_LABEL_DIR, kittimots_sequences, '*.png')

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

    token = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    SAVE_PATH = OUTPUT_PATH + os.sep + model_name + os.sep + exp_name + os.sep + token
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH, exist_ok=True)
        print('created path ', SAVE_PATH)

    print("Registering datasets...")
    for d in ['train', 'test']:
        DatasetCatalog.register("merge_" + d, lambda d=d: get_merge_dicts(d))
        MetadataCatalog.get("merge_" + d).set(thing_classes=[cat for cat, _ in KITTIMOTS_CATEGORIES.items()])
    kitti_metadata = MetadataCatalog.get("merge_train")
    print(kitti_metadata)

    # Load model using config file and specify hyper-parameters
    print('Loading Faster R-CNN Model...')
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_config))
    cfg.DATASETS.TRAIN = ('merge_train',)
    cfg.DATASETS.TEST = ('merge_test',)
    cfg.TEST.EVAL_PERIOD = 200
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.OUTPUT_DIR = SAVE_PATH
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_config)
    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.SOLVER.BASE_LR = lr
    cfg.SOLVER.MAX_ITER = num_iter
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

    # Start training model
    print('Training...')
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    print('Training Done.')

    # Evaluation process
    print('Evaluating...')
    print('COCO EVALUATOR....')
    evaluator = COCOEvaluator('merge_test', cfg, False, output_dir=SAVE_PATH)
    trainer.test(cfg, trainer.model, evaluators=[evaluator])

    # Qualitative results
    print('Inference on trained model')
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    predictor = DefaultPredictor(cfg)
    kitti_test_dicts = get_merge_dicts('test')
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
        cv2.imwrite(os.path.join(SAVE_PATH, f'{str(i)}.png'),
                    v.get_image()[:, :, ::-1])
