import os
import cv2
import random
import string
import pickle
from glob import glob
from tqdm import tqdm

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator
from detectron2.structures import BoxMode

from pycocotools import coco

DATA_DIR = '/home/mcv/datasets/KITTI-MOTS/training/image_02'
WORK_DIR = '~/week3/output/inference'

KITTIMOTS_IMAGE_DIR = '/home/mcv/datasets/KITTI-MOTS/training/image_02'
KITTIMOTS_LABEL_DIR = '/home/mcv/datasets/KITTI-MOTS/training/instances_txt'

KITTIMOTS_train_seqs = ['0000', '0001', '0003', '0004', '0005', '0009', '0011', '0012', '0015', '0017', '0019', '0020']
KITTIMOTS_val_seqs = ['0002', '0006', '0007', '0008', '0010', '0013', '0014', '0016', '0018']

KITTIMOTS_CATEGORIES = {
    'Car': 1,
    'Pedestrian': 2,
}
COCO_CATEGORIES = {
    1: 2,
    2: 0
}


def add_rec(i_path, l_path, sequences, fm):
    d = []
    for seq in sequences:
        seq_image_paths = sorted(glob(i_path + os.sep + seq + os.sep + fm))
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
                d.append(record)
    return d


def get_KITTIMOTS_dicts(phase='train'):
    kittimots_sequences = KITTIMOTS_train_seqs if phase == 'train' else KITTIMOTS_val_seqs
    dataset_dicts = []
    dataset_dicts += add_rec(KITTIMOTS_IMAGE_DIR, KITTIMOTS_LABEL_DIR, kittimots_sequences, '*.png')
    return dataset_dicts


def get_samples(num_samples=10):
    sample_images = []
    for _ in range(num_samples):
        seq = random.sample(KITTIMOTS_val_seqs, 1)[0]
        seq_path = os.path.join(KITTIMOTS_IMAGE_DIR, seq)
        image = random.sample(os.listdir(seq_path), 1)[0]
        image_path = os.path.join(seq_path, image)
        sample_images.append(image_path)

    return sample_images


def inference_evaluate(model_name, model_config, sample_images, test_dicts, score_thresh=0.5):
    token = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    save_dir = WORK_DIR + os.sep + model_name + os.sep + token
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_config))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_config)
    predictor = DefaultPredictor(cfg)

    for i, s_img in enumerate(sample_images):
        img = cv2.imread(s_img)
        outputs = predictor(img)
        # scores = outputs['instances'].scores
        # classes = outputs['instances'].pred_classes
        # print(f'scores: {scores}')
        # print(f'classes: {classes}')
        v = Visualizer(
            img[:, :, ::-1],
            metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            scale=0.8,
            instance_mode=ColorMode.IMAGE
        )
        out = v.draw_instance_predictions(outputs['instances'].to('cpu'))
        cv2.imwrite(os.path.join(save_dir, f'{str(i)}_.png'),
                    out.get_image()[:, :, ::-1])

    predictions_path = os.path.join(save_dir, "predictions.pkl")
    if os.path.exists(predictions_path):
        predictions = pickle.load(open(predictions_path, "rb"))
    else:
        print('Using Model to predict on input')
        predictions = []
        for i, input_test in tqdm(enumerate(KITTIMOTS_test_dicts), desc='Making prediction on validation set'):
            img_path = input_test['file_name']
            img = cv2.imread(img_path)
            prediction = predictor(img)
            predictions.append(prediction)
        pickle.dump(predictions, open(predictions_path, "wb"))

    print('Predictions length ' + str(len(predictions)))
    print('Inputs length ' + str(len(test_dicts)))

    print('Evaluating......')
    evaluator = COCOEvaluator('KITTIMOTS_test', cfg, False, output_dir='./drive/MyDrive/output' + os.sep + model_name)
    evaluator.reset()
    evaluator.process(test_dicts, predictions)
    evaluator.evaluate()


if __name__ == '__main__':
    """ 
    Model name and config file:
        - Name: FASTER_RCNN_X_101 ------ config file: COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml
        - Name: FASTER_RCNN_R_101 ------ config file: COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml
        - Name: FASTER_RCNN_R_50_DC5 ------ config file: COCO-Detection/faster_rcnn_R_50_DC5_3x.yaml
        - Name: FASTER_RCNN_R_50_C4 ------ config file: COCO-Detection/faster_rcnn_R_50_C4_3x.yaml
        - Name: RETINA_NET_R_50 ------ config file: COCO-Detection/retinanet_R_50_FPN_3x.yaml
        - Name: RETINA_NET_R_101 ------ config file: COCO-Detection/retinanet_R_101_FPN_3x.yaml
    """

    m_names = ['FASTER_RCNN_X_101',
               'FASTER_RCNN_R_101',
               'FASTER_RCNN_R_50_DC5',
               'FASTER_RCNN_R_50_C4',
               'RETINA_NET_R_50',
               'RETINA_NET_R_101']
    m_configs = ['COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml',
                 'COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml',
                 'COCO-Detection/faster_rcnn_R_50_DC5_3x.yaml',
                 'COCO-Detection/faster_rcnn_R_50_C4_3x.yaml',
                 'COCO-Detection/retinanet_R_50_FPN_3x.yaml',
                 'COCO-Detection/retinanet_R_101_FPN_3x.yaml']
    thresh = 0.5  # 0.7 for RetinaNet
    samples = get_samples(10)

    print('Register Dataset.')
    for d in ['train', 'test']:
        DatasetCatalog.register("KITTIMOTS_" + d, lambda d=d: get_KITTIMOTS_dicts(d))
        MetadataCatalog.get("KITTIMOTS_" + d).set(thing_classes=list(KITTIMOTS_CATEGORIES.keys()))
    kitti_metadata = MetadataCatalog.get("KITTIMOTS_train")
    print(kitti_metadata)

    KITTIMOTS_test_dicts = get_KITTIMOTS_dicts('test')

    for (m_name, m_config) in zip(m_names, m_configs):
        inference_evaluate(m_name, m_config, samples, KITTIMOTS_test_dicts, thresh)
