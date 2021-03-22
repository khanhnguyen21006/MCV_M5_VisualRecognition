import os
import cv2
import random
import string

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog


DATA_DIR = '/home/mcv/datasets/MIT_split/test'
WORK_DIR = '~/week3'


def inference(model_name, model_config, num_sample=10):
    save_dir = os.path.join(WORK_DIR + os.sep + 'inference', model_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    token = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))

    category = random.sample(os.listdir(DATA_DIR), 1)[0]
    print(f'Class: {category}')
    cat_path = os.path.join(DATA_DIR, category)
    sample_images = random.sample(os.listdir(cat_path), num_sample)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_config))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_config)
    predictor = DefaultPredictor(cfg)

    for i, s_img in enumerate(sample_images):
        img = cv2.imread(os.path.join(cat_path, s_img))
        outputs = predictor(img)
        v = Visualizer(
            img[:, :, ::-1],
            metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            scale=0.8,
            instance_mode=ColorMode.IMAGE
        )
        out = v.draw_instance_predictions(outputs['instances'].to('cpu'))

        cv2.imwrite(os.path.join(save_dir, f'{model_name}_{token}_{str(i)}_.png'),
                    out.get_image()[:, :, ::-1])


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

    m_name = 'FASTER_RCNN_X_101'
    m_config = 'COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml'
    inference(m_name, m_config, num_sample=10)
