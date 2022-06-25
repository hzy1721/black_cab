import os
import sys

ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

import torch
from mmdet.apis import init_detector, inference_detector
import logging

logger = logging.getLogger(__name__)

config_path = f'{ROOT}/configs/yolo/yolov3_d53_mstrain-608_273e_coco.py'
checkpoint_path = f'{ROOT}/checkpoints/yolov3_d53_mstrain-608_273e_coco/latest.pth'
det_model = None


def init_det_model(device):
    global det_model
    det_model = init_detector(config_path, checkpoint_path, device=device)
    logger.info(f'私家车检测模型加载完成 ({device})')


def detect_private_vehicle(img):
    """
    私家车目标检测
    :param img: ndarray
    :return: [x1, y1, x2, y2, conf]
    """
    vehicle_dets = inference_detector(det_model, img)
    private_dets = vehicle_dets[0]
    final_dets = []
    conf_thresh = 0.7
    h, w = img.shape[:2]
    for x1, y1, x2, y2, conf in private_dets:
        if conf > conf_thresh:
            final_dets.append([
                int(max(0, min(x1, w))),
                int(max(0, min(y1, h))),
                int(max(0, min(x2, w))),
                int(max(0, min(y2, h))),
                conf
            ])
    logger.debug(f'dets: {final_dets}')
    return final_dets
