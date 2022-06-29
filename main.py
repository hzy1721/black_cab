import argparse
import logging
import cv2
import warnings
import torch
import os
from datasource import create_output_dir
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--channel', type=str,
                        help='channel id')
    parser.add_argument('--video', type=str,
                        help='video path')
    parser.add_argument('--frame_dir', type=str,
                        help='frames dir path')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu',
                        help='cuda device or cpu')
    parser.add_argument('--fps', type=int, default=25,
                        help='cv2.imshow fps')
    parser.add_argument('--show', action='store_true',
                        help='show video using cv2.imshow')
    parser.add_argument('--save', action='store_true',
                        help='save result frames')
    args = parser.parse_args()
    print(f'args: {args}')
    return args


args = parse_args()
out_dir = create_output_dir(args.channel, args.video, args.frame_dir)
logging.getLogger('faiss.loader').setLevel(logging.WARNING)
logging.getLogger('fastreid.utils.checkpoint').setLevel(logging.WARNING)
logging.getLogger('fastreid.engine.defaults').setLevel(logging.WARNING)
logging.getLogger('root.tracker').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
logging.getLogger('matplotlib.pyplot').setLevel(logging.WARNING)
# prod
logging.basicConfig(format='[%(levelname)s] [%(name)s] %(message)s',
                    level=logging.INFO,
                    filename=os.path.join(out_dir, 'log.txt'))
# dev
# logging.basicConfig(format='[%(levelname)s] [%(name)s] %(message)s',
#                     level=logging.DEBUG)
warnings.filterwarnings("ignore", category=UserWarning)

from datasource import get_img_generator
from detection import init_det_model, detect_private_vehicle
from tracking import init_track_model, track_vehicle, update_tracker_info, get_lost_track
from LPR import vehicle_LPR, plate_voting, final_voting
from db import save_database, vehicle_appear_count, get_db_warning
from utils import cv2PutChineseText

logger = logging.getLogger('main')


def main():
    init_det_model(args.device)
    init_track_model(args.device)
    generator = get_img_generator(args.channel, args.video, args.frame_dir)
    for idx, img in enumerate(generator):
        t1 = time.perf_counter()
        frame, warning_plates = black_cab_warning(img, visualize=args.show or args.save)
        if args.show:
            cv2.imshow('Test', frame)
            cv2.waitKey(1)
        if args.save:
            cv2.imwrite(os.path.join(out_dir, '%07d.jpg' % idx), frame)
        t2 = time.perf_counter()
        logger.debug(f'Time cost per frame: {t2 - t1} s ({ 1 / (t2 - t1)} FPS)')
    voted_plates = final_voting(tracker_info)
    save_database(voted_plates)
    # TODO: 黑/白名单
    appear_counts = vehicle_appear_count(voted_plates)
    warning_plates = get_db_warning(appear_counts)


tracker_info = {
    # track_id: {
    #     'plates': {
    #         plate1: 1,
    #         plate2: 2
    #     },
    #     'lost_frames': 1
    # }
}


def black_cab_warning(img, visualize=False):
    """
    黑出租预警单帧处理程序
    :param img: ndarray
    :param visualize:
    :return: frame(ndarray/None), logs([str])
    """
    dets = detect_private_vehicle(img)  # [[x1, y1, x2, y2, conf]]
    tracks = track_vehicle(img, dets)   # [[x1, y1, x2, y2, track_id]]
    plates = vehicle_LPR(img, tracks)   # [[track_id, x1, y1, x2, y2, plate]]
    update_tracker_info(tracker_info, tracks, plates)
    lost_track_plates = get_lost_track(tracker_info, tracks)
    voted_plates = plate_voting(lost_track_plates)
    save_database(voted_plates)
    # TODO: 黑/白名单
    appear_counts = vehicle_appear_count(voted_plates)
    warning_plates = get_db_warning(appear_counts)
    if visualize:
        frame = draw_frame(img, dets, tracks, plates)
        return frame, warning_plates
    return None, warning_plates


def draw_frame(img, dets, tracks, plates):
    frame = img.copy()
    for x1, y1, x2, y2, conf in dets:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), thickness=3)
    for x1, y1, x2, y2, track_id in tracks:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=3)
        cv2.putText(frame, str(track_id), (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), thickness=2)
    for track_id, x1, y1, x2, y2, plate in plates:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), thickness=3)
        # cv2.putText(frame, plate, (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), thickness=2)
        frame = cv2PutChineseText(frame, plate, (x1, y2 + 5), (255, 0, 0), 20)
    return frame


if __name__ == '__main__':
    main()
