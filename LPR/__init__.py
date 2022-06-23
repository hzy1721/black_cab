from .hyperlpr import HyperLPR_plate_recognition
import logging

logger = logging.getLogger(__name__)


def vehicle_LPR(img, tracks):
    """
    车牌识别
    :param img:
    :param tracks: [[x1, y1, x2, y2, track_id]]
    :return: [[track_id, x1, y1, x2, y2, plate]]
    """
    plates = []
    for x1, y1, x2, y2, track_id in tracks:
        img_vehicle = img[y1:y2, x1:x2]
        plate_list = HyperLPR_plate_recognition(img_vehicle)
        if plate_list:
            plate_list.sort(key=lambda e: e[1], reverse=True)
            plate, pconf, pbox = plate_list[0]
            conf_thresh = 0.7
            if pconf > conf_thresh:
                plates.append([track_id, pbox[0] + x1, pbox[1] + y1, pbox[2] + x1, pbox[3] + y1, plate])
    logger.debug(f'plates: {plates}')
    return plates


def plate_voting(track_plates):
    """

    :param track_plates: [[track_id, { plate1: 1, plate2: 2 }]]
    :return: [[track_id, votes, voted_plate]]
    """
    voted_plates = []
    for track_id, plates in track_plates:
        votes, voted_plate = single_voting(plates)
        logger.info(f'Tracker {track_id} 识别到车牌：{voted_plate}')
        logger.info(f'投票细节：{votes}')
        voted_plates.append([track_id, votes, voted_plate])
    return voted_plates


def single_voting(plates):
    max_plate_len = 8
    votes = []
    for i in range(max_plate_len):
        votes.append({})
    for plate, count in plates.items():
        for i in range(min(max_plate_len, len(plate))):
            c = plate[i]
            if c not in votes[i]:
                votes[i][c] = count
            else:
                votes[i][c] += count
    voted_plate = ''
    for digit_info in votes:
        max_voted = ''
        max_count = 0
        for digit, count in digit_info.items():
            if count > max_count:
                max_voted = digit
                max_count = count
        voted_plate += max_voted
    return votes, voted_plate


def final_voting(tracker_info):
    voted_plates = []
    for track_id in tracker_info:
        if not tracker_info[track_id]['plates']:
            continue
        votes, voted_plate = single_voting(tracker_info[track_id]['plates'])
        logger.info(f'Tracker {track_id} 识别到车牌：{voted_plate}')
        logger.info(f'投票细节：{votes}')
        voted_plates.append([track_id, votes, voted_plate])
    return voted_plates
