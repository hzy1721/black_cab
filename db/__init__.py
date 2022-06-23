import os
import sqlite3
from datetime import datetime, timedelta
import logging

ROOT = os.path.abspath(os.path.dirname(__file__))
DATABASE = os.path.join(ROOT, 'garage.db')
conn = sqlite3.connect(DATABASE)
cursor = conn.cursor()

logger = logging.getLogger(__name__)


def save_database(voted_plates):
    time_str = datetime.now().strftime('%Y%m%d%H%M%S')
    for _, _, plate in voted_plates:
        sql = 'INSERT INTO vehicle_record (time, plate) VALUES (?, ?)'
        cursor.execute(sql, (time_str, plate))
        conn.commit()
        logger.info(f'保存车辆出现信息到数据库: ({time_str}, {plate})')


def vehicle_appear_count(voted_plates):
    appear_counts = []
    time_interval = timedelta(days=1)
    time_bound = datetime.now() - time_interval
    time_bound_str = time_bound.strftime('%Y%m%d%H%M%S')
    for _, _, plate in voted_plates:
        sql = f'SELECT COUNT(*) FROM vehicle_record WHERE plate = ? AND time > ?'
        cursor.execute(sql, (plate, time_bound_str))
        count = cursor.fetchone()[0]
        appear_counts.append([plate, count])
        logger.info(f'{plate} 最近 1 天出现 {count} 次')
    return appear_counts


def get_db_warning(appear_counts):
    count_thresh = 3
    warning_plates = []
    for plate, count in appear_counts:
        if count >= count_thresh:
            warning_plates.append(plate)
            logger.warning(f'车辆 {plate} 今天出现了 {count} 次，有可能在从事非法载客营运')
    return warning_plates
