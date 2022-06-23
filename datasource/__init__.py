import logging

from .generator import channel_generator, video_generator, frame_dir_generator

logger = logging.getLogger(__name__)


def get_img_generator(channel, video, frame_dir):
    if channel:
        logger.info(f'数据源(channel): {channel}')
        return channel_generator(channel)
    if video:
        logger.info(f'数据源(视频文件): {video}')
        return video_generator(video)
    if frame_dir:
        logger.info(f'数据源(帧目录): {frame_dir}')
        return frame_dir_generator(frame_dir)
    logger.error('Datasource not provided')
    raise 'Datasource not provided'
