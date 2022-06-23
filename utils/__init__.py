import cv2
from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np

ROOT = os.path.abspath(os.path.dirname(__file__))


def cv2PutChineseText(img, text, org, color, size):
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(f'{ROOT}/SimHei.ttf', size, encoding='utf-8')
    draw.text(org, text, color, font)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
