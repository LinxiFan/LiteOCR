'''
@author: jimfan
'''
import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import subprocess
from PIL import Image

from old_ocr import OCREngine

IMG = sys.argv[1]
COLOR = {
    'aquamarine': (127,255,212),
    'teal': (0, 128, 128),
    'gold': (255, 215, 0),
    'pink': (255, 105, 180)
}['pink']


def display(img):
    TMP = 'temp.png'
    cv2.imwrite(TMP, img)
    subprocess.call(['open', TMP])
    input('continue ...')
    os.remove(TMP)

engine = OCREngine()
# img = np.asarray(Image.open(IMG)) # same as cv2.imread
img_original = cv2.imread(IMG)
page = engine.recognize(IMG)

# display words, lines, blocks segmentation
for elem in ['words', 'lines', 'blocks']:
    print('='*10, elem, 'segmentation', '='*10)
    img = np.copy(img_original)
    elems = getattr(page, elem)
    for i, elem in enumerate(elems):
        # Warning: color is in BGR
        if str(elem).strip():
            print(i, '\t', elem)
            cv2.rectangle(img, *elem.box.corners, color=COLOR, thickness=4)
        else:
            print(i, '\t N/A')

    # Warning: OpenCV reads np array in reverse order: 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    display(img)