'''
@author: jimfan
'''
import sys
from liteocr.ocr import OCREngine, load_img, draw_rect, draw_text, disp

engine = OCREngine(all_unicode=False)

image_file = sys.argv[1]
img = load_img(image_file)

for text, box, conf in engine.recognize(image_file):
    print(box, '\tconf =', conf, '\t\t', text)
    draw_rect(img, box)
    draw_text(img, text, box, color='bw')
disp(img, pause=False)
