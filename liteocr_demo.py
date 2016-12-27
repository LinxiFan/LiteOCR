'''
@author: jimfan
'''
import sys
from liteocr import OCREngine, load_img, draw_rect, draw_text, disp

engine = OCREngine(all_unicode=False)

img = load_img(sys.argv[1])
for text, box, conf in engine.recognize(sys.argv[1]):
    print(box, '\tconf =', conf, '\t\t', text)
    draw_rect(img, box)
    draw_text(img, text, box, color='bw')
disp(img, pause=False)